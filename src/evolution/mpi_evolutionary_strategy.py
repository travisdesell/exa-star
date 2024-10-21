from abc import abstractmethod
from dataclasses import field
from enum import Enum
import sys
import traceback
import types
from typing import Any, Dict, Optional, Self, Tuple, Type

import dill

from config import configclass
from dataset import Dataset
from evolution.evolutionary_strategy import EvolutionaryStrategy, EvolutionaryStrategyConfig
from genome import Genome
from evolution.init_task import InitTask, InitTaskConfig
from util.typing import overrides

import loguru
from loguru import logger
from mpi4py import MPI
import numpy as np


class Tags(Enum):
    """
    All of the various MPI tags that are used to send messages between processes.
    """
    # Indicates that a message contains a list of tasks for the worker to execute.
    TASK = 0

    # Indicates that a message contains a list of results for the master process. Should only be sent by workers to the
    # master process.
    RESULT = 1

    # Indicates that the receiving worker process should safely close.
    FINALIZE = 2

    # Sent by a worker to the master process to begin the asynchronous exchange of tasks and results.
    INITIALIZE = 3

    # Sent by a worker process to the master process; signals that an exception occurred and the master should begint
    # the process of safely shutting down.
    FAILURE = 4


class MPIEvolutionaryStrategy[G: Genome, D: Dataset](EvolutionaryStrategy[G, D]):
    """
    Base class for any evolutionary strategy that uses MPI. Provides a few utility methods and sets up the log as well
    as some MPI data.
    """

    def __init__(self, init_tasks: Dict[str, InitTask[Self]], **kwargs) -> None:
        super().__init__(**kwargs)

        # default MPI communicator
        self.comm: MPI.Comm = MPI.COMM_WORLD

        self.rank: int = self.comm.Get_rank()

        # Whether this strategy is done executing; can be used by implmentations of EvolutionaryStrategy::run
        self.done: bool = False

        # Switch the pickle implementation in MPI to dill, which is more flexible / can serialize more things.
        MPI.pickle.__init__(dill.dumps, dill.loads)

        # Add rank to log lines
        loguru.logger.remove()
        loguru.logger.add(
            sys.stderr,
            format=(
                "| <level>{level: <6}</level>| RANK " + str(self.rank)
                + " | <cyan>{name}.{function}</cyan>:<yellow>{line}</yellow> | {message}"
            )
        )

        # Run initialization tasks locally.
        for task_name, init_task in init_tasks.items():
            logger.info(f"Executing initialization task '{task_name}' on rank {self.rank}")
            init_task.run(init_task.values(self))

    def recv(self, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG) -> Tuple[MPI.Status, Any]:
        """
        Calls `recv` using the communicator `self.comm`. By default, this will messages from any source and of any tag.

        Args:
            source: The source of the message we want to receive. By default, will receive messages from any source.
            tag: The tag of the message we want to receive. By default, will receive messages of any tag.

        Returns:
            A tuple containing the Status and the object received.
        """
        status: MPI.Status = MPI.Status()
        logger.info(f"Waiting for message from {source}")
        obj: Any = self.comm.recv(source=source, status=status, tag=tag)

        logger.info(f"Received {obj} from {status.Get_source()}")

        return status, obj

    @abstractmethod
    def _run_inner(self) -> None:
        """
        This method should contain the actual evolutionary loop. It will be wrapped in a try-except by
        `MPIEvolutionaryStrategy::run`.
        """
        ...

    @overrides(EvolutionaryStrategy)
    def run(self) -> None:
        """
        Calls the inner evolutionary loop and catches any exceptions; these should be handles by `self.__exit__` before
        reeaching this method.
        """
        try:
            self._run_inner()
        except Exception as e:
            logger.error("Failed because of exception:")
            logger.error("".join(traceback.format_exception(None, e, e.__traceback__)))


class AsyncMPIMasterStrategy[G: Genome, D: Dataset](MPIEvolutionaryStrategy[G, D]):
    """
    The master process logic for an asynchronous MPI-based evolutionary strategy.

    Until `self.nsteps` genoms has been evaluated, this strategy will:
      1. Wait until a worker process sends a work request.
      2. If the work request contains prior results, integrate them into the population.
      3. Generate a new piece of work for the worker.
      4. Send it to the worker.
      5. Goto step 1.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_workers: int = self.comm.Get_size() - 1
        self.n_generations: int = 0

        assert self.n_workers > 0, "MPI was launched with only 1 process, there must be at least 2."

    @overrides(EvolutionaryStrategy)
    def __enter__(self) -> Self:
        """
        Initializes the popluation.
        """
        self.population.initialize(self.genome_factory, self.dataset, self.rng)
        return super().__enter__()

    @overrides(EvolutionaryStrategy)
    def __exit__(
        self,
        exc_type: Optional[Type],
        exc_value: Optional[Exception],
        trace: Optional[types.TracebackType]
    ) -> None:
        """
        Cleanly terminates worker processes in the case of normal execution or in the case of an event.

        See `EvolutionaryStrategy::__exit__` for technical details.
        """
        super().__exit__(exc_type, exc_value, trace)

        # Finalize all workers.
        for _ in range(self.n_workers):
            status, _ = self.recv()
            self.comm.send(None, status.Get_source(), tag=Tags.FINALIZE.value)
            logger.info(f"Finalized worker {status.Get_source()}")

        logger.info("Sent finalize message to all workers...")

    @overrides(EvolutionaryStrategy)
    def step(self) -> None:
        """
        Received one message from a worker and processes it. Will integrate results into the population if they are
        valid, and will generate and send a task to the worker if the message type is expected - either an
        initialization task or a results task.
        """
        status, obj = self.recv()

        match status.tag, obj:
            # Indicates the worker process encountered an uncaught exception -- propagate the failure to
            # workers to ensure all log data is captured despite the premature termination.
            case Tags.FAILURE.value, _:
                logger.error("A worker process failed because of an exception.")
                raise Exception("Terminating due to a falure on a worker process.")

            # Result(s) - integrate into the population
            case Tags.RESULT.value, _:
                assert obj is not None
                assert type(obj) is list

                for genome in obj:
                    assert genome is None or isinstance(genome, Genome)

                logger.info("Updated log...")
                self.population.integrate_generation(obj)
                self.update_log(self.n_generations)
                self.n_generations += 1

            # An initialization task. We just have to send a task to the worker to start the exchange.
            case Tags.INITIALIZE.value, _:
                logger.info("Received initialize request from worker.")

            # Anything else is unexpected and will make the program explode.
            case _:
                raise Exception(f"Received message with unexpected tag {status.tag}")

        # Create a generation for the worker
        tasks = self.population.make_generation(self.genome_factory, self.rng)
        logger.info(f"Created generation of {len(tasks)} tasks for worker {status.Get_source()}")

        # Send the newly created generation
        self.comm.send(tasks, dest=status.Get_source(), tag=Tags.TASK.value)
        logger.info("Sent tasks")

    @overrides(EvolutionaryStrategy)
    def get_log_path(self) -> str:
        return f"{self.output_directory}/log.csv"

    @overrides(MPIEvolutionaryStrategy)
    def _run_inner(self) -> None:
        """
        With `self` as a context, calls `self.step` until the requisite number of generations has been evaluated.
        """
        with self:
            while True:
                logger.info(f"Starting step {self.n_generations} / {self.nsteps}")
                self.step()

                if self.n_generations >= self.nsteps:
                    logger.info(f"Ending on step {self.n_generations} / {self.nsteps}")
                    break


class AsyncMPIWorkerStrategy[G: Genome, D: Dataset](MPIEvolutionaryStrategy[G, D]):
    """
    The worker process logic for an asynchronous MPI-based evolutionary strategy.

    Until this receives the teermination message from the master process, this strategy will:
      1. Send an initialization message to the worker.
      2. Wait for a task.
      3. Execute the task.
      4. Send the results to the master.
      5. Goto step 2.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Number of rows written to the data log
        self.log_rows: int = 0

    @overrides(EvolutionaryStrategy)
    def __enter__(self) -> Self:
        """
        Sends an initialization message to the master and sets the population to None, as workers do not have a
        popluation.
        """
        logger.info("Sending initialization message to master")

        self.comm.send(None, dest=0, tag=Tags.INITIALIZE.value)
        self.population = None  # type: ignore

        return super().__enter__()

    @overrides(EvolutionaryStrategy)
    def __exit__(
        self,
        exc_type: Optional[Type],
        exc_value: Optional[Exception],
        trace: Optional[types.TracebackType]
    ) -> None:
        """
        Closes this process and, if this was caused by an exception, sends a failure signal to the master so it may
        terminate other workers.

        See `EvolutionaryStrategy::__exit__` for technical details.
        """
        super().__exit__(exc_type, exc_value, trace)

        if exc_value is not None:

            # Notify master of the failure on this process.
            self.comm.send(exc_value, dest=0, tag=Tags.FAILURE.value)

            # We want to wait for the finalize signal from the master, but it only gets sent in response to a message.
            self.comm.send(None, dest=0, tag=Tags.RESULT.value)

            # Wait for the finalize signal.
            self.recv(source=0, tag=Tags.FINALIZE.value)

    @overrides(EvolutionaryStrategy)
    def step(self) -> None:
        """
        A single step for this asychronous worker involves waiting for tasks from the master process, executing those
        tasks, and sending the results to the mater process.

        If a FINALIZE message is received instead of a task, `self.done` is set to True.
        """
        logger.info("Waiting for a task")
        status, obj = self.recv(source=0)

        logger.info(f"Received tasks {type(obj)} from {status.Get_source()}")

        match status.Get_tag(), obj:
            # We received a task(s) to execute - do so and send back the results.
            case Tags.TASK.value, _:
                results = []
                for task in obj:
                    logger.info(f"Evaluating task {task}")
                    genome: Optional[G] = task(self.rng)

                    if genome:
                        logger.info("Evaluating fitness")
                        genome.fitness = genome.evaluate(self.fitness, self.dataset)
                        logger.info("Finished evaluation")
                    results.append(genome)

                logger.info("Sending results to main")
                self.comm.send(results, dest=status.Get_source(), tag=Tags.RESULT.value)
                logger.info("Done.")

            # We receievd the termination signal from the master - exit.
            case Tags.FINALIZE.value, _:
                self.done = True

            # Received something we were not expecting.
            case _:
                ...

    @overrides(EvolutionaryStrategy)
    def get_log_path(self) -> str:
        return f"{self.output_directory}/worker_{self.rank}_log.csv"

    @overrides(MPIEvolutionaryStrategy)
    def _run_inner(self) -> None:
        """
        With `self` as a context, calls `self.step` until `self.done` is set. Updates the log after each step.
        """
        with self:
            while not self.done:
                logger.info(f"Starting step {self.log_rows}")
                self.step()

                self.update_log(self.log_rows)
                self.log_rows += 1
                logger.info(f"Ending step {self.log_rows}")


def async_mpi_strategy_factory(**kwargs):
    """
    A factory method for Asychronous MPI strategies. This will instantiate `AsyncMPIMasterStrategy` for the process with
    rank 0, and a `AsyncMPIWorkerStrategy` for all other processes.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        return AsyncMPIMasterStrategy(**kwargs)
    else:
        return AsyncMPIWorkerStrategy(**kwargs)


@configclass(name="base_async_mpi_strategy", target=async_mpi_strategy_factory)
class AsyncMPIStrategyConfig(EvolutionaryStrategyConfig):
    init_tasks: Dict[str, InitTaskConfig] = field(default_factory=lambda: {})


class SynchronousMPIMasterStrategy[G: Genome, D: Dataset](MPIEvolutionaryStrategy[G, D]):
    """
    Logic for the master process in a synchronous, MPI-based evolutionary strategy.

    The basic logic is:
      1. Create a generation of tasks.
      2. Partition the tasks and send them to the workers.
      3. Wait for workers to send results.
      4. Combine results and integrate into population.
      5. Goto step 1 if not done.

    There is also a good amount of code in here for handling errors; if it were not there it is pretty easy for the
    program to freeze and / or to lose log data.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_workers: int = self.comm.Get_size() - 1
        self.n_generations: int = 0

        assert self.n_workers > 0, "MPI was launched with only 1 process, there must be at least 2."

    @overrides(EvolutionaryStrategy)
    def __enter__(self) -> Self:
        """
        Initializes the population
        """
        self.population.initialize(self.genome_factory, self.dataset, self.rng)
        return super().__enter__()

    @overrides(EvolutionaryStrategy)
    def __exit__(
        self,
        exc_type: Optional[Type],
        exc_value: Optional[Exception],
        trace: Optional[types.TracebackType]
    ) -> None:
        """
        Safely closes all worker processes and synchronizes with a barrier to ensure they are all able to complete any
        I/O operations.

        See `EvolutionaryStrategy::__exit__` for technical details.
        """
        super().__exit__(exc_type, exc_value, trace)

        logger.info("Exiting...")

        for i in range(1, self.n_workers + 1):
            self.comm.send(None, dest=i, tag=Tags.FINALIZE.value)
            logger.info(f"Finalized worker {i}")

        self.comm.Barrier()
        logger.info("All workers have been finalized.")

    @overrides(EvolutionaryStrategy)
    def step(self) -> None:
        """
        Creates a generation of tasks and splits them into `self.n_workers` chunks. These chunks are distributed to
        workers, executed, and the results are gathered in this method. These results are combined and integrated into
        the popluation.

        This method also must watch out for failures on the workers. If that occurs, an exception is thrown after gather
        all results from worker. Then, an exception is thrown which will be handled by `self.__exit__`. See that for
        details on the safe termination sequence.
        """
        logger.info("About to generate tasks...")
        tasks = self.population.make_generation(self.genome_factory, self.rng)

        chunks = np.array_split(tasks, self.n_workers)  # type: ignore

        for i, chunk in enumerate(chunks):
            logger.info(f"CHUNK {i} = {chunk}")
            self.comm.send(chunk, dest=i + 1, tag=Tags.TASK.value)

        logger.info("Sent tasks")

        stats, genomes = [], []
        # Used to track of we encountered at least one failure.
        failure = False

        for ichunk in range(self.n_workers):
            logger.info(f"Waiting for chunk {ichunk}")

            status, obj = self.recv()

            logger.info("Done.")
            stats.append(status)

            match status.Get_tag():
                case Tags.FAILURE.value:
                    failure = True
                case Tags.RESULT.value:
                    assert obj is not None
                    assert type(obj) is list

                    for g in obj:
                        assert g is None or isinstance(g, Genome)

                    genomes += obj

            if failure:
                raise Exception("A worker signaled failure.")

        self.population.integrate_generation(genomes)
        logger.info("Integrated generation")

    @overrides(MPIEvolutionaryStrategy)
    def _run_inner(self) -> None:
        """
        With `self` as a context, performs steps and updates the log until `self.nsteps` generations have been
        evaluated.
        """
        with self:
            while True:
                logger.info(f"Starting step {self.n_generations} / {self.nsteps}")
                self.step()

                self.update_log(self.n_generations)
                self.n_generations += 1

                if self.n_generations >= self.nsteps:
                    logger.info(f"Ending on step {self.n_generations} / {self.nsteps}")
                    break


class SynchronousMPIWorkerStrategy[G: Genome, D: Dataset](MPIEvolutionaryStrategy[G, D]):
    """
    Logic for a single worker in a synchronous MPI-based evolutionary strategy.

    The logic is roughly as follows:
      1. Wait for a message from the master process.
      2. If it has tag FINALIZE, exit.
      3. Otherwise it should contain tasks: execute them.
      4. Send results to the master process.
      5. Goto step 1.

    There is also a good amount of logic related to error handling; this is to ensure no data is lost upon abnormal
    termination.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Number of rows written to the data log
        self.log_rows: int = 0

    @overrides(EvolutionaryStrategy)
    def __enter__(self) -> Self:
        """
        Sets the population to None since workers do not have populatons.
        """
        self.population = None  # type: ignore
        return super().__enter__()

    @overrides(EvolutionaryStrategy)
    def __exit__(
        self,
        exc_type: Optional[Type],
        exc_value: Optional[Exception],
        trace: Optional[types.TracebackType]
    ) -> None:
        """
        If an exception caused this method to be called, then the master process will be notified of the error at which
        point it will begin safely closing out other processes.

        Otherwise, this method simply writes its log (done in the super implementation) and waits at the barrier to
        sychronize with all processes.
        """
        super().__exit__(exc_type, exc_value, trace)

        if exc_type is not None:
            self.comm.isend(None, dest=0, tag=Tags.FAILURE.value)
            self.recv(source=0, tag=Tags.FINALIZE.value)

        self.comm.Barrier()

    @overrides(EvolutionaryStrategy)
    def step(self) -> None:
        """
        A single sychronous step of evolution for the worker. Essentially, the worker receives a list of tasks and
        executes them. Then, the results are sent to the master process.
        """
        logger.info("Waiting for a task")
        status, obj = self.recv(source=0)

        logger.info(f"Received tasks {type(obj)} from {status.Get_source()}")

        match status.Get_tag(), obj:
            # Task(s) from the master; execute and relay the results.
            case Tags.TASK.value, _:
                logger.info(f"Received {obj}")
                assert obj is not None
                assert type(obj) is np.ndarray

                results = []
                for task in obj:
                    genome: Optional[G] = task(self.rng)

                    if genome:
                        genome.fitness = genome.evaluate(self.fitness, self.dataset)
                    results.append(genome)

                self.comm.send(results, dest=status.Get_source())

            # Termination signal from master.
            case Tags.FINALIZE.value, _:
                self.done = True

    @overrides(EvolutionaryStrategy)
    def get_log_path(self) -> str:
        return f"{self.output_directory}/worker_{self.rank}_log.csv"

    @overrides(MPIEvolutionaryStrategy)
    def _run_inner(self) -> None:
        """
        Enters `self` as a context and calls `self.step` until `self.done` is set to True. Updates log after each step.

        If an exception occurs in the body of the loop, it will first be handled by `self.__exit__`. This will start a
        safe termination sequence, see self.__exit__.
        """
        with self:
            while not self.done:
                logger.info(f"Starting step {self.log_rows}")
                self.step()

                self.update_log(self.log_rows)
                self.log_rows += 1

                logger.info(f"Ending step {self.log_rows}")


def sync_mpi_strategy_factory(**kwargs):
    """
    A factory method that will instantiate `SynchronousMPIMasterStrategy` on the process with rank 0, and a
    `SynchronousMPIWorkerStrategy` on all other processes.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        return SynchronousMPIMasterStrategy(**kwargs)
    else:
        return SynchronousMPIWorkerStrategy(**kwargs)


@configclass(name="base_sync_mpi_strategy", target=sync_mpi_strategy_factory)
class SynchronousMPIStrategyConfig(EvolutionaryStrategyConfig):
    init_tasks: Dict[str, InitTaskConfig] = field(default_factory=lambda: {})
