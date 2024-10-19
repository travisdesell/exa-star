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
    TASK = 0
    RESULT = 1
    FINALIZE = 2
    INTIALIZE = 3
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
            format="| <level>{level: <6}</level>| RANK " + str(self.rank) +
            " | <cyan>{name}.{function}</cyan>:<yellow>{line}</yellow> | {message}"
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

    def abort(self, e: Exception) -> None:
        """
        Prints the stacktrace of the supplied exception and aborts MPI - this kills all processes.
        """
        error_message = traceback.format_exc()
        logger.info(f"FAILED with exception '{e}':\n{error_message}")
        logger.info("Exiting prematurely :(...")

        # This calls sys.exit internally
        self.comm.Abort()


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
        self.population.initialize(self.genome_factory, self.dataset, self.rng)
        return self

    @overrides(EvolutionaryStrategy)
    def __exit__(
        self,
        exc_type: Optional[Type],
        exc_value: Optional[Exception],
        trace: Optional[types.TracebackType]
    ) -> None:
        super().__exit__(exc_type, exc_value, trace)

        # Finalize all workers.
        for _ in range(self.n_workers):
            status, _ = self.recv()
            self.comm.send(None, status.Get_source(), tag=Tags.FINALIZE.value)
            logger.info(f"Finalized worker {status.Get_source()}")

        logger.info("Sent finalize message to all workers...")

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

            # A result of None - the task failed.
            case Tags.RESULT.value, None:
                logger.info("Received None from worker result - a task must have failed.")

            # A non-None result - integrate it into the population
            case Tags.RESULT.value, _:
                assert type(obj) is list

                for genome in obj:
                    assert genome is None or isinstance(genome, Genome)

                logger.info("Updated log...")
                self.population.integrate_generation(obj)
                self.update_log(self.n_generations)
                self.n_generations += 1

            # An initialization task. We just have to send a task to the worker to start the exchange.
            case Tags.INTIALIZE.value:
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

    @overrides(EvolutionaryStrategy)
    def run(self):
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
        logger.info("Sending initialization message to master")
        self.comm.send(None, dest=0, tag=Tags.RESULT.value)
        self.population = None  # type: ignore

        return self

    @overrides(EvolutionaryStrategy)
    def __exit__(
        self,
        exc_type: Optional[Type],
        exc_value: Optional[Exception],
        trace: Optional[types.TracebackType]
    ) -> None:
        super().__exit__(exc_type, exc_value, trace)

        if exc_value is not None:
            logger.error("Encountered an uncaught exception.")
            logger.error("".join(traceback.format_exception(None, exc_value, trace)))

            # Notify master of the failure on this process.
            self.comm.send(exc_value, dest=0, tag=Tags.FAILURE.value)

            # We want to wait for the finalize signal from the master, but it only gets sent in response to a message.
            self.comm.send(None, dest=0, tag=Tags.RESULT.value)

            # Wait for the finalize signal.
            self.recv(source=0, tag=Tags.FINALIZE.value)

    def step(self) -> None:
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

    @overrides(EvolutionaryStrategy)
    def run(self):
        with self:
            while not self.done:
                logger.info(f"Starting step {self.log_rows}")
                self.step()

                self.update_log(self.log_rows)
                self.log_rows += 1
                logger.info(f"Ending step {self.log_rows}")

        self.log[:self.log_rows].to_csv(self.get_log_path())


def async_mpi_strategy_factory(**kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        return AsyncMPIMasterStrategy(**kwargs)
    else:
        return AsyncMPIWorkerStrategy(**kwargs)


@configclass(name="base_async_mpi_strategy", target=async_mpi_strategy_factory)
class AsyncMPIStrategyConfig(EvolutionaryStrategyConfig):
    init_tasks: Dict[str, InitTaskConfig] = field(default_factory=lambda: {})


class SynchronousMPIMasterStrategy[G: Genome, D: Dataset](MPIEvolutionaryStrategy[G, D]):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_workers: int = self.comm.Get_size() - 1
        self.n_generations: int = 0

        assert self.n_workers > 0, "MPI was launched with only 1 process, there must be at least 2."

    def __enter__(self) -> Self:
        self.population.initialize(self.genome_factory, self.dataset, self.rng)
        return self

    def __exit__(self, *_) -> None:
        workers_closed: int = 0

        logger.info("Exiting...")
        while workers_closed < self.n_workers:
            status, _ = self.recv()
            self.comm.send(None, status.Get_source(), tag=Tags.FINALIZE.value)
            logger.info(f"Finalized worker {status.Get_source()}")

        logger.info("All workers have been finalized.")
        super().__exit__(*_)

    def step(self) -> None:
        logger.info("About to generate tasks...")
        tasks = self.population.make_generation(self.genome_factory, self.rng)

        chunks = np.array_split(tasks, self.n_workers)  # type: ignore

        for i, chunk in enumerate(chunks):
            self.comm.send(chunk, dest=i + 1, tag=Tags.TASK.value)

        logger.info("Sent tasks")

        stats, genomes = [], []

        for ichunk in range(self.n_workers):
            logger.info(f"Waiting for chunk {ichunk}")
            status, obj = self.recv()
            logger.info("Done.")
            stats.append(status)

            if obj is not None:
                assert type(obj) is list

                for g in obj:
                    assert g is None or isinstance(g, Genome)

                genomes += obj

        self.population.integrate_generation(genomes)
        logger.info("Integrated generation")

    def get_log_path(self) -> str:
        return f"{self.output_directory}/log.csv"

    def run(self):
        try:
            with self:
                while True:
                    logger.info(f"Starting step {self.n_generations} / {self.nsteps}")
                    self.step()

                    self.update_log(self.n_generations)
                    self.n_generations += 1

                    if self.n_generations >= self.nsteps:
                        logger.info(f"Ending on step {self.n_generations} / {self.nsteps}")
                        break

        except Exception as e:
            self.log.to_csv(self.get_log_path())
            self.abort(e)


class SynchronousMPIWorkerStrategy[G: Genome, D: Dataset](MPIEvolutionaryStrategy[G, D]):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Number of rows written to the data log
        self.log_rows: int = 0

    def __enter__(self) -> Self:
        self.population = None  # type: ignore
        return self

    def step(self) -> None:
        logger.info("Waiting for a task")
        status, obj = self.recv(source=0)

        logger.info(f"Received tasks {type(obj)} from {status.Get_source()}")

        if obj is not None:
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
            self.comm.send(results, dest=status.Get_source())
            logger.info("Done.")
        else:
            self.done = True

    def get_log_path(self) -> str:
        return f"{self.output_directory}/worker_{self.rank}_log.csv"

    def run(self):
        try:
            with self:
                while not self.done:
                    logger.info(f"Starting step {self.log_rows}")
                    self.step()

                    self.update_log(self.log_rows)
                    self.log_rows += 1
                    logger.info(f"Ending step {self.log_rows}")

            self.log[:self.log_rows].to_csv(self.get_log_path())
        except Exception as e:
            logger.info(f"FAILED with exception {e}")


def sync_mpi_strategy_factory(**kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        return SynchronousMPIMasterStrategy(**kwargs)
    else:
        return SynchronousMPIWorkerStrategy(**kwargs)


@configclass(name="base_sync_mpi_strategy", target=sync_mpi_strategy_factory)
class SynchronousMPIStrategyConfig(EvolutionaryStrategyConfig):
    init_tasks: Dict[str, InitTaskConfig] = field(default_factory=lambda: {})
