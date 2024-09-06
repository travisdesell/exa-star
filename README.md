# EXA-STAR
This repository contains the source code for a generic, evolutionary algorithms framework. While geared towards neuroevolution, the interfaces are generic enough to define a wide variety of evolutionary algorithms.

## Overview
The code is in the `src/` directory, and defines the following generic modules that are used to define evolutionary algorithms:
```
src
├── dataset    # defines a generic dataset interface (quite sparse currently).
├── evolution  # top-level definition of evolutionary algorithms - puts the next two modules together.
├── genome     # defines interfaces for genomes, mutation, crossover, operator selection, genome fitness, etc.
├── population # defines an interface for population strategy and some simple populations.
└── util       # some stuff related to logging and some useful type traits.
```

There are two other modules which define evolutionary algorithms using these components:
```
├── exastar
│   ├── genome
│   │   ├── component
│   │   └── seed
│   ├── genome_operators
│   └── util
└── toy
```

Unsurprisingly, the `toy` EA is a lot simpler than the `exastar` EA. As of now, EXA-STAR is the primary substance of this repository.

You can find explanations of each of these modules in their respective `__init__.py` files, and of course within each individual source file. It is recommended to start in `src/evolution` as this is the highest-level in the object hierarchy and should give you a clear idea of how everything interacts and is pieced together.

## Hydra
If you pick through the code, you will notice most classes are shadowed by a corresponding config class, e.g.
```
class CrossoverOperator[G: Genome](GenomeOperator[G]):

    def __init__(self, weight: float) -> None:
        super().__init__(weight)

    @abstractmethod
    def __call__(self, parents: List[G], rng: np.random.Generator) -> Optional[G]:
        ...

@dataclass
class CrossoverOperatorConfig(GenomeOperatorConfig):
    ...
```

This is to support a tool called Hydra, which makes the composition of hierarchical experiment configurations super easy and clear (and statically typed). In addition to saving us from command-line purgatory, it saves us a ton of LOC that would be spent writing boilerplate code.
Using Hydra should also ensure that our experiments are robust and easily reproducible by avoiding the need to float various shell scripts with dozens of command line parameters around.

Experiments in Hydra are defined in `.yaml` files - these can be modular or all defined in-line, it is up to you. The EA defined in `src/toy` should serve as a basic starter for how Hydra is used in this project.

# Running
You need a new-ish version of python - at least enough to support `match` meaning >= 3.10.

Install depdendencies (you should probably do this in a virtual environment):
```
pip install -r requirements.txt
```

Some Mac systems have a safeguard against fork-bombs (or something like that) that you may need to disable to run the program:
```
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```
(note that this must be done before the program is run, modifying the environment variables w/ the hydra config does not work here)

Consider adding it to your `~/.profile` / `~/.bashrc` / init script for whatever shell you use.

Once installed you should be able to run the default configuration of `exastar` with:
```
python3 src --config-path ../conf/exastar --config-name conf
```

If you are having trouble with Hydra, you can set the following environment variable:
```
export HYDRA_FULL_ERROR=1
```

You can also modify the environment in your hydra config for this:
```
defaults:
  ...

environment:
  HYDRA_FULL_ERROR: "1"
...
```

# Contributing
See CONTRIBUTING.md for details.
