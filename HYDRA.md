# Hydra
As is stated in the README, Hydra is a tool managing for hierarchical experiment configurations. To see a super simple example that demonstrates what Hydra is, see the quick start guide here: https://hydra.cc/docs/intro/
You should give the official Hydra documentation a scim before trying to do much of anything.

# Structured Configs
By default, Hydra is not statically typed and is truly like a configuration file concatenation tool. It is, however, possible to introduce static typing into the mix. The basic idea is to create a hierarchy of configuration dataclasses (lookup python dataclasses if you are not familiar) that define every configurable facet of a program.

These dataclasses can depend one other dataclasses that should form a hierarchy (it might be possible for this to be cyclic, but I'm not sure). Here is a simple example of what a hierarchy could look like:
```python
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import hydra

@dataclass
class DatasetConfig:
    name: str
    batch_size: int
    num_workers: int

@dataclass
class ModelConfig:
    type: str
    hidden_layers: int

@dataclass
class OptimizerConfig:
    name: str = field(default="adam")
    learning_rate: float

@dataclass
class TrainingConfig:
    epochs: int = field(default=4)
    optimizer: OptimizerConfig

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
```

We also have to register these in a global `ConfigStore` in order for Hydra to properly understand the class hierarchy:
```python
ConfigStore.instance().store(name="config", node=Config)
ConfigStore.instance().store(name="dataset_config", node=DatasetConfig)
ConfigStore.instance().store(name="model_config", node=ModelConfig)
ConfigStore.instance().store(name="optimizer_config", node=TrainingConfig)
ConfigStore.instance().store(name="training_config", node=TrainingConfig)
```

The "root" configuration here would of course be the last class called `Config`. If you wanted to place this all in a single configuration file, it would look something like this:

`config.yaml`:
```yaml
dataset:
  name: cifar10
  batch_size: 32
  num_workers: 4

model:
  type: resnet
  hidden_layers: 18

training:
  optimizer:
    name: adagrad
    learning_rate: 0.001
  epochs: 10
```

We can also define things modularly using the defaults list:
`./config.yaml`
```yaml
defaults:
  - dataset: my_dataset
  - model: my_model
  - training: my_training_config
```

in `./my_dataset.yaml`:
```yaml
name: cifar10
batch_size: 32
num_workers: 4
```

in `./my_model.yaml`
```yaml
type: resnet
hidden_layers: 18
```

and in `./my_training_config.yaml`
```yaml
defaults:
  - optimizer: my_optimizer
epochs: 10
```

and finally, `./my_optimizer.yaml`:
```yaml
name: adagrad
learning_rate: 0.001
```

We can also refer to the default values in the dataclasses by using the name we used to register them:
`./config.yaml`
```yaml
defaults:
  - dataset: my_dataset
  - model: my_model
  - training: training_config
  # the optimizer field of TrainingConfig has no default value, so specify it here.
  - training/optimizer: my_optimizer
  
```

Hydra will look at the defaults list and search for configuration files or default values. You can store your configruation files in subdirectories if you'd like.
For more regarding the defaults list, see here: https://hydra.cc/docs/advanced/defaults_list/.

# Object Instantiation
Hydra uses a library called OmegaConf to do most of the heavy lifting when it comes to merging configuration objects. One of the cool features that OmegaConf supports is object instantiation: if a config has a `_target_` that points to a fully qualified python class name, it will automatically create that object with the arguments in the config.

Here is a simple example of this:
```python

class MyModel:

  def __init__(self, x: int, y: int) -> None:
    self.x: int = x
    self.y: int = y

  # calling this a model is generous I admit
  def forward(self, z: int) -> int:
    return self.x + self.y * z

@dataclass
class MyModelConfig:
  _target_ = "my_module.MyModel"
  x: int = field(default=4)
  y: int = field(default=0)

# Registering omitted
```

Then in our Hydra main function:

```python
@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: MyModelConfig) -> None:
    # Instantiate and run the evolutionary strategy.
    # See its definition in the `evolution` package.
    my_model = hydra.util.instantiate(cfg)
    assert type(e) == MyModel
```

with our configuration file `default.yaml`:
```yaml
x: 4
y: 2
```

This concept can be applied to class hierarchies as well, and Hydra will instantiate everything correctly:
```python
class MySubModel:
  ...

class MyModel:

  def __init__(self, x: int, sub_model: MySubModel) -> None:
    ...

@dataclass
class MySubModelConfig:
  _target_ = "my_module.MySubModel"
  parameters: list

@dataclass
class MyModelConfig:
  _target_ = "my_module.MyModel"
  x: int = field(default=4)
  sub_model: MySubModelConfig

# Class registration omitted
```

Structured configs and instantiation also support inheritance, subclassing etc. however the subclasses must override the `_target_` field in order to be properly instantiated.

# Hydra and EXA-STAR
EXA-STAR uses all of these features to remove the boilerplate out of application and experimental configuration. The so called "root" configuration class is called `EvolutionaryStrategyConfig` which you can find in `src/evolution/evolutionary_strategy.py`.

In the class you will see an `EvolutionaryStrategy` class and its corresponding config class:
```python
@dataclass(kw_only=True)
class EvolutionaryStrategyConfig(LogDataAggregatorConfig):
    environment: Dict[str, str] = field(default_factory=dict)
    output_directory: str
    population: PopulationConfig
    genome_factory: GenomeFactoryConfig
    fitness: FitnessConfig
    dataset: DatasetConfig
    nsteps: int = field(default=10000)
```

This defines all of the information needed to define an experiment / application configuration. All of the config classes here are abstract, meaning they can't actually be instantiated on their own.

Lets take a look at a concrete class definition, in `src/evolution/mp_evolutionary_strategy.py`:
```python
@configclass(name="base_async_mp_strategy", target=AsyncMPStrategy)
class AsyncMPStrategyConfig(ParallelMPStrategyConfig):
    ...
```

`ParallelMPStrategyConfig` is its parent class, which itself is a child class of `EvolutionaryStrategyConfig`. It is absolutely key that you note the `@configclass` decorator that is being used: this automates the task of specifying the `_target_` field, pointing it to the specified target. It also registers things in the global register configuration that Hydra requires us to use.


## Advanced Syntax
There is some "advanced" Hydra syntax used in the default `conf/exastar/conf.yaml` - though it is really a work around the limitations of the `defaults` list in Hydra:
```yaml
defaults:
  - genome_factory/mutation_operators@genome_factory.mutation_operators.add_edge: base_add_edge_mutation
... # incomplete config
```

The `@` syntax looks confusing but it really is not, though it is ugly and should not be neccessary but Hydra lacks this feature.

Basically what it says is: in the dictionary `genome_factory.mutation_operators` add the key `add_node` and set it to the value given by `genome_factory/mutation_operators/base_add_edge_mutation`.

Something we haven't talked about is configuration groups - that is what `genome_factory/mutation_operators` is: a group of `mutation_operators`. Group is an optional field to the `configclass` decorator, and is used as follows for mutations:

```python
@configclass(name="base_add_recurrent_edge_mutation", group="genome_factory/mutation_operators",
             target=AddRecurrentEdge)
class AddRecurrentEdgeConfig(EXAStarMutationOperatorConfig):
    # TODO: Fix the type checking error that seems to come from config classes
    recurrent_edge_generator: RecurrentEdgeGeneratorConfig = field(
        default_factory=lambda: RecurrentEdgeGeneratorConfig(p_recurrent=1.0))  # pyright: ignore
```

Hopefully this clears up that absurd syntax.

# Further Reading
The Hydra documentation is actually quite good, most likely better than this document. Particularly for advanced features and syntax, refer to the documentation: https://hydra.cc/docs/intro/
