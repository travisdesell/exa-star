from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple

from config import configclass

from loguru import logger
import pandas as pd
import numpy as np
import torch


class Dataset(ABC):
    ...


@dataclass
class DatasetConfig:
    ...
