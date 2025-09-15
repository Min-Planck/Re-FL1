import flwr as fl
import os
import copy
import torch
import numpy as np
import pandas as pd

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Status,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional
from functools import partial, reduce
from utils import train, get_parameters, set_parameters, test

from algo.fedavg import FedAvg
from algo.fedaaw import FedAAW
from algo.fedadp import FedAdp
from algo.fedntd import FedNTD
from algo.feddisco import FedDisco
from algo.fedimp import FedImp
from algo.moon import MOON
from algo.fedcls import FedCLS
from algo.scaffold import Scaffold  
from algo.fedhcw import FedHCW