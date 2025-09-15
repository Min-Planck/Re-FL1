import flwr as fl
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import nn
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Context

from utils import get_model, get_train_data, clustering, set_seed, compute_entropy, get_parameters, set_parameters, get_fedhcw_config, get_num_clients, get_num_rounds, get_algorithm, get_distribution_diff_from_uniform
from default_config import ALGORITHM_CONFIG, CLIENT_CONFIG
from fit_handler import fit_handler
from client_manager import SimpleClientManager


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
ALGO = 'fedhcw'
cluster_algo = 'bkmeans'  # 'bkmeans' / 'agglomerative' / 'kmeans' / 'optics'
BATCH_SIZE = 32
PARTITION_FRACTION = [0.3, 0.3, 0.3, 0.1]
ALPHA = [0.01, 0.05, 0.1, 100]
LR = 0.01
LOCAL_TRAINING = 10
DATASET_NAME = 'fmnist' 
DISTANCE = 'hellinger'
AGGREGATE_CLUSTER_ALGORITHM = 'fedadp' # 'feddisco' / 'fedcls' / 'fedadp' / 'fedavg' 

NUM_CLIENTS = get_num_clients(DATASET_NAME) 
NUM_ROUNDS = get_num_rounds(DATASET_NAME)

EXP_NAME = f'{ALGO}_{DATASET_NAME}-dataset_{LR}-lr_{NUM_CLIENTS}-clients_{LOCAL_TRAINING}-epochs'
set_seed(RANDOM_SEED)
# ------------------------------------ PREPROCESS -----------------------------------------
ids, dist, trainloaders, testloader, client_dataset_ratio = get_train_data(
    dataset_name=DATASET_NAME,
    num_clients=NUM_CLIENTS,
    batch_size=BATCH_SIZE, 
    fractions=PARTITION_FRACTION,
    alphas=ALPHA
)

if ALGO == 'fedhcw': 
    ALGORITHM_CONFIG, client_cluster_index = get_fedhcw_config(dist, cluster_algo, NUM_CLIENTS, DISTANCE, AGGREGATE_CLUSTER_ALGORITHM, ALGORITHM_CONFIG)

if ALGO == 'feddisco': 
    dk = get_distribution_diff_from_uniform(dist, NUM_CLIENTS)
    ALGORITHM_CONFIG = {**ALGORITHM_CONFIG, **{"dk": dk}}
elif ALGO in ['fedimp', 'fedhcw']: 
    entropies = [compute_entropy(dist[i]) for i in range(NUM_CLIENTS)]
    ALGORITHM_CONFIG = {**ALGORITHM_CONFIG, **{"entropies": entropies}}
elif ALGO == 'fedcls':
    ALGORITHM_CONFIG = {**ALGORITHM_CONFIG, **{"all_classes": len(dist[0])}}

# ------------------------------------ CLIENT -----------------------------------------

class BaseClient(fl.client.NumPyClient):
    def __init__(self, 
                 cid, 
                 net, 
                 trainloader, 
                 device):
         
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.client_control = None
        self.num_classes = sum(v is not None and v > 0 for v in dist[cid].values())

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        config = {**CLIENT_CONFIG, **config, **{"learning_rate": LR, "epochs": LOCAL_TRAINING, "num_classes": self.num_classes, "device": self.device}}
        set_parameters(self.net, parameters)
        metrics = fit_handler(algo_name=ALGO, cid=self.cid, net=self.net, trainloader=self.trainloader, config=config, client_control=self.client_control, parameters=parameters)
        
        if ALGO == "fedcls":
            metrics["num_classes"] = self.num_classes
        if ALGO == 'fedhcw': 
            metrics["cluster_id"] = client_cluster_index[self.cid]

        if ALGO == "scaffold":
            self.client_control = metrics["client_control"]
            params_obj = metrics['params']
            client_params_news = parameters_to_ndarrays(params_obj)
            _, _ = metrics.pop("params", None), metrics.pop("client_control", None)
        else: 
            client_params_news = get_parameters(self.net)

        metrics = {k: v for k, v in metrics.items() if v is not None}
        return client_params_news, len(self.trainloader.sampler), metrics    
    
    def evaluate(self, parameters, config):
        return None
    
def client_fn(context: Context) -> BaseClient:
    cid = int(context.node_config["partition-id"])
    is_moon_type = True if ALGO == 'moon' else False
    net = get_model(dataset_name=DATASET_NAME, moon_type=is_moon_type) 
    net.to(DEVICE)
    trainloader = trainloaders[int(cid)]  
    return BaseClient(cid, net, trainloader, DEVICE)

# ------------------------------------ RUN -----------------------------------------

is_moon_type = True if ALGO == 'moon' else False 
net_ = get_model(DATASET_NAME, is_moon_type)
current_parameters = ndarrays_to_parameters(get_parameters(net_)) 
client_resources = {"num_cpus": 2, "num_gpus": 0.2} if DEVICE == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}

algo = get_algorithm(ALGO)
strategy = algo(
    exp_name=EXP_NAME,
    net=net_,
    num_rounds=NUM_ROUNDS,
    num_clients=NUM_CLIENTS,
    testloader=testloader,
    algorithm_config=ALGORITHM_CONFIG,
    learning_rate=LR,
    current_parameters=current_parameters
)


fl.simulation.start_simulation(
            client_fn           = client_fn,
            num_clients         = NUM_CLIENTS,
            config              = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy            = strategy,
            client_manager      = SimpleClientManager(),
            client_resources     = client_resources
        )