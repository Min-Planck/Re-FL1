from utils import normalize_distribution, compute_uniform_distribution, kl_divergence, clustering
from algo import FedAvg, FedAdp, MOON, FedHCW, FedImp, FedAAW, FedDisco, FedNTD, FedCLS, Scaffold
from collections import defaultdict, Counter

def get_num_clients(dataset_name): 
    if dataset_name in ['fmnist', 'agnews']:
        return 120
    elif dataset_name in ['cifar10', 'cifar100']:
        return 80
    
def get_num_rounds(dataset_name):
    if dataset_name in ['fmnist', 'agnews']:
        return 100
    elif dataset_name == 'cifar10':
        return 300
    elif dataset_name == 'cifar100': 
        return 500
    
def make_cluster_sizes(num_clients, num_clusters):
    base = num_clients // num_clusters
    rem = num_clients % num_clusters
    sizes = [base + (1 if i < rem else 0) for i in range(num_clusters)]
    return sizes


def get_distribution_diff_from_uniform(distribution, num_clients): 
    
    dk = {} 
    num_classes = len(distribution[0]) 

    uniform_dist = compute_uniform_distribution(num_classes=num_classes)

    for client_id in range(num_clients): 
        client_dist = distribution[client_id]
        normalized_client_dist = normalize_distribution(client_dist)
        diff = kl_divergence(p=normalized_client_dist, q=uniform_dist)
        dk[client_id] = diff
    
    return dk 

def get_fedhcw_config(dist, cluster_algo, NUM_CLIENTS, DISTANCE, AGGREGATE_CLUSTER_ALGORITHM, config): 
    cluster_size = make_cluster_sizes(NUM_CLIENTS, 10)  
    client_cluster_index, distrib_ = clustering(
        dist, 
        algo=cluster_algo,
        num_clusters=10,
        cluster_size=cluster_size, 
        distance=DISTANCE,
    )
    
    num_cluster = len(set(client_cluster_index.values()))
    if -1 in client_cluster_index.values():
        num_cluster -= 1  

    print(f'Number of Clusters: {num_cluster}')
    
    increment = 0
    for k, v in client_cluster_index.items():
        if v == -1:
            client_cluster_index[k] = num_cluster + increment
            increment += 1

    dist_cluster = defaultdict(Counter)
    if AGGREGATE_CLUSTER_ALGORITHM in ['feddisco', 'fedcls']:
        for i in range(NUM_CLIENTS):
            c_id = client_cluster_index[i]
            dist_cluster[c_id].update(dist[i])
            
        if AGGREGATE_CLUSTER_ALGORITHM == 'feddisco': 
            dist_cluster = {cid: dict(c) for cid, c in dist_cluster.items()}
            dk = get_distribution_diff_from_uniform(dist_cluster, num_cluster) 
            config = {**config, **{"dk": dk}}
        elif AGGREGATE_CLUSTER_ALGORITHM == 'fedcls':
            num_class_per_cluster = {
                cid: sum(v > 0 for v in dist_cluster[cid].values())
                for cid in dist_cluster
            }
            config = {**config, **{"num_class_per_cluster": num_class_per_cluster}}


    for k, v in client_cluster_index.items():
        print(f'Client {k + 1}: Cluster: {v}')
    for i in range(NUM_CLIENTS):
        print(f"Client {i+1}: {dist[i]}")

    config = {**config, **{"aggregate_cluster_algo": AGGREGATE_CLUSTER_ALGORITHM}}
    return config, client_cluster_index

def get_algorithm(ALGO):
    if ALGO == 'fedavg':
        return FedAvg
    elif ALGO == 'fedadp':
        return FedAdp
    elif ALGO == 'moon':
        return MOON 
    elif ALGO == 'fedhcw':
        return FedHCW
    elif ALGO == 'fedimp':
        return FedImp
    elif ALGO == 'fedaaw':
        return FedAAW
    elif ALGO == 'feddisco': 
        return FedDisco
    elif ALGO == 'fedntd':
        return FedNTD
    elif ALGO == 'fedcls':
        return FedCLS
    elif ALGO == 'scaffold':
        return Scaffold