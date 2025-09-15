from algo import *

class FedHCW(FedAvg): 

    def __init__(self, 
                 *args, 
                 temperature: float = 0.2, 
                 alpha_adp: float = 5, 
                 alpha_disco: float = 0.5, 
                 beta_disco: float = 0.1,
                 alpha_cls: float = 0.5, 
                 **kwargs
    ): 
        super().__init__(*args, **kwargs) 

        self.temperature = temperature 
        self.alpha_adp = alpha_adp 
        self.alpha_disco = alpha_disco
        self.beta_disco = beta_disco
        self.alpha_cls = alpha_cls

        self.entropies = self.algorithm_config['entropies']
        self.aggregate_cluster_algorithm = self.algorithm_config.get('aggregate_cluster_algo', 'fedavg')
        self.num_classes_per_cluster = self.algorithm_config['num_class_per_cluster']
        
        self.fedadp = FedAdp(*args, alpha=self.alpha_adp, **kwargs)
        self.feddisco = FedDisco(*args, dk=self.algorithm_config['dk'], alpha=self.alpha_disco, beta=self.beta_disco, **kwargs)
        self.fedcls = FedCLS(*args, alpha=self.alpha_cls, **kwargs)

    def __repr__(self): 
        return 'FedHCW'
    
    
    def aggregate_cluster(self, cluster_id, cluster_clients: List[FitRes]):
        weight_results = [(parameters_to_ndarrays(fit_res.parameters),
                            fit_res.num_examples * np.exp(self.entropies[int(fit_res.metrics["id"])]/self.temperature))
                            for fit_res in cluster_clients]
        losses = [fit_res.num_examples * fit_res.metrics["loss"] for fit_res in cluster_clients]
        correct = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for fit_res in cluster_clients]
        examples = [fit_res.num_examples for fit_res in cluster_clients]
        loss = sum(losses) / sum(examples)
        accuracy = sum(correct) / sum(examples)

        aggregated_params = ndarrays_to_parameters(aggregate(weight_results))

        total_examples = sum(fit_res.num_examples for fit_res in cluster_clients)

        representative_metrics = dict(cluster_clients[0].metrics)

        representative_metrics["id"] = cluster_id
        representative_metrics["loss"] = loss
        representative_metrics["accuracy"] = accuracy

        if self.aggregate_cluster_algorithm == 'fedcls':
            representative_metrics["num_classes"] = self.num_classes_per_cluster[cluster_id]
        # print([fit_res.metrics["id"] for fit_res in cluster_clients])
        return FitRes(parameters=aggregated_params,
                      num_examples=total_examples,
                      metrics=representative_metrics,
                      status=Status(code=0, message="Aggregated successfully")
                    )


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        cluster_data = {}

        for client_res in results:
            client, fit_res = client_res
            cluster_id = fit_res.metrics["cluster_id"]
            if cluster_id not in cluster_data:
                cluster_data[cluster_id] = []

            cluster_data[cluster_id].append(fit_res)

        cluster_results = {}
        for cluster_id, fit_res_list in cluster_data.items():
            if len(fit_res_list) > 1:
                fit_res = self.aggregate_cluster(cluster_id, fit_res_list)
            else:
                fit_res = fit_res_list[0]

            cluster_results[cluster_id] = fit_res

        if self.aggregate_cluster_algorithm == 'fedavg': 
            self.current_parameters, metrics_aggregated = super().aggregate_fit(
                server_round, 
                [(None, fit_res) for _, fit_res in cluster_results.items()], 
                []
            )
        elif self.aggregate_cluster_algorithm == 'fedcls': 
            self.current_parameters, metrics_aggregated = self.fedcls.aggregate_fit(
                server_round, 
                [(None, fit_res) for _, fit_res in cluster_results.items()], 
                []
            )
        elif self.aggregate_cluster_algorithm == 'feddisco':
            self.current_parameters, metrics_aggregated = self.feddisco.aggregate_fit(
                server_round, 
                [(None, fit_res) for _, fit_res in cluster_results.items()], 
                []
            )
        elif self.aggregate_cluster_algorithm == 'fedadp': 
            self.current_parameters, metrics_aggregated = self.fedadp.aggregate_fit(
                server_round, 
                [(None, fit_res) for _, fit_res in cluster_results.items()], 
                []
            )

        num_examples = [fit_res.num_examples for _, fit_res in cluster_results.items()]
        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in cluster_results.items()]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in cluster_results.items()]

        loss = sum(losses) / sum(num_examples)
        accuracy = sum(corrects) / sum(num_examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated