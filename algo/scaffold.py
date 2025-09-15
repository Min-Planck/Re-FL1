from algo import *

class Scaffold(FedAvg):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.server_controls  = {}
        self.client_controls  = {}
        self.num_model_params = None

    def __repr__(self) -> str:
        return "Scaffold"
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        weights = parameters_to_ndarrays(parameters)
        if not self.server_controls: 
            self.server_controls = [np.zeros_like(w) for w in weights]

        if self.num_model_params is None:
            self.num_model_params = len(weights)

        instructions = []
        for client in clients:
            cid = int(client.cid)
            if cid not in self.client_controls.keys():
                self.client_controls[cid] = [np.zeros_like(w) for w in weights]
            client_control = self.client_controls[cid]
            # Combine parameters: [model_weights, server_control, client_control]
            combined_weights = [*weights, *self.server_controls, *client_control]
            parameters_with_control = ndarrays_to_parameters(combined_weights)
            # Create FitIns with combined parameters
            fit_ins = FitIns(parameters=parameters_with_control, config={"learning_rate": self.learning_rate})
            instructions.append((client, fit_ins))

        return instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Sum number of examples for weighting
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        # Prepare accumulators
        sum_model_params = [np.zeros_like(c) for c in self.server_controls]
        sum_control_updates = [np.zeros_like(c) for c in self.server_controls]

        # Iterate over client results
        for client, fit_res in results:
            cid = fit_res.metrics['id']
            res_weights = parameters_to_ndarrays(fit_res.parameters)
            
            # Split parameters
            model_params = res_weights[:self.num_model_params]
            client_control_update = res_weights[2*self.num_model_params:3*self.num_model_params]
            
            # Accumulate model parameters (example-weighted)
            for idx, w in enumerate(model_params):
                sum_model_params[idx] += w * fit_res.num_examples
                
            # Accumulate control updates (uniform weighting)
            for idx, cv in enumerate(client_control_update):
                sum_control_updates[idx] += cv
                
            # Update stored client control
            if cid in self.client_controls:
                for idx in range(len(self.client_controls[cid])):
                    self.client_controls[cid][idx] += client_control_update[idx]

        # Compute weighted average of model parameters
        new_global_weights = [param_sum / total_examples for param_sum in sum_model_params]
        
        # Compute average control update (uniform average)
        avg_control_update = [
            cv_sum / len(results)
            for cv_sum in sum_control_updates
        ]
        
        # Update server control variate
        total_clients = len(self.client_controls)
        cv_multiplier = len(results) / total_clients if total_clients > 0 else 1.0
        for idx in range(len(self.server_controls)):
            self.server_controls[idx] += cv_multiplier * avg_control_update[idx]

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        # Create Parameters object
        loss = sum(losses) / total_examples
        accuracy = sum(corrects) / total_examples

        self.current_parameters = ndarrays_to_parameters(new_global_weights)
        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        return self.current_parameters, {}