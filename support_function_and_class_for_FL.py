import torch.nn.functional as F
from torch import nn
import torch
# ====== FedNTD ======
def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits

class NTD_Loss(nn.Module):

    def __init__(self, num_classes=10, tau=3, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss


# ====== MOON ======
def compute_accuracy(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss().to(device)

    loss_collector = []
 
    with torch.no_grad():
        for _, (x, target) in enumerate(dataloader):

            x = x.to(device)
            target = target.to(dtype=torch.int64, device=device)
            _, _, out = model(x)
            
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

        avg_loss = sum(loss_collector) / len(loss_collector)
   

    if was_training:
        model.train()

    return correct / float(total), avg_loss

def train_moon(
    net,
    global_net,
    previous_net,
    train_dataloader,
    lr,
    temperature,
    device="cpu",
    epochs=1,
    mu=1,
):
    """Training function for MOON."""
    net.to(device)
    global_net.to(device)
    previous_net.to(device)

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss().cuda()

    previous_net.eval()
    for param in previous_net.parameters():
        param.requires_grad = False

    cos = torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []

        for _, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            # pro1 is the representation by the current model (Line 14 of Algorithm 1)
            _, pro1, out = net(x)
            # pro2 is the representation by the global model (Line 15 of Algorithm 1)
            _, pro2, _ = global_net(x)
            # posi is the positive pair
            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            # pro 3 is the representation by the previous model (Line 16 of Algorithm 1)
            _, pro3, _ = previous_net(x)
            # nega is the negative pair
            nega = cos(pro1, pro3)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1) / temperature

            labels = torch.zeros(x.size(0), device=device).long()
            # compute the model-contrastive loss (Line 17 of Algorithm 1)
            loss2 = mu * criterion(logits, labels)
            # compute the cross-entropy loss (Line 13 of Algorithm 1)
            loss1 = criterion(out, target)
            # compute the loss (Line 18 of Algorithm 1)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)

    previous_net.to("cpu")
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    net.to("cpu")
    global_net.to("cpu")
    print(f">> Training accuracy: {train_acc:.6f}")
    print(" ** Training complete **")
    return net, epoch_loss, train_acc

# ===== Scaffold ======

def train_scaffold(
    net,
    trainloader,
    learning_rate,
    epochs,
    device, 
    client_control_old,
    server_control,
    client_control
):
    from utils import get_parameters
    from flwr.common import ndarrays_to_parameters
    correction_tensors = [
            torch.tensor(c_i - c_s, dtype=torch.float32, device=device)
            for c_i, c_s in zip(client_control_old, server_control)
        ]
    initial_weights = get_parameters(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    net.train() 
    total_loss, correct, total = 0.0, 0, 0
    num_batches = 0
    
    for _ in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            
            with torch.no_grad():
                for param, corr in zip(net.parameters(), correction_tensors):
                    if param.grad is not None:
                        param.grad -= corr
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            num_batches += 1

        # Compute updated weights (wT)
        updated_weights = get_parameters(net)

        
        # Calculate control update: Δc_i = -c_s + (w0 - wT)/(Kη)
        K = num_batches  # Number of local steps
        eta = learning_rate  # Learning rate
        control_update = [
            (w0 - wT) / (K * eta) - c_s
            for w0, wT, c_s in zip(initial_weights, updated_weights, server_control)
        ]
        
        # Update client control: c_i^{new} = c_i^{old} + Δc_i
        client_control = [
            c_old + delta 
            for c_old, delta in zip(client_control_old, control_update)
        ]
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': correct / total,
        'params': ndarrays_to_parameters(
                updated_weights + server_control + control_update
            ),
        'client_control': client_control
        }