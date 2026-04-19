import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PrunableLinear(nn.Module):
    """
    Part 1: The 'Prunable' Linear Layer
    A custom linear layer that learns to prune its own weights dynamically during training.

    Each weight has a paired gate_score parameter. The gate is computed as
    sigmoid(gate_score) ∈ (0, 1). Pruned weights = weight * gate.
    When gate ≈ 0, the connection is effectively removed.
    Gradients flow through both weight and gate_scores via autograd.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Gate scores — same shape as weight, also updated by optimizer
        self.gate_scores = nn.Parameter(torch.empty((out_features, in_features)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Init at 1.0 so sigmoid(1.0) ≈ 0.73 — gates start open, optimizer learns what to prune
        nn.init.constant_(self.gate_scores, 1.0)

    def forward(self, input):
        # Compute gates ∈ (0, 1)
        gates = torch.sigmoid(self.gate_scores)
        # Element-wise multiplication prunes weights by their gate values
        pruned_weights = self.weight * gates
        # Standard linear transform — gradients flow to both weight and gate_scores
        return F.linear(input, pruned_weights, self.bias)

    def get_sparsity_loss(self):
        # L1 norm of gates — since gates > 0, this equals their sum
        return torch.sum(torch.sigmoid(self.gate_scores))

    def get_sparsity_level(self, threshold=0.1):
        gates = torch.sigmoid(self.gate_scores)
        pruned_count = torch.sum(gates < threshold).item()
        total_count = gates.numel()
        return pruned_count, total_count

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores).detach().cpu().numpy().flatten()


class PrunableMLP(nn.Module):
    """
    Feed-forward network for CIFAR-10 using PrunableLinear layers throughout.
    Input: 3072 (flattened 32x32x3), Output: 10 classes.
    """
    def __init__(self):
        super(PrunableMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_total_sparsity_loss(self):
        # Sum L1 penalty across all prunable layers
        return self.fc1.get_sparsity_loss() + self.fc2.get_sparsity_loss() + self.fc3.get_sparsity_loss()

    def get_sparsity_level(self, threshold=0.1):
        prun1, tot1 = self.fc1.get_sparsity_level(threshold)
        prun2, tot2 = self.fc2.get_sparsity_level(threshold)
        prun3, tot3 = self.fc3.get_sparsity_level(threshold)
        return float(prun1 + prun2 + prun3) / (tot1 + tot2 + tot3) * 100.0

    def get_all_gate_values(self):
        return np.concatenate([
            self.fc1.get_gate_values(),
            self.fc2.get_gate_values(),
            self.fc3.get_gate_values()
        ])


def train_eval_model(lambda_reg_val, num_epochs=20):
    """
    Part 3: Train the self-pruning network and evaluate test accuracy + sparsity.
    Total Loss = CrossEntropyLoss + lambda * SparsityLoss (L1 on all gates)
    """
    print(f"\n{'='*50}\nTraining Self-Pruning Network [Lambda = {lambda_reg_val}]\n{'='*50}")

    model = PrunableMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        cls_loss_total = 0.0
        sp_loss_total = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            cls_loss = criterion(outputs, labels)
            sp_loss = model.get_total_sparsity_loss()

            # Part 2: Total Loss = ClassificationLoss + λ * SparsityLoss
            loss = cls_loss + lambda_reg_val * sp_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            cls_loss_total += cls_loss.item()
            sp_loss_total += sp_loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Total Loss: {running_loss/len(trainloader):.4f} | "
              f"Cls Loss: {cls_loss_total/len(trainloader):.4f} | "
              f"Reg Loss: {sp_loss_total/len(trainloader):.4f}")

    print("Training finished. Processing test evaluations...")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    sparsity = model.get_sparsity_level(threshold=0.1)
    gates = model.get_all_gate_values()
    mean_gate = gates.mean()

    print(f">> Test Accuracy: {accuracy:.2f}% | Sparsity (gate < 0.1): {sparsity:.2f}% | Mean Gate: {mean_gate:.4f}")
    return accuracy, sparsity, gates


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Three lambda values: low / medium / high — shows sparsity-accuracy trade-off
    lambda_values = [1e-4, 1e-3, 5e-3]
    num_epochs = 20

    results = []
    all_gates_list = []

    for l_val in lambda_values:
        acc, spars, gates = train_eval_model(l_val, num_epochs)
        results.append({'lambda': l_val, 'accuracy': acc, 'sparsity': spars})
        all_gates_list.append((acc, spars, gates))

    print("\n" + "="*55)
    print("RESULTS SUMMARY")
    print("="*55)
    print(f"{'Lambda':<10} | {'Test Accuracy (%)':<20} | {'Sparsity (<0.1) (%)'}")
    print("-" * 55)
    for res in results:
        print(f"{res['lambda']:<10} | {res['accuracy']:<20.2f} | {res['sparsity']:.2f}")

    # Plot: side-by-side gate distributions for all three lambda values
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    colors = ['steelblue', 'royalblue', 'darkblue']
    for ax, res, (_, __, stored_gates), color in zip(axes, results, all_gates_list, colors):
        x_max = max(min(float(np.percentile(stored_gates, 99.9)) * 1.5, 0.3), 0.15)
        ax.hist(stored_gates, bins=100, range=(0, x_max), alpha=0.80, color=color, edgecolor='none')
        ax.axvline(x=0.1, color='red', linestyle='--', linewidth=1.5, label='Threshold (0.1)')
        ax.set_title(
            f'λ = {res["lambda"]}\nAcc: {res["accuracy"]:.1f}%  |  Sparsity: {res["sparsity"]:.1f}%',
            fontsize=11)
        ax.set_xlabel('Gate Value σ(gate_scores)', fontsize=10)
        ax.set_ylabel('Number of Weights', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle('Distribution of Final Gate Values Across λ Settings', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gate_distribution.png', dpi=200, bbox_inches='tight')
    print("\nPlot saved to: gate_distribution.png")