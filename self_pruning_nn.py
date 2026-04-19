import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# Define device specifically for optimal CUDA performance if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PrunableLinear(nn.Module):
    """
    Part 1: The 'Prunable' Linear Layer
    A custom linear layer that learns to prune its own weights dynamically during training.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Standard weight and bias parameters
        # Shape: (out_features, in_features) for weights as expected by F.linear
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # 2. Gate scores tensor with exact same shape as weight tensor
        # This will be passed through a sigmoid to generate probabilistic gating values in [0, 1]
        self.gate_scores = nn.Parameter(torch.empty((out_features, in_features)))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization for continuous gradients (PyTorch default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Bias initialization calculated based on structural fan-in
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize gate scores. 
        # By initializing with a small positive constant like 1.0, 
        # the initial gates are active (sigmoid(1.0) ~ 0.73). 
        # This allows the network to start training normally before strictly identifying what must prune limitlessly.
        nn.init.constant_(self.gate_scores, 1.0)

    def forward(self, input):
        # 3. Apply transformation mapping to gate_scores -> bounds between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # Calculate isolated "pruned weights" safely via elemental-multiplication 
        pruned_weights = self.weight * gates
        
        # Perform the structurally standard linear layer formulation using these adapted weights
        return F.linear(input, pruned_weights, self.bias)

    def get_sparsity_loss(self):
        # L1 norm estimation of evaluated active connections. 
        # Since gates = sigmoid(gate_scores) are bounded entirely in [0,1], the L1 norm behaves exactly identical to their mathematical sum.
        gates = torch.sigmoid(self.gate_scores)
        return torch.sum(gates)

    def get_sparsity_level(self, threshold=1e-2):
        # Calculates exactly the isolated count of fully pruned correlations mapped below the threshold
        gates = torch.sigmoid(self.gate_scores)
        pruned_count = torch.sum(gates < threshold).item()
        total_count = gates.numel()
        return pruned_count, total_count

    def get_gate_values(self):
        # Flatten extraction arrays explicitly for final matplotlib visual plotting
        return torch.sigmoid(self.gate_scores).detach().cpu().numpy().flatten()


class PrunableMLP(nn.Module):
    """
    Standard feed-forward neural network for specific CIFAR-10 image classifications. 
    Augmenting layer abstractions purely with our custom parameter structures.
    """
    def __init__(self):
        super(PrunableMLP, self).__init__()
        self.flatten = nn.Flatten()
        # Initial image Input size configuration logic: 3 color channels * 32 * 32 standard pixels = 3072 input dimensionality
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)  # Standard 10 target classes bounding

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_total_sparsity_loss(self):
        # Method handles propagating cumulative network summation globally for regularizer integration natively
        return self.fc1.get_sparsity_loss() + self.fc2.get_sparsity_loss() + self.fc3.get_sparsity_loss()

    def get_sparsity_level(self, threshold=1e-2):
        # Collect ratio values resolving the isolated pruned properties of the overall network explicitly
        prun1, tot1 = self.fc1.get_sparsity_level(threshold)
        prun2, tot2 = self.fc2.get_sparsity_level(threshold)
        prun3, tot3 = self.fc3.get_sparsity_level(threshold)
        return float(prun1 + prun2 + prun3) / (tot1 + tot2 + tot3) * 100.0

    def get_all_gate_values(self):
        # Sequential gathering logic linking to layer array processing
        g1 = self.fc1.get_gate_values()
        g2 = self.fc2.get_gate_values()
        g3 = self.fc3.get_gate_values()
        return np.concatenate([g1, g2, g3])


def train_eval_model(lambda_reg_val, num_epochs=20):
    """
    Part 3: Complete execution pipeline testing integrated logic implementation
    """
    print(f"\n{'='*50}\nTraining Self-Pruning Network [Lambda = {lambda_reg_val}]\n{'='*50}")
    
    model = PrunableMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Adding Cosine-annealing learning rate scheduler 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Utilizing data augmentations (RandomCrop, RandomHorizontalFlip) to prevent overfitting
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])
    
    # Standardized num_workers limit safely ensuring 100% OS compatibilities across user hardware
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    
    # Commencing standard epoch training iteration constraints 
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
            
            # Generating Sparse values from internally modeled variables mappings
            sp_loss = model.get_total_sparsity_loss()
            
            # Part 2 requirement mapping: Total Loss structurally balanced mapping classification vs active constraints 
            loss = cls_loss + lambda_reg_val * sp_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            cls_loss_total += cls_loss.item()
            sp_loss_total += sp_loss.item()
            
        # Advancing the learning rate scheduler at the end of each epoch
        scheduler.step()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Total Loss: {running_loss/len(trainloader):.4f} | "
              f"Cls Loss: {cls_loss_total/len(trainloader):.4f} | "
              f"Reg Loss: {sp_loss_total/len(trainloader):.4f}")
        
    print("Training finished. Processing test evaluations...")
    
    # Model evaluation metrics 
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
    sparsity = model.get_sparsity_level()
    gates = model.get_all_gate_values()
    
    print(f">> Test Accuracy Result: {accuracy:.2f}% | Network Sparsity Limit (<1e-2): {sparsity:.2f}%")
    return accuracy, sparsity, gates


if __name__ == "__main__":
    # Ensure reproducability baseline metrics correctly process natively on system
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Testing specific lambda regularizations exploring Sparsity vs. Accuracy trade-offs
    lambda_values = [1e-4, 1e-3, 5e-3]
    
    # Epoch limits matching expected performance modeling
    num_epochs = 20
    
    results = []
    best_results = {'sparsity': -1, 'gates': None, 'lambda': None}
    
    for l_val in lambda_values:
        acc, spars, gates = train_eval_model(l_val, num_epochs)
        results.append({'lambda': l_val, 'accuracy': acc, 'sparsity': spars})
        
        # Caches properties of models natively demonstrating active pruning capabilities vs baseline accuracy preservations
        if spars > best_results['sparsity'] and acc > 30: 
            best_results.update({'sparsity': spars, 'gates': gates, 'lambda': l_val})
            
    # Systematically generated terminal output displaying values 
    print("\n" + "="*50)
    print("FINAL ALGORITHM RESULTS SUMMARY")
    print("="*50)
    print(f"{'Lambda':<10} | {'Test Accuracy (%)':<20} | {'Sparsity Level (%)':<20}")
    print("-" * 55)
    for res in results:
        print(f"{res['lambda']:<10} | {res['accuracy']:<20.2f} | {res['sparsity']:<20.2f}")
        
    # Null recovery handling 
    if best_results['gates'] is None:
        best_results['gates'] = gates
        best_results['lambda'] = lambda_values[-1]
        
    # Plot generation rendering specifically visualising histogram sparsity
    plt.figure(figsize=(10, 6))
    plt.hist(best_results['gates'], bins=50, alpha=0.75, color='royalblue', edgecolor='black')
    plt.title(f'Distribution of Final Gate Values ($\lambda$ = {best_results["lambda"]})')
    plt.xlabel('Gate Value ($\sigma(gate\_scores)$)')
    plt.ylabel('Frequency (Number of Model Weights)')
    
    # Marked boundaries specifically for analytic verifications of L1 norm interactions 
    plt.axvline(x=0.01, color='red', linestyle='dashed', linewidth=2, label='Pruning Threshold Boundary (<0.01)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_filename = 'gate_distribution.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nDistribution plot generated properly and securely rendered to: {plot_filename}")
