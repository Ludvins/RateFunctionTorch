import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ratefunctiontorch import OnlineCumulant, RateCumulant

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import roc_auc_score
from torchvision import datasets
from tqdm import tqdm

from torch import optim, nn

import pandas as pd

STEPS = 10 
BATCH_SIZE = 250
BATCH_SIZE_VAL = 500
RANDOM_SEED = 12345678
BUFFER_SIZE = BATCH_SIZE_VAL
FORGETTING_FACTOR = 1.0

n_iters = 200


torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(RANDOM_SEED)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.cuda.manual_seed(RANDOM_SEED)
else:
    device = torch.device("cpu")

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Split the trainset into training and validation datasets
train_size = int(0.8 * len(trainset))  # 80% for training
val_size = len(trainset) - train_size  # 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

# Create data loaders for the new train and validation datasets
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE_VAL,
    shuffle=False,  # Typically, validation data is not shuffled
)

testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE_VAL,
    shuffle=True,
)



model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False
).to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
cce = nn.CrossEntropyLoss(reduction="none")


lambdas = torch.arange(-0.2, 0.5, step=0.01, device=device)
onlinecumulant = OnlineCumulant(device, BUFFER_SIZE, forgetting_factor=FORGETTING_FACTOR)


# Add these before the training loop
test_metrics = {
    'L_test': [],
    'alphaD_test': [],
    'L_val': [],
    'alphaD_val': [],
    'Loss_train': [],
    'batch_loss': [],
    'learning_rate': [],
    'lambdas_test': [],  # Add this line
    'cummulants_test': [],  # Add this line
    'lambdas_val': [],  # Add this line
    'cummulants_val': [],  # Add this line
    'it': []
}

# Initialize data iterator
data_iter = iter(train_loader)
# Update the data iterator for validation
val_data_iter = iter(val_loader)

iters_per_epoch = len(data_iter)
aux_loss = 0
loss_count = 0  # Counter to keep track of the number of losses added

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_iters,
    eta_min=1e-6
)

tq = tqdm(range(n_iters))
for it in tq:
    
    model.eval()

    # Get inputs and targets. If loader is exhausted, reinitialize.
    try:
        inputs_val, target_val = next(val_data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        val_data_iter = iter(test_loader)
        inputs_val, target_val = next(val_data_iter)

    # Move data to device
    inputs_val = inputs_val.to(device)
    target_val = target_val.to(device)

    onlinecumulant.update_losses(model, inputs_val, target_val)

    # Set model to train mode
    model.train()

    # Get inputs and targets. If loader is exhausted, reinitialize.
    try:
        inputs, target = next(data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        data_iter = iter(train_loader)
        inputs, target = next(data_iter)

    # Move data to device
    inputs = inputs.to(device)
    target = target.to(device)

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    logits = model(inputs)

    # Compute the loss
    loss = torch.mean(cce(logits, target))
    
    # Update aux_loss and loss_count
    aux_loss += loss.detach().cpu().numpy()
    loss_count += 1  # Increment the count

    # Calculate the average loss
    avg_loss = aux_loss / loss_count

    # Log the average loss
    tq.set_postfix(
        {"Train cce": avg_loss}
    )

    # Backward pass
    loss.backward()
    # Update the weights
    optimizer.step()
    scheduler.step()

    # Step the scheduler and check for early stopping
    if it % iters_per_epoch == 0 and it != 0:
        if avg_loss < 0.01:
            break
        aux_loss = 0
        loss_count = 0  # Reset the count after each epoch
        
    if it % STEPS == 0 and it > 0:
        # Add this line to store average training loss
        test_metrics['it'].append(it)
        test_metrics['Loss_train'].append(avg_loss)
        test_metrics['batch_loss'].append(loss.detach().cpu().numpy())

        # Clear cache before heavy operations
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        ratecumulant = RateCumulant(model, test_loader)
        #true_cumulant = ratecumulant.compute_cumulants(lambdas).detach().cpu().numpy()
        #approx_cummulant = onlinecumulant.compute_cumulants(lambdas).detach().cpu().numpy()

        L_test = ratecumulant.compute_mean()
        alphaD_test, lambdas_test, cummulants_test = ratecumulant.compute_rate_function(L_test - loss, return_lambdas=True, return_cummulants=True)
        L_val = onlinecumulant.compute_mean()
        alphaD_val, lambdas_val, cummulants_val = onlinecumulant.compute_rate_function(L_val - loss, return_lambdas=True, return_cummulants=True)
       
        # Store metrics - remove .item() calls since values are already floats
        test_metrics['L_test'].append(L_test.item())
        test_metrics['alphaD_test'].append(alphaD_test.item())
        test_metrics['L_val'].append(L_val.item())
        test_metrics['alphaD_val'].append(alphaD_val.item())
        test_metrics['lambdas_test'].append(lambdas_test.item())
        test_metrics['cummulants_test'].append(cummulants_test.item())
        test_metrics['lambdas_val'].append(lambdas_val.item())
        test_metrics['cummulants_val'].append(cummulants_val.item())


        current_lr = optimizer.param_groups[0]['lr']
        test_metrics['learning_rate'].append(current_lr)

        #plt.plot(lambdas.cpu(), true_cumulant, label = "True Cumulant", linestyle = "--", linewidth = 3)
        #plt.plot(lambdas.cpu(), approx_cummulant, label = "Approx Cumulant", linewidth = 3)
        #plt.title("Cumulant " + str(it))
        #plt.grid()
        #plt.legend()
        #plt.ylim(0,0.3)
        #plt.show()
        
        

# After training loop, save metrics to CSV

# Convert metrics to DataFrame
df = pd.DataFrame(test_metrics)
# Save to CSV
df.to_csv('./data/test_metrics.csv', index=False)
