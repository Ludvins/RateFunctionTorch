import torch
from torch import nn
import numpy as np

@torch.no_grad()
def get_loss(model, loader, loss_fn=nn.CrossEntropyLoss(reduction="none")):
    """
    Compute the loss for a given model using data from a DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    loader : torch.utils.data.DataLoader
        A DataLoader providing the dataset to evaluate.
    loss_fn : callable, optional
        The loss function to use. Default is nn.CrossEntropyLoss with no reduction.
        
    Returns
    -------
    torch.Tensor
        Tensor containing the losses for each batch in the loader.
    
    """
    # Get a parameter from the model to determine its device (GPU/CPU)
    p = next(model.parameters())
    device = p.device

    # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    
    losses = []
    
    # Loop over the data loader
    for data, targets in loader:
        # Move input data and targets to the appropriate device (same as the model)
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass: compute the model's output
        logits = model(data)
        
        # Compute the loss between the predictions (logits) and the true labels
        loss = loss_fn(logits, targets)
        
        # Append the loss for the current batch to the losses list
        losses.append(loss)
        
    # Concatenate all batch losses into a single tensor and return it
    return torch.cat(losses)


@torch.no_grad()
def eval_cumulant(model, evaluation_points, loader, loss_fn=nn.CrossEntropyLoss(reduction="none")):
    """
    Compute the cumulants of the loss distribution at specified evaluation points.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    evaluation_points : scalar, list, tuple or torch.Tensor
        Evaluation points (lambda values) where the cumulants are computed.
    loader : torch.utils.data.DataLoader
        A DataLoader providing the dataset to evaluate.
    loss_fn : callable, optional
        The loss function to use. Default is nn.CrossEntropyLoss with no reduction.
        
    Returns
    -------
    torch.Tensor
        Tensor containing the cumulants evaluated at each point in evaluation_points.
    """
    
    # Convert evaluation_points to tensor if it isn't already
    if not isinstance(evaluation_points, torch.Tensor):
        evaluation_points = torch.tensor(evaluation_points)
    
    # If evaluation_points is a scalar (0-d tensor), unsqueeze it to make it 1-d
    if evaluation_points.ndim == 0:
        evaluation_points = evaluation_points.unsqueeze(0)

    # Get the losses from the model on the loader data
    losses = get_loss(model, loader, loss_fn)

    
    # Compute the log-sum-exp constant for numerical stability
    logsumexp_constant = torch.log(torch.tensor(losses.shape[0], device=losses.device))
    
    # Compute the mean loss (L) over the dataset
    L = torch.mean(losses)
    
    cumulants = []
    
    # For each evaluation point (lambda), compute the corresponding cumulant
    for lamb in evaluation_points:
        # logsumexp helps stabilize computation of log-sum-exp for the cumulant
        cumulant = torch.logsumexp(-lamb * losses, 0) - logsumexp_constant + lamb * L
        cumulants.append(cumulant)
        
    # Stack the results into a tensor and return
    return torch.stack(cumulants)


def eval_cumulant_from_losses(losses, evaluation_points, requires_grad=False):
    """
    Compute the cumulants for a given set of precomputed losses at specified evaluation points.

    This function allows calculating cumulants when the losses are already available, avoiding 
    recomputation of the losses.

    Parameters
    ----------
    losses : torch.Tensor
        Tensor containing the precomputed losses.
    evaluation_points : scalar, list, tuple or torch.Tensor
        Evaluation points (lambda values) where the cumulants are computed.
        
    Returns
    -------
    torch.Tensor
        Tensor containing the cumulants evaluated at each point in evaluation_points.
    """
    # Convert evaluation_points to tensor if it isn't already
    if not isinstance(evaluation_points, torch.Tensor):
        evaluation_points = torch.tensor(evaluation_points)
    
    # If evaluation_points is a scalar (0-d tensor), unsqueeze it to make it 1-d
    if evaluation_points.ndim == 0:
        evaluation_points = evaluation_points.unsqueeze(0)
    
    with torch.set_grad_enabled(requires_grad):
        logsumexp_constant = torch.log(torch.tensor(losses.shape[0], device=losses.device))
        cumulants = []
        
        for lamb in evaluation_points:
            cumulant = torch.logsumexp(-lamb * losses, 0) - logsumexp_constant + torch.mean(lamb * losses)
            cumulants.append(cumulant)
    
        return torch.stack(cumulants)


def eval_cumulant_from_weighted_losses(losses, weights, evaluation_points, requires_grad=False):
    """
    Compute the cumulants for a given set of precomputed losses with weights at specified evaluation points.

    Parameters
    ----------
    losses : torch.Tensor
        Tensor containing the precomputed losses.
    weights : torch.Tensor
        Tensor containing the weights for each loss. Must be the same length as losses.
    evaluation_points : scalar, list, tuple or torch.Tensor
        Evaluation points (lambda values) where the cumulants are computed.
        
    Returns
    -------
    torch.Tensor
        Tensor containing the cumulants evaluated at each point in evaluation_points.
    """
    # Convert evaluation_points to tensor and detach
    if not isinstance(evaluation_points, torch.Tensor):
        evaluation_points = torch.tensor(evaluation_points)
    evaluation_points = evaluation_points.detach()
    
    # If evaluation_points is a scalar (0-d tensor), unsqueeze it to make it 1-d
    if evaluation_points.ndim == 0:
        evaluation_points = evaluation_points.unsqueeze(0)
    
    # Normalize weights to sum to 1 and detach
    weights = (weights / weights.sum()).detach()
    
    with torch.set_grad_enabled(requires_grad):
        cumulants = []
        
        for lamb in evaluation_points:
            cumulant = torch.logsumexp(-lamb * losses + torch.log(weights), 0) + torch.sum(weights * lamb * losses)
            cumulants.append(cumulant)
    
        return torch.stack(cumulants)
