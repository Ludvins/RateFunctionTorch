import torch
from tqdm import tqdm
from torch import nn
from ratefunctiontorch.cumulant import get_loss, eval_cumulant_from_losses
import numpy as np

@torch.no_grad()
def rate_function(model, 
                  evaluation_points, 
                  loader, 
                  loss_fn=nn.CrossEntropyLoss(reduction="none"),
                  return_lambdas=False,
                  return_cummulants=False,
                  epsilon=0.01,
                  max_lambda=100000,
                  strategy="TernarySearch",
                  verbose=False):
    """
    Compute the rate function of a model over a given set of evaluation points, based on 
    the data provided by the DataLoader and a loss function.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    evaluation_points : torch.Tensor
        Points at which the rate function will be computed.
    loader : torch.utils.data.DataLoader
        The DataLoader providing data for evaluation.
    loss_fn : callable, optional
        The loss function used for evaluation. Default is nn.CrossEntropyLoss(reduction="none").
    return_lambdas : bool, optional
        Whether to return the corresponding lambda values that maximize the auxiliary function.
    return_cummulants : bool, optional
        Whether to return the computed cummulants.
    epsilon : float, optional
        The precision for ternary search. Default is 0.01.
    max_lambda : float, optional
        Maximum value for lambda in the search range. Default is 100000.
    strategy : str, optional
        Strategy used to compute the rate function. Currently only "TernarySearch" is supported.
    verbose : bool, optional
        If True, display a progress bar. Default is False.
        
    Returns
    -------
    torch.Tensor or tuple
        The rate function values. If return_lambdas or return_cummulants is True, additional 
        arrays with lambdas or cummulants are returned.
    """
    assert strategy in ["TernarySearch"], "Invalid strategy: only 'TernarySearch' is supported"
    
    if strategy == "TernarySearch":
        # Precompute losses
        losses = get_loss(model, loader, loss_fn)
        
        # Initialize progress bar if verbose
        evaluation_points = tqdm(evaluation_points) if verbose else evaluation_points
        
        rates = []
        cummulants = []
        lambdas = []
        
        # Loop through each evaluation point
        for a in evaluation_points:
            a = torch.tensor(a).to(losses.device)
            
            # Set bounds for lambda based on the sign of 'a'
            if a < 0:
                min_lambda = torch.tensor(-max_lambda).to(losses.device)
                max_lambda = torch.tensor(0).to(losses.device)
            else:
                min_lambda = torch.tensor(0).to(losses.device)
                max_lambda = torch.tensor(max_lambda).to(losses.device)
                
            # Define the auxiliary function: lambda * a - cumulant(lambda)
            def auxiliar_function(lamb):
                cummulant = eval_cumulant_from_losses(losses, torch.tensor([lamb])).item()
                return lamb * a - cummulant
            
            # Perform ternary search to find the maximum
            rate, lambda_star = ternary_search_max(auxiliar_function,
                                                   min_lambda,
                                                   max_lambda,
                                                   epsilon=epsilon)
            
            # Adjust sign of the rate based on the sign of 'a'
            rate = torch.sign(a) * rate
            
            # Collect lambda and cumulant if requested
            if return_lambdas:
                lambdas.append(lambda_star)
            if return_cummulants:
                cummulants.append(-rate + lambda_star * a)
            
            # Append the rate to the result list
            rates.append(rate)
        
        # Return requested results
        if return_lambdas and return_cummulants:
            return np.array(rates), np.array(lambdas), np.array(cummulants)
        elif return_lambdas:
            return np.array(rates), np.array(lambdas)
        elif return_cummulants:
            return np.array(rates), np.array(cummulants)
        else:
            return np.array(rates)


@torch.no_grad()
def inverse_rate_function(model, 
                          evaluation_points, 
                          loader, 
                          loss_fn=nn.CrossEntropyLoss(reduction="none"),
                          return_lambdas=False,
                          return_cummulants=False,
                          epsilon=0.01,
                          max_lambda=100000,
                          strategy="TernarySearch", 
                          verbose=False):
    """
    Compute the inverse of the rate function for the model at specific evaluation points.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    evaluation_points : torch.Tensor
        Points at which the inverse rate function will be computed.
    loader : torch.utils.data.DataLoader
        The DataLoader providing data for evaluation.
    loss_fn : callable, optional
        The loss function used for evaluation. Default is nn.CrossEntropyLoss(reduction="none").
    return_lambdas : bool, optional
        Whether to return the corresponding lambda values that minimize the auxiliary function.
    return_cummulants : bool, optional
        Whether to return the computed cummulants.
    epsilon : float, optional
        The precision for ternary search. Default is 0.01.
    max_lambda : float, optional
        Maximum value for lambda in the search range. Default is 100000.
    strategy : str, optional
        Strategy used to compute the inverse rate function. Currently only "TernarySearch" is supported.
    verbose : bool, optional
        If True, display a progress bar. Default is False.
        
    Returns
    -------
    torch.Tensor or tuple
        The inverse rate function values. If return_lambdas or return_cummulants is True, additional 
        arrays with lambdas or cummulants are returned.
    """
    assert strategy in ["TernarySearch"], "Invalid strategy: only 'TernarySearch' is supported"
    
    if strategy == "TernarySearch":
        # Precompute losses
        losses = get_loss(model, loader, loss_fn)
        
        # Initialize progress bar if verbose
        evaluation_points = tqdm(evaluation_points) if verbose else evaluation_points
        
        rates = []
        cummulants = []
        lambdas = []
        
        # Loop through each evaluation point
        for a in evaluation_points:
            # Set bounds for lambda based on the sign of 'a'
            if a < 0:
                min_lambda = torch.tensor(-max_lambda).to(losses.device)
                max_lambda = torch.tensor(0).to(losses.device)
            else:
                min_lambda = torch.tensor(0).to(losses.device)
                max_lambda = torch.tensor(max_lambda).to(losses.device)
                
            # Define the auxiliary function for inverse rate:
            # This is the infimum of (cummulant(lambda) + a) / lambda
            def auxiliar_function(lamb):
                cummulant = eval_cumulant_from_losses(losses, torch.tensor([lamb])).item()
                return (cummulant + a) / lamb
            
            # Perform ternary search to find the minimum
            rate, lambda_star = ternary_search_min(auxiliar_function, min_lambda, max_lambda, epsilon=epsilon)
            
            # Collect lambda and cumulant if requested
            if return_lambdas:
                lambdas.append(lambda_star)
            if return_cummulants:
                cummulants.append(rate*lambda_star - a)
            
            # Append the rate to the result list
            rates.append(rate)
        
        # Return requested results
        if return_lambdas and return_cummulants:
            return np.array(rates), np.array(lambdas), np.array(cummulants)
        elif return_lambdas:
            return np.array(rates), np.array(lambdas)
        elif return_cummulants:
            return np.array(rates), np.array(cummulants)
        else:
            return np.array(rates)
        

def ternary_search_max(function, low, high, epsilon):
    """
    Perform ternary search to find the maximum of a unimodal function.

    Parameters
    ----------
    function : callable
        The function to maximize.
    low : float
        The lower bound of the search.
    high : float
        The upper bound of the search.
    epsilon : float
        The precision of the search.
        
    Returns
    -------
    tuple
        The maximum value of the function and the corresponding argument.
    """
    while (high - low) > epsilon:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        # Evaluate function at mid1 and mid2
        f_mid1 = function(mid1)
        f_mid2 = function(mid2)

        # Narrow the search range based on the evaluated values
        if f_mid1 > f_mid2:
            high = mid2
        else:
            low = mid1
            
    mid = (low + high) / 2
    return function(mid), mid


def ternary_search_min(function, low, high, epsilon):
    """
    Perform ternary search to find the minimum of a unimodal function.

    Parameters
    ----------
    function : callable
        The function to minimize.
    low : float
        The lower bound of the search.
    high : float
        The upper bound of the search.
    epsilon : float
        The precision of the search.
        
    Returns
    -------
    tuple
        The minimum value of the function and the corresponding argument.
    """
    while (high - low) > epsilon:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        # Evaluate function at mid1 and mid2
        f_mid1 = function(mid1)
        f_mid2 = function(mid2)

        # Narrow the search range based on the evaluated values
        if f_mid1 < f_mid2:
            high = mid2
        else:
            low = mid1
            
    mid = (low + high) / 2
    return function(mid), mid
