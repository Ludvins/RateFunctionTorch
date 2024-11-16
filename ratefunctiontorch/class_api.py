import torch
import torch.nn as nn
from .rate import rate_function_from_losses, inverse_rate_function_from_losses
from .cumulant import eval_cumulant_from_losses, eval_cumulant_from_weighted_losses, get_loss


class RateCumulant:
    def __init__(self, model, loader, loss_fn=nn.CrossEntropyLoss(reduction="none"), epsilon=0.01, max_lambda=100000, strategy="TernarySearch", verbose=False):
        """
        Initialize the RateFunctionAPI with a model and dataloader.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        loader : torch.utils.data.DataLoader
            The DataLoader providing data for evaluation.
        loss_fn : callable, optional
            The loss function used for evaluation. Default is nn.CrossEntropyLoss(reduction="none").
        epsilon : float, optional
            The precision for ternary search. Default is 0.01.
        max_lambda : float, optional
            Maximum value for lambda in the search range. Default is 100000.
        strategy : str, optional
            Strategy used to compute the rate function. Currently only "TernarySearch" is supported.
        verbose : bool, optional
            If True, display a progress bar. Default is False.
        """
        self.model = model
        self.loader = loader
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.max_lambda = max_lambda
        self.strategy = strategy
        self.verbose = verbose

        # Precompute losses and store them
        self.losses = get_loss(self.model, self.loader, self.loss_fn)

    def compute_rate_function(self, evaluation_points, return_lambdas=False, return_cummulants=False):
        """
        Compute the rate function at given evaluation points.
        
        Parameters
        ----------
        evaluation_points : scalar, list, tuple or torch.Tensor
            Points at which the rate function will be computed.
        return_lambdas : bool, optional
            Whether to return the corresponding lambda values that maximize the auxiliary function.
        return_cummulants : bool, optional
            Whether to return the computed cumulants.
            
        Returns
        -------
        torch.Tensor or tuple
            The rate function values and optionally lambdas or cumulants.
        """
        return rate_function_from_losses(
            self.losses,
            evaluation_points,
            return_lambdas=return_lambdas,
            return_cummulants=return_cummulants,
            epsilon=self.epsilon,
            max_lambda=self.max_lambda,
            strategy=self.strategy,
            verbose=self.verbose
        )

    def compute_inverse_rate_function(self, evaluation_points, return_lambdas=False, return_cummulants=False):
        """
        Compute the inverse rate function at given evaluation points.
        
        Parameters
        ----------
        evaluation_points : scalar, list, tuple or torch.Tensor
            Points at which the inverse rate function will be computed.
        return_lambdas : bool, optional
            Whether to return the corresponding lambda values that minimize the auxiliary function.
        return_cummulants : bool, optional
            Whether to return the computed cumulants.
            
        Returns
        -------
        torch.Tensor or tuple
            The inverse rate function values and optionally lambdas or cumulants.
        """
        return inverse_rate_function_from_losses(
            self.losses,
            evaluation_points,
            return_lambdas=return_lambdas,
            return_cummulants=return_cummulants,
            epsilon=self.epsilon,
            max_lambda=self.max_lambda,
            strategy=self.strategy,
            verbose=self.verbose
        )
    
    def compute_cumulants(self, evaluation_points):
        """
        Compute the cumulants at given evaluation points using precomputed losses.
        
        Parameters
        ----------
        evaluation_points : scalar, list, tuple or torch.Tensor
            Points at which the cumulants will be computed.
            
        Returns
        -------
        torch.Tensor
            The cumulants evaluated at each point in evaluation_points.
        """
        return eval_cumulant_from_losses(self.losses, evaluation_points)
    
    def compute_mean(self):
        """
        Compute the arithmetic mean of the stored losses.
        
        Returns
        -------
        float
            The mean of all losses.
        """
        return torch.mean(self.losses)
    
    def compute_variance(self):
        """
        Compute the variance of the stored losses.
        
        Returns
        -------
        float
            The variance of all losses.
        """
        mean_loss = torch.mean(self.losses)
        variance = torch.mean((self.losses - mean_loss) ** 2)
        return variance.item()  # Return as a Python float
    
    

class OnlineCumulant:
    def __init__(self, device, buffer_size, forgetting_factor=1.0, loss_fn=nn.CrossEntropyLoss(reduction="none"), 
                 epsilon=0.01, max_lambda=100000, strategy="TernarySearch", verbose=False):
        """
        Initialize the OnlineCumulant with a device and buffer size.
        
        Parameters
        ----------
        device : torch.device
            The device to store tensors on (CPU, CUDA, or MPS).
        buffer_size : int
            Size of the buffer to store losses.
        forgetting_factor : float, optional
            Factor to decay old weights (between 0 and 1). Default is 1.0 (no forgetting).
            Lower values give more importance to recent samples.
        loss_fn : callable, optional
            The loss function used for evaluation. Default is nn.CrossEntropyLoss(reduction="none").
        epsilon : float, optional
            The precision for ternary search. Default is 0.01.
        max_lambda : float, optional
            Maximum value for lambda in the search range. Default is 100000.
        strategy : str, optional
            Strategy used to compute the rate function. Default is "TernarySearch".
        verbose : bool, optional
            If True, display a progress bar. Default is False.
        """
        self.device = device
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.buffer_size = buffer_size
        self.forgetting_factor = forgetting_factor
        self.epsilon = epsilon
        self.max_lambda = max_lambda
        self.strategy = strategy
        
        # Initialize empty tensors for both losses and weights
        self.losses = torch.empty(0, device=self.device)
        self.losses_weighted = torch.empty(0, device=self.device)
    
    def update_losses(self, model, inputs, targets):
        """
        Update losses using the provided model and data.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        inputs : torch.Tensor
            Input data.
        targets : torch.Tensor
            Target labels.
        """
        # Set the model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            logits = model(inputs)
            new_losses = self.loss_fn(logits, targets)
        
        # Add new weights (1.0) for new losses
        new_weights = torch.ones(len(new_losses), device=self.device)
        
        # Update both losses and weights, using forgetting_factor for old weights
        self.losses = torch.cat([self.losses, new_losses])
        self.losses_weighted = torch.cat([self.losses_weighted * self.forgetting_factor, new_weights])
        
        # If buffer size is exceeded, keep only the most recent ones
        if len(self.losses) > self.buffer_size:
            self.losses = self.losses[-self.buffer_size:]
            self.losses_weighted = self.losses_weighted[-self.buffer_size:]
    
    def compute_cumulants(self, evaluation_points):
        """
        Compute the cumulants at given evaluation points using precomputed losses.
        
        Parameters
        ----------
        evaluation_points : scalar, list, tuple or torch.Tensor
            Points at which the cumulants will be computed.
            
        Returns
        -------
        torch.Tensor
            The cumulants evaluated at each point in evaluation_points.
        """
        
        return eval_cumulant_from_weighted_losses(self.losses, self.losses_weighted, evaluation_points)
    
    def compute_rate_function(self, evaluation_points, return_lambdas=False, return_cummulants=False):
        """
        Compute the rate function at given evaluation points using weighted losses.
        
        Parameters
        ----------
        evaluation_points : scalar, list, tuple or torch.Tensor
            Points at which the rate function will be computed.
        return_lambdas : bool, optional
            Whether to return the corresponding lambda values that maximize the auxiliary function.
        return_cummulants : bool, optional
            Whether to return the computed cumulants.
            
        Returns
        -------
        torch.Tensor or tuple
            The rate function values and optionally lambdas or cumulants.
        """
        return rate_function_from_losses(
            self.losses,
            evaluation_points,
            return_lambdas=return_lambdas,
            return_cummulants=return_cummulants,
            epsilon=self.epsilon,
            max_lambda=self.max_lambda,
            strategy=self.strategy,
            verbose=self.verbose,
            weights=self.losses_weighted
        )
    
    def compute_inverse_rate_function(self, evaluation_points, return_lambdas=False, return_cummulants=False):
        """
        Compute the inverse rate function at given evaluation points using weighted losses.
        
        Parameters
        ----------
        evaluation_points : scalar, list, tuple or torch.Tensor
            Points at which the inverse rate function will be computed.
        return_lambdas : bool, optional
            Whether to return the corresponding lambda values that minimize the auxiliary function.
        return_cummulants : bool, optional
            Whether to return the computed cumulants.
            
        Returns
        -------
        torch.Tensor or tuple
            The inverse rate function values and optionally lambdas or cumulants.
        """
        return inverse_rate_function_from_losses(
            self.losses,
            evaluation_points,
            return_lambdas=return_lambdas,
            return_cummulants=return_cummulants,
            epsilon=self.epsilon,
            max_lambda=self.max_lambda,
            strategy=self.strategy,
            verbose=self.verbose,
            weights=self.losses_weighted
        )
        
    def compute_mean(self):
        """
        Compute the weighted mean of the stored losses using the current weights.
        
        Returns
        -------
        float
            The weighted mean of all losses.
        """
        # Normalize weights to sum to 1
        weights = self.losses_weighted / self.losses_weighted.sum()
        return (weights * self.losses).sum()
    
    def compute_variance(self):
        """
        Compute the variance of the stored losses using the current weights.
        
        Returns
        -------
        float
            The weighted variance of all losses.
        """
        if len(self.losses) == 0:
            return 0.0  # Return 0 if there are no losses
        
        weights = self.losses_weighted / self.losses_weighted.sum()  # Normalize weights
        mean_loss = (weights * self.losses).sum()  # Weighted mean
        variance = (weights * (self.losses - mean_loss) ** 2).sum()  # Weighted variance
        return variance.item()  # Return as a Python float