
import torch
import torch.nn as nn
from .rate import rate_function_from_losses, inverse_rate_function_from_losses
from .cumulant import eval_cumulant_from_losses, get_loss


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
    
    

class OnlineCumulant:
    def __init__(self, model, buffer_size, loss_fn=nn.CrossEntropyLoss(reduction="none"), verbose=False):
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
        verbose : bool, optional
            If True, display a progress bar. Default is False.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.buffer_size = buffer_size

        # Initialize the losses buffer
        self.losses_weighted = torch.ones(buffer_size)
        self.losses = torch.empty(0)
        
    def update_losses(self, inputs, targets):

        # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
        self.model.eval()
                
        logits = self.model(inputs)
            
        # Compute the loss between the predictions (logits) and the true labels
        new_losses = self.loss_fn(logits, targets)
        
        self.losses = torch.cat([self.losses, new_losses])
        if len(self.losses) > self.buffer_size:
            self.losses = self.losses[-self.buffer_size:]
        
        # Scale the weight of old losses by 0.9
        self.losses_weighted[:-len(new_losses)] *= 0.9


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