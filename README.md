
# Rate Function and Cumulant Computation

This repository provides PyTorch-based utilities for evaluating rate functions and cumulants for machine learning models. The code is designed to compute losses, evaluate cumulants, and invert rate functions using numerical methods such as ternary search.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Functions Overview](#functions-overview)
  - [Example Usage](#example-usage)
- [API Reference](#api-reference)
  - [get_loss](#get_loss)
  - [eval_cumulant](#eval_cumulant)
  - [eval_cumulant_from_losses](#eval_cumulant_from_losses)
- [License](#license)

## Installation

To use this code, you need Python 3.11.9+ and PyTorch 2.3.1+ installed. You can install PyTorch and other dependencies with the following commands:

```bash
pip install -r requirements.txt
```

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/rate-function-cumulant-computation.git
cd rate-function-cumulant-computation
```

## Usage

### Functions Overview

The repository includes the following key functions:

1. **`get_loss`**: Computes the loss of a model on a given dataset.
2. **`eval_cumulant`**: Evaluates cumulants of the loss distribution at specified points.
3. **`eval_cumulant_from_losses`**: Computes cumulants using precomputed losses.
4. **`rate_function`** and **`inverse_rate_function`** (not shown here): These functions compute the rate function and its inverse using numerical search methods.

### Example Usage

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from my_module import get_loss, eval_cumulant

# Assuming you have a model and a DataLoader
model = ...  # your model here
data_loader = DataLoader(...)  # your dataset

# Define loss function
loss_fn = nn.CrossEntropyLoss(reduction="none")

# Example 1: Compute the loss of the model on the data
losses = get_loss(model, data_loader, loss_fn)
print(f"Losses: {losses}")

# Example 2: Evaluate cumulants at specified points
evaluation_points = torch.tensor([0.1, 0.2, 0.5])  # Example evaluation points
cumulants = eval_cumulant(model, evaluation_points, data_loader, loss_fn)
print(f"Cumulants: {cumulants}")
```

## API Reference

### `get_loss`

```python
get_loss(model: nn.Module, loader: DataLoader, loss_fn: callable = nn.CrossEntropyLoss(reduction="none")) -> torch.Tensor
```

**Description**: Computes the loss of a model on the data from a `DataLoader`.

- **Parameters**:
  - `model (nn.Module)`: The model to evaluate.
  - `loader (DataLoader)`: The DataLoader providing data to the model.
  - `loss_fn (callable)`: The loss function to use (default: `nn.CrossEntropyLoss(reduction="none")`).

- **Returns**: A tensor of losses for each batch in the data.

### `eval_cumulant`

```python
eval_cumulant(model: nn.Module, evaluation_points: torch.Tensor, loader: DataLoader, loss_fn: callable = nn.CrossEntropyLoss(reduction="none")) -> torch.Tensor
```

**Description**: Computes the cumulants of the loss distribution at specified points.

- **Parameters**:
  - `model (nn.Module)`: The model to evaluate.
  - `evaluation_points (torch.Tensor)`: Tensor of evaluation points (lambda values).
  - `loader (DataLoader)`: The DataLoader providing data to the model.
  - `loss_fn (callable)`: The loss function to use (default: `nn.CrossEntropyLoss(reduction="none")`).

- **Returns**: A tensor containing cumulants at the given evaluation points.

### `eval_cumulant_from_losses`

```python
eval_cumulant_from_losses(losses: torch.Tensor, evaluation_points: torch.Tensor) -> torch.Tensor
```

**Description**: Computes cumulants from precomputed losses at specified points.

- **Parameters**:
  - `losses (torch.Tensor)`: The precomputed losses.
  - `evaluation_points (torch.Tensor)`: Tensor of evaluation points (lambda values).

- **Returns**: A tensor containing cumulants evaluated at the given points.

### `rate_function`

```python
rate_function(
    model: nn.Module,
    evaluation_points: torch.Tensor,
    loader: torch.utils.data.DataLoader,
    loss_fn: callable = nn.CrossEntropyLoss(reduction="none"),
    return_lambdas: bool = False,
    return_cummulants: bool = False,
    epsilon: float = 0.01,
    max_lambda: float = 100000,
    strategy: str = "TernarySearch",
    verbose: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
```

**Description**:  
Computes the rate function of a model at specific evaluation points, based on the data provided by a DataLoader and a loss function. The rate function is calculated using an auxiliary function that is optimized through a ternary search process.

**Parameters**:
- `model` (`torch.nn.Module`): The model to evaluate.
- `evaluation_points` (`torch.Tensor`): Points at which the rate function will be computed.
- `loader` (`torch.utils.data.DataLoader`): The DataLoader providing data for evaluation.
- `loss_fn` (`callable`, optional): Loss function used for evaluation (default is `nn.CrossEntropyLoss(reduction="none")`).
- `return_lambdas` (`bool`, optional): If `True`, return the lambda values that maximize the auxiliary function (default is `False`).
- `return_cummulants` (`bool`, optional): If `True`, return the computed cummulants (default is `False`).
- `epsilon` (`float`, optional): Precision for ternary search (default is `0.01`).
- `max_lambda` (`float`, optional): Maximum value for lambda in the search range (default is `100000`).
- `strategy` (`str`, optional): Strategy used to compute the rate function (only "TernarySearch" is supported; default is "TernarySearch").
- `verbose` (`bool`, optional): If `True`, display a progress bar (default is `False`).

**Returns**:
- If `return_lambdas` or `return_cummulants` is `True`, returns the rate function values along with the corresponding lambdas and/or cummulants as numpy arrays.
- Otherwise, returns the rate function as a tensor.

---

### `inverse_rate_function`

```python
inverse_rate_function(
    model: nn.Module,
    evaluation_points: torch.Tensor,
    loader: torch.utils.data.DataLoader,
    loss_fn: callable = nn.CrossEntropyLoss(reduction="none"),
    return_lambdas: bool = False,
    return_cummulants: bool = False,
    epsilon: float = 0.01,
    max_lambda: float = 100000,
    strategy: str = "TernarySearch",
    verbose: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
```

**Description**:  
Computes the inverse rate function of a model at specified evaluation points using data from a DataLoader. The function performs a minimization process using ternary search to compute the inverse of the rate function.

**Parameters**:
- `model` (`torch.nn.Module`): The model to evaluate.
- `evaluation_points` (`torch.Tensor`): Points at which the inverse rate function will be computed.
- `loader` (`torch.utils.data.DataLoader`): The DataLoader providing data for evaluation.
- `loss_fn` (`callable`, optional): Loss function used for evaluation (default is `nn.CrossEntropyLoss(reduction="none")`).
- `return_lambdas` (`bool`, optional): If `True`, return the lambda values that minimize the auxiliary function (default is `False`).
- `return_cummulants` (`bool`, optional): If `True`, return the computed cummulants (default is `False`).
- `epsilon` (`float`, optional): Precision for ternary search (default is `0.01`).
- `max_lambda` (`float`, optional): Maximum value for lambda in the search range (default is `100000`).
- `strategy` (`str`, optional): Strategy used to compute the inverse rate function (only "TernarySearch" is supported; default is "TernarySearch").
- `verbose` (`bool`, optional): If `True`, display a progress bar (default is `False`).

**Returns**:
- If `return_lambdas` or `return_cummulants` is `True`, returns the inverse rate function values along with the corresponding lambdas and/or cummulants as numpy arrays.
- Otherwise, returns the inverse rate function as a tensor.




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

