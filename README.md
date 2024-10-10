# Rate Function and Cumulant Computation

This repository provides PyTorch-based utilities for evaluating rate functions and cumulants for machine learning models. The code is designed to compute losses, evaluate cumulants, and invert rate functions using numerical methods such as ternary search.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Functional API](#functional-api)
  - [Class API](#class-api)
  - [Example Usage](#example-usage)
- [API Reference](#api-reference)
  - [Functional API](#functional-api-1)
  - [Class API](#class-api-1)
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

### Functional API

The Functional API offers individual functions to compute rate functions, cumulants, and inverse rate functions directly.

1. **`get_loss`**: Computes the loss of a model on a given dataset.
2. **`eval_cumulant`**: Evaluates cumulants of the loss distribution at specified points.
3. **`eval_cumulant_from_losses`**: Computes cumulants using precomputed losses.
4. **`rate_function`**: Computes the rate function for a model at specific evaluation points.
5. **`rate_function_from_losses`**: Computes the rate function using precomputed losses.
6. **`inverse_rate_function`**: Computes the inverse of the rate function for a model at specific evaluation points.
7. **`inverse_rate_function_from_losses`**: Computes the inverse rate function using precomputed losses.

### Class API

For users who prefer an object-oriented approach, the **`RateCumulant`** class encapsulates models, data loaders, loss functions, and related parameters. It provides methods to compute rate functions, inverse rate functions, and cumulants.

### Example Usage

Below are example use cases for both the Functional API and Class API.

#### Functional API Example

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from my_module import get_loss, eval_cumulant, rate_function

# Initialize model and DataLoader
model = ...  # your model here
data_loader = DataLoader(...)  # your dataset

# Compute the loss using the functional API
losses = get_loss(model, data_loader)

# Compute cumulants at specified points
evaluation_points = torch.tensor([0.1, 0.5, 1.0])
cumulants = eval_cumulant(model, evaluation_points, data_loader)

# Compute rate function at specified points
rate_values = rate_function(model, evaluation_points, data_loader)
```

#### Class API Example

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from my_module import RateCumulant

# Initialize model and DataLoader
model = ...  # your model here
data_loader = DataLoader(...)  # your dataset

# Initialize RateCumulant class
rate_cumulant = RateCumulant(model, data_loader)

# Compute rate function
evaluation_points = torch.tensor([0.1, 0.5, 1.0])
rate_values = rate_cumulant.compute_rate_function(evaluation_points)

# Compute inverse rate function
inverse_values = rate_cumulant.compute_inverse_rate_function(evaluation_points)

# Compute cumulants
cumulants = rate_cumulant.compute_cumulants(evaluation_points)
```

## API Reference

### Functional API

#### `get_loss`

```python
get_loss(
    model: nn.Module, 
    loader: DataLoader, 
    loss_fn: callable = nn.CrossEntropyLoss(reduction="none")
) -> torch.Tensor
```

**Description**: Computes the loss of a model on data provided by a `DataLoader`.

- **Parameters**:
  - `model (nn.Module)`: The model to evaluate.
  - `loader (DataLoader)`: The data loader providing batches of data.
  - `loss_fn (callable)`: The loss function to use (default: `nn.CrossEntropyLoss(reduction="none")`).

- **Returns**: A tensor of loss values.

#### `eval_cumulant`

```python
eval_cumulant(
    model: nn.Module, 
    evaluation_points: torch.Tensor, 
    loader: DataLoader, 
    loss_fn: callable = nn.CrossEntropyLoss(reduction="none")
) -> torch.Tensor
```

**Description**: Computes the cumulants of the loss distribution at specified evaluation points.

- **Parameters**:
  - `model (nn.Module)`: The model to evaluate.
  - `evaluation_points (torch.Tensor)`: Tensor of lambda values.
  - `loader (DataLoader)`: The data loader providing batches of data.
  - `loss_fn (callable)`: The loss function to use (default: `nn.CrossEntropyLoss(reduction="none")`).

- **Returns**: A tensor containing the cumulants.

#### `eval_cumulant_from_losses`

```python
eval_cumulant_from_losses(
    losses: torch.Tensor, 
    evaluation_points: torch.Tensor
) -> torch.Tensor
```

**Description**: Computes cumulants from precomputed loss values at specified points.

- **Parameters**:
  - `losses (torch.Tensor)`: Precomputed loss values.
  - `evaluation_points (torch.Tensor)`: Tensor of lambda values.

- **Returns**: A tensor containing the cumulants.

#### `rate_function`

```python
rate_function(
    model: nn.Module, 
    evaluation_points: torch.Tensor, 
    loader: DataLoader, 
    loss_fn: callable = nn.CrossEntropyLoss(reduction="none"), 
    return_lambdas: bool = False,
    return_cummulants: bool = False, 
    epsilon: float = 0.01, 
    max_lambda: float = 100000, 
    strategy: str = "TernarySearch", 
    verbose: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
```

**Description**: Computes the rate function of the model at specific evaluation points.

- **Parameters**:
  - `model (nn.Module)`: The model to evaluate.
  - `evaluation_points (torch.Tensor)`: Points at which to evaluate the rate function.
  - `loader (DataLoader)`: DataLoader providing the data.
  - `loss_fn (callable)`: The loss function to use.
  - `return_lambdas` (`bool`, optional): Return the lambda values that minimize the auxiliary function.
  - `return_cummulants` (`bool`, optional): Return the computed cumulants.
  - `epsilon` (`float`, optional): Precision for the ternary search.
  - `max_lambda` (`float`, optional): Maximum value of lambda for the search.
  - `strategy` (`str`, optional): Strategy for inversion (`TernarySearch`).
  - `verbose` (`bool`, optional): Show progress bar.

- **Returns**: Tensor containing rate function values, and optionally, lambdas and cumulants.

#### `inverse_rate_function`

```python
inverse_rate_function(
    model: nn.Module, 
    evaluation_points: torch.Tensor, 
    loader: DataLoader, 
    loss_fn: callable = nn.CrossEntropyLoss(reduction="none"), 
    return_lambdas: bool = False,
    return_cummulants: bool = False, 
    epsilon: float = 0.01, 
    max_lambda: float = 100000, 
    strategy: str = "TernarySearch", 
    verbose: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
```

**Description**: Computes the inverse rate function at specified evaluation points.

- **Parameters**: Same as `rate_function`.


### Class API

#### `RateCumulant`

```python
class RateCumulant:
    def __init__(self, model, loader, loss_fn=nn.CrossEntropyLoss(reduction="none"), epsilon=0.01, max_lambda=100000, strategy="TernarySearch", verbose=False):
        ...
```

**Description**: Initializes a `RateCumulant` class with model, data, and parameters for rate function and cumulant computation.

- **Parameters**:
  - `model (nn.Module)`: The model to evaluate.
  - `loader (DataLoader)`: The data loader providing data for evaluation.
  - `loss_fn (callable, optional)`: Loss function used for evaluation (default is `nn.CrossEntropyLoss(reduction="none")`).
  - `epsilon (float, optional)`: Precision for ternary search (default is `0.01`).
  - `max_lambda (float, optional)`: Maximum value for lambda in the search range.
  - `strategy (str, optional)`: Search strategy (`TernarySearch`).
  - `verbose (bool, optional)`: Show progress bar.

#### `compute_rate_function`

```python
def compute_rate_function(
    self, 
    evaluation_points: torch.Tensor, 
    return_lambdas: bool = False, 
    return_cummulants: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ...
```

**Description**: Computes the rate function at specified evaluation points.

- **Parameters**:
  - `evaluation_points (torch.Tensor)`: Points at which to evaluate the rate function.
  - `return_lambdas (bool, optional)`: Return the lambda values that minimize the auxiliary function.
  - `return_cummulants (bool, optional)`: Return the computed cumulants.

- **Returns**: Tensor containing rate function values, and optionally lambdas and cumulants.

#### `compute_inverse_rate_function`

```python
def compute_inverse_rate_function(
    self, 
    evaluation_points: torch.Tensor, 
    return_lambdas: bool = False, 
    return_cummulants: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ...
```

**Description**: Computes the inverse rate function at specified evaluation points.

- **Parameters**: Same as `compute_rate_function`.

#### `compute_cumulants`

```python
def compute_cumulants(
    self, 
    evaluation_points: torch.Tensor
) -> torch.Tensor:
    ...
```

**Description**: Computes the cumulants of the loss distribution at specified evaluation points.

- **Parameters**:
  - `evaluation_points (torch.Tensor)`: Points at which to evaluate the cumulants.

- **Returns**: Tensor containing the computed cumulants.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.