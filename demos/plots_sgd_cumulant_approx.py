import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame()

# Load metrics back from CSV and convert string values to float
loaded_metrics = pd.read_csv('./data/test_metrics.csv')
for column in ['L_test', 'L_val', 'alphaD_test', 'alphaD_val', 'Loss_train', 'batch_loss', 
               'lambdas_test', 'lambdas_val', 'cummulants_test', 'cummulants_val']:
    loaded_metrics[column] = pd.to_numeric(loaded_metrics[column], errors='coerce')
loaded_metrics = loaded_metrics.to_dict('list')

# Calculate MSE between L_test and L_val
l_mse = np.mean((np.array(loaded_metrics['L_test']) - np.array(loaded_metrics['L_val'])) ** 2)
# Calculate MSE between alphaD_test and alphaD_val
alpha_mse = np.mean((np.array(loaded_metrics['alphaD_test']) - np.array(loaded_metrics['alphaD_val'])) ** 2)

print(f"MSE between L_test and L_val: {l_mse:.6f}")
print(f"MSE between alphaD_test and alphaD_val: {alpha_mse:.6f}")

# Create plots using loaded metrics
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot L values and losses using loaded metrics
ax1.plot(loaded_metrics['it'], loaded_metrics['L_test'], label='L Test', linestyle='--')
ax1.plot(loaded_metrics['it'], loaded_metrics['L_val'], label='L Val')
ax1.plot(loaded_metrics['it'], loaded_metrics['Loss_train'], label='Loss Train', linestyle=':')
ax1.plot(loaded_metrics['it'], loaded_metrics['batch_loss'], label='Batch Loss', linestyle='-.')
ax1.set_title('L Values and Losses Over Training')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Value')
ax1.grid(True)
ax1.legend()

# Plot alpha values using loaded metrics
ax2.plot(loaded_metrics['it'], loaded_metrics['alphaD_test'], label='Alpha Test', linestyle='--')
ax2.plot(loaded_metrics['it'], loaded_metrics['alphaD_val'], label='Alpha Val')
ax2.set_title('Alpha Values Over Training')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Alpha Value')
ax2.grid(True)
ax2.legend()

# Plot lambdas values using loaded metrics
ax3.plot(loaded_metrics['it'], loaded_metrics['lambdas_test'], label='Lambdas Test', linestyle='--')
ax3.plot(loaded_metrics['it'], loaded_metrics['lambdas_val'], label='Lambdas Val')
ax3.set_title('Lambdas Over Training')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Lambdas Value')
ax3.grid(True)
ax3.legend()

# Plot cumulants values using loaded metrics
ax4.plot(loaded_metrics['it'], loaded_metrics['cummulants_test'], label='Cumulants Test', linestyle='--')
ax4.plot(loaded_metrics['it'], loaded_metrics['cummulants_val'], label='Cumulants Val')
ax4.set_title('Cumulants Over Training')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Cumulants Value')
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.show()