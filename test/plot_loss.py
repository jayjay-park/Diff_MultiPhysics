import matplotlib.pyplot as plt
import pandas as pd

# List of file paths for the five different loss CSVs
num_vec = 1
file_paths = [
    f"../test_result/Losses/GCS_mse_loss_MSE_64_100_vec_0.csv",
    f"../test_result/Losses/GCS_mse_loss_JAC_64_100_vec_{num_vec}.csv",
    f"../test_result/Losses/GCS_jac_loss_JAC_64_100_vec_{num_vec}.csv",
    f"../test_result/Losses/GCS_test_loss_MSE_64_100_vec_0.csv",
    f"../test_result/Losses/GCS_test_loss_JAC_64_100_vec_{num_vec}.csv"
]

# List of custom labels for each loss type
loss_labels = [
    r'$FNO_{MSE}$ Train (in MSE)', 
    r'$FNO_{PBI}$ Train (MSE Term Only)', 
    r'$FNO_{PBI}$ Train (Gradient Matching Term Only)', 
    r'$FNO_{PBI}$ Test (in MSE)', 
    r'$FNO_{MSE}$ Test (in MSE)'
]

# Number of epochs to skip
epochs_to_skip = 30

def read_and_filter_loss_data(file_path):
    """Read a CSV file and filter out the first 30 epochs."""
    data = pd.read_csv(file_path)  # Read the CSV file
    # Check for 'Epoch' column and filter accordingly
    if 'Epoch' in data.columns:
        return data[data['Epoch'] >= epochs_to_skip]
    return data.iloc[epochs_to_skip:1001]

# Read and filter loss data from each file
loss_data = [read_and_filter_loss_data(file) for file in file_paths]

# Create a plot for all losses
plt.figure(figsize=(10, 6))

# Loop through the data and plot each loss with its corresponding label
for i, data in enumerate(loss_data):
    plt.plot(data['Epoch'], data['Loss'], label=loss_labels[i])

# Add labels and legend for the main plot
MSE= "MSE"
PBI="PBI"
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(rf'Loss Curves for FNO_MSE and FNO_PBI')
plt.legend()
plt.grid(True)

# Show the main plot
plt.savefig(f'all_loss_{num_vec}.png')

# Create an additional plot for the third loss (index 2 in the list)
third_loss_data = loss_data[2]  # Selecting the third loss data
plt.figure(figsize=(10, 6))
plt.plot(third_loss_data['Epoch'], third_loss_data['Loss'], label=loss_labels[2], color='orange')  # Customize color as needed

# Add labels and legend for the additional plot
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss Curve for {loss_labels[2]} (After Removing First {epochs_to_skip} Epochs)')
plt.legend()
plt.grid(True)

# Show the additional plot
plt.savefig(f'PBI_term_{num_vec}.png')
