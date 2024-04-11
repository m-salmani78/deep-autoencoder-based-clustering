## 5. Calculate weights for each pixel
import torch.utils.data
from math import exp
import matplotlib.pyplot as plt
import os

def calc_weights(train_dataset, device):
    # Load the weightslist if already calculated
    if os.path.isfile('mse_weights.pt'):
        mse_weights = torch.load('mse_weights.pt',map_location=torch.device(device))
        # print(len(mse_weights))

    # Select a subset to calculate weights for each pixel
    sub_set_dataset = torch.utils.data.Subset(train_dataset,list(range(0, len(train_dataset), 1000))) # num: 60
    print('subset num: ',len(sub_set_dataset))

    i = 0
    mse_weights = torch.zeros(28 * 28)
    for i in range(28 * 28):
        # calculate weight for each pixel
        sum_1 = sum_2 = 0.0
        num_1 = num_2 = 0.0
        for p in sub_set_dataset:
            x_p, l_p = p
            x_p = x_p.view(-1)
            for q in sub_set_dataset:
                x_q, l_q = q
                x_q = x_q.view(-1)
                if l_p == l_q:
                    num_1 += 1
                    sum_1 += exp(-((x_p[i] - x_q[i]) ** 2))
                else:
                    num_2 += 1
                    sum_2 += 1 - exp(-((x_p[i] - x_q[i]) ** 2))
        mse_weights[i] = (sum_1 / num_1) * (sum_2 / num_2)

    # Move weights tensor to GPU
    mse_weights = mse_weights.to(device)

    # Save results
    torch.save(mse_weights, 'mse_weights.pt')

    return mse_weights

def visualize_weights(mse_weights):
    # Reshape the 'weights' tensor into a 28x28 grid
    weight_grid = mse_weights.view(28, 28).cpu().numpy()

    # Create a figure and axis for visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(weight_grid, cmap='viridis')  # You can choose a colormap of your choice

    # Show the colorbar if needed
    cbar = ax.figure.colorbar(ax.imshow(weight_grid), ax=ax)
    cbar.ax.set_ylabel('Weight Values')

    # Show the plot
    plt.title("Weight Visualization")
    plt.show()