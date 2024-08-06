import torch
import numpy as np
import matplotlib.pyplot as plt

# Verify PyTorch environment and import libraries
print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sierpinski_carpet(size, depth, device):
    # Initialize the grid with ones
    grid = torch.ones((size, size), device=device)
    
    def cutout(x, y, size, depth):
        if depth == 0:
            return
        third_size = size // 3
        # Cut out the central square
        grid[y + third_size : y + 2 * third_size, x + third_size : x + 2 * third_size] = 0
        # Recursively cut out the smaller squares in the remaining eight regions
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                cutout(x + i * third_size, y + j * third_size, third_size, depth - 1)

    # Start the recursive cutout process
    cutout(0, 0, size, depth)
    
    return grid

# Parameters
size = 729  # Should be a power of 3 for better visualization
depth = 6

# Generate the Sierpiński carpet
sierpinski_image = sierpinski_carpet(size, depth, device)

# Plot the result
plt.figure(figsize=(10, 10))
plt.imshow(sierpinski_image.cpu().numpy(), cmap='binary')
plt.axis('off')
# plt.title("Sierpiński Carpet")
plt.show()
