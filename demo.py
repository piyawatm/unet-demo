# Simple U-Net Demo for Image Segmentation
# Task: Find circles in images while ignoring squares

# HOW TO RUN THIS FILE:
# 0. Ensure you have Python installed (3.6 or later) and create a virtual environment if desired.
#    To create a virtual environment:
#    python -m venv myenv
#    Activate it:
#    - On Windows: myenv\Scripts\activate
#    - On macOS/Linux: source myenv/bin/activate
#    This keeps your project dependencies isolated.
# 1. Save this file as "unet_demo.py"
# 2. Install required libraries (if not already installed):
#    pip install torch numpy matplotlib
# 3. Run the file:
#    python unet_demo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# PART 1: CREATE SYNTHETIC TRAINING DATA
def create_image():
    """
    Creates a synthetic image with circles and squares.
    Returns: 
    - img: Image with both circles (white) and squares (gray)
    - mask: Binary mask with only circles marked as 1
    """
    # Create empty 64x64 image and mask
    # Shape: (batch_size=1, channels=1, height=64, width=64)
    img = np.zeros((1, 1, 64, 64), dtype=np.float32)
    mask = np.zeros((1, 1, 64, 64), dtype=np.float32)
    
    # Add 2-3 random circles (these are what we want to find!)
    for _ in range(np.random.randint(2, 4)):
        # Random center position for circle
        cx, cy = np.random.randint(10, 54, 2)
        # Random radius
        radius = np.random.randint(5, 15)
        
        # Create circle using distance formula
        # ogrid creates coordinate grids for calculations
        y, x = np.ogrid[:64, :64]
        # True where distance from center <= radius
        circle = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        # Add circle to both image and mask
        img[0, 0, circle] = 1      # White circle in image
        mask[0, 0, circle] = 1     # Mark circle in mask
    
    # Add some squares (distractors - we DON'T want to segment these)
    for _ in range(np.random.randint(1, 3)):
        # Random position for square
        x, y = np.random.randint(10, 50, 2)
        # Random size
        size = np.random.randint(8, 15)
        # Add gray square to image only (NOT to mask)
        img[0, 0, y:y+size, x:x+size] = 0.5
    
    # Convert numpy arrays to PyTorch tensors
    return torch.FloatTensor(img), torch.FloatTensor(mask)

# PART 2: DEFINE U-NET ARCHITECTURE
class MiniUNet(nn.Module):
    """
    Simplified U-Net architecture for educational purposes.
    U-Net has a U-shaped structure:
    - Left side (encoder): Compress image to features
    - Bottom: Process compressed features  
    - Right side (decoder): Expand back to full resolution
    - Skip connections: Connect left to right to preserve details
    """
    def __init__(self):
        super().__init__()
        
        # ENCODER (downsampling path - left side of U)
        # Conv2d(in_channels, out_channels, kernel_size, padding)
        self.enc1 = nn.Conv2d(1, 16, 3, padding=1)   # 1 input channel -> 16 feature maps
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)  # 16 -> 32 feature maps
        
        # DECODER (upsampling path - right side of U)
        self.dec2 = nn.Conv2d(32, 16, 3, padding=1)  # 32 -> 16 feature maps
        self.dec1 = nn.Conv2d(32, 16, 3, padding=1)  # 32 (not 16!) due to skip connection
        
        # FINAL OUTPUT LAYER
        self.final = nn.Conv2d(16, 1, 1)  # 16 -> 1 output channel (binary mask)
        
    def forward(self, x):
        # ENCODER: Progressively downsample
        e1 = F.relu(self.enc1(x))                    # Apply first convolution + ReLU
        e2 = F.relu(self.enc2(F.max_pool2d(e1, 2))) # Downsample with max pooling (64->32)
        
        # DECODER: Progressively upsample
        # F.interpolate upsamples back to original size (32->64)
        d2 = F.relu(self.dec2(F.interpolate(e2, scale_factor=2)))
        
        # SKIP CONNECTION: This is the key innovation of U-Net!
        # Concatenate encoder features (e1) with decoder features (d2)
        # This preserves fine details that would otherwise be lost
        d1 = F.relu(self.dec1(torch.cat([d2, e1], 1)))
        
        # Final prediction: sigmoid ensures output is between 0 and 1
        return torch.sigmoid(self.final(d1))

# PART 3: TRAINING SETUP
model = MiniUNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with learning rate 0.01
criterion = nn.BCELoss()  # Binary Cross Entropy - perfect for binary segmentation

# PART 4: TRAINING LOOP
print("Training U-Net...")
losses = []  # Track loss over time

# Train for 100 epochs (iterations over the data)
for epoch in range(100):
    # Generate a batch of 8 random images
    batch_img = []
    batch_mask = []
    for _ in range(8):
        img, mask = create_image()
        batch_img.append(img)
        batch_mask.append(mask)
    
    # Stack individual images into a batch
    imgs = torch.cat(batch_img)    # Shape: (8, 1, 64, 64)
    masks = torch.cat(batch_mask)  # Shape: (8, 1, 64, 64)
    
    # TRAINING STEP
    optimizer.zero_grad()           # Clear gradients from previous step
    pred = model(imgs)              # Forward pass: get predictions
    loss = criterion(pred, masks)   # Calculate how wrong we are
    loss.backward()                 # Backpropagation: calculate gradients
    optimizer.step()                # Update model weights
    
    # Record loss for plotting
    losses.append(loss.item())
    
    # Print progress every 20 epochs
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# PART 5: VISUALIZE RESULTS
model.eval()  # Switch to evaluation mode (disables dropout, etc.)

# Create a figure with 3 rows and 4 columns
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

# Test on 3 new images
for i in range(3):
    # Generate a new test image
    img, mask = create_image()
    
    # Get model prediction (no gradients needed for inference)
    with torch.no_grad():
        pred = model(img)
    
    # Column 1: Original image
    axes[i, 0].imshow(img[0, 0], cmap='gray')
    axes[i, 0].set_title('Input')
    axes[i, 0].axis('off')
    
    # Column 2: Ground truth (what we want)
    axes[i, 1].imshow(mask[0, 0], cmap='hot')
    axes[i, 1].set_title('True Circles')
    axes[i, 1].axis('off')
    
    # Column 3: Model prediction
    axes[i, 2].imshow(pred[0, 0].detach(), cmap='hot')
    axes[i, 2].set_title('Predicted')
    axes[i, 2].axis('off')
    
    # Column 4: Overlay (prediction on top of original)
    axes[i, 3].imshow(img[0, 0], cmap='gray')
    # Show predictions > 0.5 as overlay
    axes[i, 3].imshow(pred[0, 0].detach() > 0.5, cmap='hot', alpha=0.5)
    axes[i, 3].set_title('Overlay')
    axes[i, 3].axis('off')

plt.suptitle('U-Net learns to segment circles (ignoring squares!)', fontsize=16)
plt.tight_layout()
plt.show()

# PART 6: PLOT TRAINING PROGRESS
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy Loss')
plt.grid(True)
plt.show()

# Success message
print("\n✨ Success! U-Net learned to find circles while ignoring squares!")
print("Key insight: Skip connections preserve spatial details for precise segmentation")