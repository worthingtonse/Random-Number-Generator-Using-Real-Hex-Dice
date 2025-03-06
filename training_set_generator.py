# This generates a traing 
#Below is a Python script that generates synthetic images of a six-sided die with the specifications you described: black die with white holes, black background, varying x, y coordinates, rotations, and zooms. Each image will be resized to 28x28 pixels, and the pixel values will be saved into a CSV file suitable for loading into a pandas DataFrame. We'll generate 10,000 images per die side (60,000 total).
#This script uses Pillow (PIL) for image generation and pandas for CSV handling. I'll explain the steps and provide the code.
#Steps:
# 1. Generate Synthetic Dice Images: Create images of a die with 1 to 6 white holes, varying position, rotation, and scale.

# 2. Resize to 28x28: Convert each image to 28x28 pixels in grayscale.

# 3. Save to CSV: Flatten each image into a 784-element array (28x28) and store it in a CSV file with a label column (1–6).

#Requirements:
# Install these libraries if you don’t have them:
# pip install Pillow pandas numpy
"""
Explanation:
Die Face Generation: The draw_die_face function creates a die with 1–6 white dots on a black background. Dot positions are predefined for each face.

Transformations: The generate_die_image function applies random scaling (zoom), rotation, and translation (x, y offsets) to simulate variability.

Output: Each image is resized to 28x28 pixels, flattened to a 784-element array, and stored in a DataFrame with a label column. The DataFrame is saved as a CSV file (dice_dataset.csv).

CSV Format: The CSV has 785 columns: 784 pixel values (0–255 grayscale) and 1 label (1–6).

Notes:
The script generates 60,000 images total (10,000 per side), which might take some time depending on your hardware (a few minutes to an hour).

The images are grayscale (0 = black, 255 = white), matching your black die with white holes.

The CSV file will be large (~450 MB uncompressed). You can compress it with df.to_csv(OUTPUT_CSV, compression='gzip') if needed.

Next Steps:
Run the script to generate the dataset.

Load it in Python for training:
python

import pandas as pd
df = pd.read_csv("dice_dataset.csv")
X = df.iloc[:, :-1].values  # Pixel data
y = df["label"].values      # Labels

Use this data to train your torchvision model (e.g., a CNN).


"""



import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pandas as pd
import random
import os

# Constants
NUM_IMAGES_PER_SIDE = 10000  # 10,000 images per die face
DIE_SIDES = [1, 2, 3, 4, 5, 6]
OUTPUT_CSV = "dice_dataset.csv"
IMG_SIZE = 28  # Final image size (28x28 pixels)

# Function to draw a die face with specified number of dots
def draw_die_face(num_dots, base_size=50):
    img = Image.new("L", (base_size, base_size), color=0)  # Black background (grayscale)
    draw = ImageDraw.Draw(img)
    
    # Define dot positions for each face (relative to base_size)
    dot_size = base_size // 10
    dot_positions = {
        1: [(base_size//2, base_size//2)],
        2: [(base_size//4, base_size//4), (3*base_size//4, 3*base_size//4)],
        3: [(base_size//4, base_size//4), (base_size//2, base_size//2), (3*base_size//4, 3*base_size//4)],
        4: [(base_size//4, base_size//4), (base_size//4, 3*base_size//4), 
            (3*base_size//4, base_size//4), (3*base_size//4, 3*base_size//4)],
        5: [(base_size//4, base_size//4), (base_size//4, 3*base_size//4), 
            (base_size//2, base_size//2), 
            (3*base_size//4, base_size//4), (3*base_size//4, 3*base_size//4)],
        6: [(base_size//4, base_size//4), (base_size//4, base_size//2), (base_size//4, 3*base_size//4),
            (3*base_size//4, base_size//4), (3*base_size//4, base_size//2), (3*base_size//4, 3*base_size//4)]
    }
    
    # Draw white dots on black die
    for pos in dot_positions[num_dots]:
        draw.ellipse([pos[0]-dot_size, pos[1]-dot_size, pos[0]+dot_size, pos[1]+dot_size], fill=255)
    
    return img

# Function to generate a single transformed die image
def generate_die_image(num_dots):
    # Start with a larger base image to allow for transformations
    base_size = 50
    canvas_size = 100  # Larger canvas to accommodate rotation and zooming
    canvas = Image.new("L", (canvas_size, canvas_size), color=0)  # Black background
    
    # Generate the die face
    die = draw_die_face(num_dots, base_size)
    
    # Random transformations
    scale = random.uniform(0.5, 1.5)  # Random zoom (50% to 150% of base size)
    rotation = random.uniform(0, 360)  # Random rotation (0 to 360 degrees)
    x_offset = random.randint(0, canvas_size - int(base_size * scale))  # Random x position
    y_offset = random.randint(0, canvas_size - int(base_size * scale))  # Random y position
    
    # Apply transformations
    die = die.resize((int(base_size * scale), int(base_size * scale)), Image.Resampling.LANCZOS)
    die = die.rotate(rotation, expand=True, fillcolor=0)
    
    # Paste the transformed die onto the canvas
    canvas.paste(die, (x_offset, y_offset))
    
    # Resize to 28x28 pixels
    final_img = canvas.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    
    return final_img

# Generate dataset
data = []
for side in DIE_SIDES:
    print(f"Generating images for die face: {side}")
    for _ in range(NUM_IMAGES_PER_SIDE):
        img = generate_die_image(side)
        img_array = np.array(img).flatten()  # Flatten 28x28 to 784 elements
        data.append(np.append(img_array, side))  # Append label (1-6)

# Convert to DataFrame
columns = [f"pixel_{i}" for i in range(IMG_SIZE * IMG_SIZE)] + ["label"]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"Dataset saved to {OUTPUT_CSV}")

# Optional: Verify the dataset
print("Dataset shape:", df.shape)
print("Sample row:")
print(df.iloc[0])
