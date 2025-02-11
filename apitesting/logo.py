import cv2
import numpy as np
import os
from pathlib import Path
import sys
# Define Directories
PNG_DIR = Path(r"C:\Diagramgenerator\prompts_png")
LOGO_PATH = Path(r"C:\Diagramgenerator\logo.jpg")

# Ensure directories exist
if not PNG_DIR.exists():
    print("Error: Diagram directory does not exist.")
    exit(1)

if not LOGO_PATH.exists():
    print("Error: Logo file not found.")
    exit(1)

# Define background color
background_color =  sys.argv[1] # Beige

# Convert HEX to BGR (since OpenCV uses BGR format)
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)  # Convert RGB to BGR

bg_color_bgr = hex_to_bgr(background_color)

# Get the latest diagram image
diagram_files = sorted(PNG_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)
if not diagram_files:
    print("Error: No diagram images found.")
    exit(1)

latest_diagram = diagram_files[0]
print(f"Using latest diagram: {latest_diagram}")

# Load images
diagram = cv2.imread(str(latest_diagram))
logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED)

if diagram is None or logo is None:
    print("Error loading images.")
    exit(1)

# Get image dimensions
h_img, w_img, _ = diagram.shape

# Define uniform padding (8% of image width for better spacing)
padding_size = int(max(h_img, w_img) * 0.05)

# Create a new blank canvas with padding
padded_image = cv2.copyMakeBorder(
    diagram, padding_size, padding_size, padding_size, padding_size,
    cv2.BORDER_CONSTANT, value=bg_color_bgr
)

# Set a fixed logo size (static dimensions)
fixed_logo_width = 140  # Fixed width in pixels
fixed_logo_height = 70  # Fixed height in pixels
logo = cv2.resize(logo, (fixed_logo_width, fixed_logo_height))

# Define fixed logo positions **AFTER** adding padding
possible_positions = {
    "top-left": (10, 10),
    "top-right": (padded_image.shape[1] - fixed_logo_width - 10, 10),
    "bottom-left": (10, padded_image.shape[0] - fixed_logo_height - 10),
    "bottom-right": (padded_image.shape[1] - fixed_logo_width - 10, padded_image.shape[0] - fixed_logo_height - 10)
}

# User input for logo position
valid_positions = ["top-left", "top-right", "bottom-left", "bottom-right", "none"]
logo_position = input(f"Enter logo position ({', '.join(valid_positions)}): ").strip().lower()

if logo_position == "none":
    print("No logo will be added.")
    exit(0)

x_logo, y_logo = possible_positions.get(logo_position, (10, 10))

# Overlay the logo with transparency handling
if logo.shape[2] == 4:  # If logo has an alpha channel
    alpha = logo[:, :, 3] / 255.0
    for c in range(3):  # Apply alpha blending to RGB channels
        padded_image[y_logo:y_logo + fixed_logo_height, x_logo:x_logo + fixed_logo_width, c] = (
            alpha * logo[:, :, c] + (1 - alpha) * padded_image[y_logo:y_logo + fixed_logo_height, x_logo:x_logo + fixed_logo_width, c]
        )
else:
    padded_image[y_logo:y_logo + fixed_logo_height, x_logo:x_logo + fixed_logo_width] = logo

# Save output
output_counter = max(
    [int(f.stem.split("_")[-1]) for f in PNG_DIR.glob("diagram_with_logo_*.png") if f.stem.split("_")[-1].isdigit()],
    default=0
) + 1
output_path = PNG_DIR / f"diagram_with_logo_{output_counter}.png"
cv2.imwrite(str(output_path), padded_image)
print(f"Diagram with logo saved at: {output_path}")
