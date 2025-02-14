import cv2
import numpy as np
import os
from pathlib import Path

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
background_color = "#F5F5DC"  # Beige

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
padding_size = int(max(h_img, w_img) * 0.06)

# Create a new blank canvas with padding
padded_image = cv2.copyMakeBorder(
    diagram, padding_size, padding_size, padding_size, padding_size,
    cv2.BORDER_CONSTANT, value=bg_color_bgr
)

# Resize logo dynamically (keeping aspect ratio)
max_logo_width = w_img // 7  # 1/7th of image width
max_logo_height = h_img // 15  # 1/15th of image height
logo = cv2.resize(logo, (max_logo_width, max_logo_height))

# Define correct logo positions **AFTER** adding padding
possible_positions = {
    "top-left": (10, 10),
    "top-right": (padded_image.shape[1] - max_logo_width - 10, 10),
    "bottom-left": (10, padded_image.shape[0] - max_logo_height - 10),
    "bottom-right": (padded_image.shape[1] - max_logo_width - 10, padded_image.shape[0] - max_logo_height - 10)
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
        padded_image[y_logo:y_logo + logo.shape[0], x_logo:x_logo + logo.shape[1], c] = (
            alpha * logo[:, :, c] + (1 - alpha) * padded_image[y_logo:y_logo + logo.shape[0], x_logo:x_logo + logo.shape[1], c]
        )
else:
    padded_image[y_logo:y_logo + logo.shape[0], x_logo:x_logo + logo.shape[1]] = logo

# Save output
output_counter = max(
    [int(f.stem.split("_")[-1]) for f in PNG_DIR.glob("diagram_with_logo_*.png") if f.stem.split("_")[-1].isdigit()],
    default=0
) + 1
output_path = PNG_DIR / f"diagram_with_logo_{output_counter}.png"
cv2.imwrite(str(output_path), padded_image)
print(f"Diagram with logo saved at: {output_path}")
