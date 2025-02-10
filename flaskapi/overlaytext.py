import cv2
import os
import numpy as np
import pytesseract
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

# Define background color (from main script)
background_color = "#F5F5DC"  # Beige

# Convert HEX background color to BGR (since OpenCV uses BGR format)
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

# Resize logo
h_img, w_img, _ = diagram.shape
max_logo_width = w_img // 6
max_logo_height = h_img // 20
logo = cv2.resize(logo, (max_logo_width, max_logo_height))

# Function to add padding
def add_padding(image, padding_size, position):
    h, w, c = image.shape

    pad_top = padding_size if position in ["top-left", "top-right"] else 0
    pad_bottom = padding_size if position in ["bottom-left", "bottom-right"] else 0
    pad_left = padding_size if position in ["top-left", "bottom-left"] else 0
    pad_right = padding_size if position in ["top-right", "bottom-right"] else 0

    new_h, new_w = h + pad_top + pad_bottom, w + pad_left + pad_right
    padded_image = np.full((new_h, new_w, c), bg_color_bgr, dtype=np.uint8)

    # Place the original image inside the padded image
    padded_image[pad_top:pad_top + h, pad_left:pad_left + w] = image

    return padded_image, pad_left, pad_top

# Ask user for logo placement
valid_positions = ["top-left", "top-right", "bottom-left", "bottom-right", "none"]
logo_position = input(f"Enter logo position ({', '.join(valid_positions)}): ").strip().lower()

if logo_position == "none":
    print("No logo will be added.")
    exit(0)

# Add padding
padding_size = max_logo_height + 10
diagram, x_offset, y_offset = add_padding(diagram, padding_size, logo_position)

# Define logo position (adjusting for padding)
possible_positions = {
    "top-left": (10, 10),
    "top-right": (diagram.shape[1] - max_logo_width - 10, 10),
    "bottom-left": (10, diagram.shape[0] - max_logo_height - 10),
    "bottom-right": (diagram.shape[1] - max_logo_width - 10, diagram.shape[0] - max_logo_height - 10)
}

x_logo, y_logo = possible_positions.get(logo_position, (10, 10))

# Overlay the logo with proper transparency handling
if logo.shape[2] == 4:  # If logo has an alpha channel
    alpha = logo[:, :, 3] / 255.0
    for c in range(3):  # Apply alpha blending to RGB channels
        diagram[y_logo:y_logo + logo.shape[0], x_logo:x_logo + logo.shape[1], c] = (
            alpha * logo[:, :, c] + (1 - alpha) * diagram[y_logo:y_logo + logo.shape[0], x_logo:x_logo + logo.shape[1], c]
        )
else:
    diagram[y_logo:y_logo + logo.shape[0], x_logo:x_logo + logo.shape[1]] = logo

# Save output
output_counter = max(
    [int(f.stem.split("_")[-1]) for f in PNG_DIR.glob("diagram_with_logo_*.png") if f.stem.split("_")[-1].isdigit()],
    default=0
) + 1
output_path = PNG_DIR / f"diagram_with_logo_{output_counter}.png"
cv2.imwrite(str(output_path), diagram)
print(f"Diagram with logo saved at: {output_path}")
