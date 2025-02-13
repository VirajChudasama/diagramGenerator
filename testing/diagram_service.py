import os
import cv2
import numpy as np
from pathlib import Path
from openai import OpenAI
import requests
import json
from datetime import datetime


# Define paths
MERMAID_FILE_PATH = Path(r"C:\\testing\\prompts")
IMAGE_FILE_PATH = Path(r"C:\\testing\\images")

LOGO_PATH = Path(r"C:\\testing\\logo.jpg")

IMAGE_FILE_PATH.mkdir(parents=True, exist_ok=True)
MERMAID_FILE_PATH.mkdir(parents=True, exist_ok=True)



# Initialize the OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_diagram(prompt):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini"
    )
    return chat_completion.choices[0].message.content

# Convert HEX to BGR (since OpenCV uses BGR format)
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)  # Convert RGB to BGR

def addLogo(filePath, background_color ):
    bg_color_bgr = hex_to_bgr(background_color)

    # Load images
    diagram = cv2.imread(str(filePath))
    logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED)

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
    fixed_logo_width = 120  # Fixed width in pixels
    fixed_logo_height = 60  # Fixed height in pixels
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

    logo_position = valid_positions[0]

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
        [int(f.stem.split("_")[-1]) for f in IMAGE_FILE_PATH.glob("diagram_with_logo_*.png") if f.stem.split("_")[-1].isdigit()],
        default=0
    ) + 1
    final_diagram_path = IMAGE_FILE_PATH / f"diagram_with_logo_{output_counter}.png"
    cv2.imwrite(str(final_diagram_path), padded_image)
    print(f"Diagram with logo saved at: {final_diagram_path}")
    return final_diagram_path

    

def save_diagram_code(diagram_code, file_path):
    print(f"Saving to: {file_path}")  # Debug print
    print(f"Diagram Code:\n{diagram_code}")  # Debug print

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(diagram_code)

    if not file_path.exists():  # Check if file was created
        print(f"Error: {file_path} was not created!")
        return None

    print(f"Diagram saved at: {file_path}")
    return 

def generate_image_from_kroki(diagram_type, diagram_code, output_path):
    """Converts Mermaid or PlantUML code into an image using the Kroki API."""
    kroki_url = f"https://kroki.io/{diagram_type}/png"
    payload = {"diagram_source": diagram_code}
    headers = {"Content-Type": "application/json"}

    response = requests.post(kroki_url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        with open(output_path, "wb") as img_file:
            img_file.write(response.content)
        print(f"Diagram image saved at: {output_path}")
    else:
        print(f"Failed to generate {diagram_type} image. HTTP Status: {response.status_code}, Response: {response.text}")


def generate_mermaid_diagram(background_color, arrow_color, box_color, custom_prompt=None):
    """Generates a Mermaid diagram with dynamic colors and a custom prompt."""
    timestamp = get_timestamp()
    mmd_path = MERMAID_FILE_PATH / f"mermaid_{timestamp}.mmd"
    img_path = IMAGE_FILE_PATH / f"mermaid_{timestamp}.png"

    # Use custom prompt if provided, otherwise use default
    mermaid_prompt = f"{custom_prompt} Use the following styles: " \
                 f"Background Color: `{background_color}`, Arrows: `{arrow_color}`, Boxes: `{box_color}`. " \
                 "Ensure the diagram is **fully correct**, avoiding invalid syntax. " \
                 "Use `\\n` for multi-line text (no `<br>`). " \
                 f"Format it as a **flowchart** and apply styling using `%%{{init: {{'theme': 'base', 'themeVariables': {{'background': '{background_color}', 'primaryColor': '{box_color}', 'edgeLabelBackground': '{arrow_color}'}}}}" \
                 f"{{'background': '{background_color}', 'primaryColor': '{box_color}', 'tertiaryColor': '{arrow_color}'}}}}}}%%`."

    response = generate_diagram(mermaid_prompt)

    if "```mermaid" in response:
        mermaid_code = response.split("```mermaid")[1].split("```")[0].strip()
    else:
        print("Error: No Mermaid diagram found in the response.")
        return None

    save_diagram_code(mermaid_code, mmd_path)
    generate_image_from_kroki("mermaid", mermaid_code, img_path)

    updated_img_path = apply_background_color(img_path, background_color, "mermaid")

    return updated_img_path  # Return updated image path with background color


def generate_plantuml_diagram(background_color, arrow_color, box_color, add_logo, custom_prompt=None):
    """Generates a PlantUML diagram with dynamic colors and a custom prompt."""
    timestamp = get_timestamp()
    puml_path = MERMAID_FILE_PATH / f"plantuml_{timestamp}.puml"
    img_path = IMAGE_FILE_PATH / f"plantuml_{timestamp}.png"

    # Use custom prompt if provided, otherwise use default
    plantuml_prompt = f"{custom_prompt} Background Color: {background_color}, Arrows: {arrow_color}, Boxes: {box_color}."


    response = generate_diagram(plantuml_prompt)

    if "```plantuml" in response:
        plantuml_code = response.split("```plantuml")[1].split("```")[0].strip()
    else:
        print("Error: No PlantUML diagram found in the response.")
        return None

    save_diagram_code(plantuml_code, puml_path)
    generate_image_from_kroki("plantuml", plantuml_code, img_path)
    return img_path  # Return image path



def apply_background_color(image_path, background_color, diagram_type):
    """
    Applies a solid background color to the entire image if the diagram type is 'mermaid'.

    Args:
        image_path (Path): Path to the generated diagram image.
        background_color (str): HEX color code for the background.
        diagram_type (str): Type of diagram ('mermaid' or 'plantuml').

    Returns:
        Path: Updated image path with applied background.
    """
    if diagram_type.lower() != "mermaid":
        return image_path  # No need to apply background for PlantUML

    bg_color_bgr = hex_to_bgr(background_color)

    # Load the diagram
    diagram = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    # If the image has an alpha channel (transparency), replace it with the background color
    if diagram.shape[2] == 4:  # RGBA image
        alpha_channel = diagram[:, :, 3] / 255.0
        for c in range(3):  # Apply alpha blending to RGB channels
            diagram[:, :, c] = (alpha_channel * diagram[:, :, c] + (1 - alpha_channel) * bg_color_bgr[c])
        diagram = diagram[:, :, :3]  # Remove alpha channel after processing

    else:
        # Add a solid background color behind the diagram
        h, w, _ = diagram.shape
        background = np.full((h, w, 3), bg_color_bgr, dtype=np.uint8)
        diagram = cv2.addWeighted(diagram, 1, background, 0.5, 0)  # Blend with the background

    # Save the modified image
    updated_image_path = image_path.parent / f"colored_{image_path.name}"
    cv2.imwrite(str(updated_image_path), diagram)

    print(f"Updated image saved at: {updated_image_path}")
    return updated_image_path
