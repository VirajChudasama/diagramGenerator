import os
import cv2
import numpy as np
from pathlib import Path
from openai import OpenAI
import requests
import json
from datetime import datetime


# Define paths
MERMAID_FILE_PATH = Path(r"C:\\Diagramgenerator\\testing\\prompts")
IMAGE_FILE_PATH = Path(r"C:\\Diagramgenerator\\testing\\images")

LOGO_PATH = Path(r"C:\\Diagramgenerator\\testing\\dark-logo.png")

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

    if logo is None:
        print(f"Error: Could not load logo from {LOGO_PATH}. Check file path and integrity.")
        return None
    
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
    if w_img <= 400 and h_img <= 300:
        fixed_logo_width = 50
        fixed_logo_height = 20
    elif w_img <= 800 and h_img <= 600:
        fixed_logo_width = 75
        fixed_logo_height = 30
    elif w_img <= 1200 and h_img <= 900:
        fixed_logo_width = 100
        fixed_logo_height = 40
    else:
        fixed_logo_width = 150
        fixed_logo_height = 60  
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


def generate_mermaid_diagram(background_color, arrow_color, box_color, custom_prompt=None, retry_count=3):
    """Generates a Mermaid diagram with dynamic colors and a custom prompt, with retry logic for syntax errors."""
    try:
        if retry_count <= 0:
            raise RuntimeError("Max retry attempts reached. Aborting diagram generation.")

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
            raise ValueError("Error: No Mermaid diagram found in the response.")

        save_diagram_code(mermaid_code, mmd_path)

        # Ensure file is created before proceeding
        if not mmd_path.exists():
            raise FileNotFoundError(f"Error: Mermaid file was not created at {mmd_path}")

        # Generate image
        try:
            generate_image_from_kroki("mermaid", mermaid_code, img_path)
        except Exception as e:
            error_message = str(e)
            if "Syntax error in graph" in error_message or "Error 400" in error_message:
                print(f"Syntax error detected: {error_message}")
                print("Retrying with adjusted prompt...")
                return generate_mermaid_diagram(background_color, arrow_color, box_color, custom_prompt, retry_count - 1)
            else:
                raise  # Raise other unexpected errors

        # Ensure image is generated before applying background
        if not img_path.exists():
            raise FileNotFoundError(f"Error: Image file was not created at {img_path}")

        updated_img_path = apply_background_color(img_path, background_color, "mermaid")

        return updated_img_path  # Return updated image path with background color

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except RuntimeError as e:
        print(e)
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Retrying diagram generation...")

        return generate_mermaid_diagram(background_color, arrow_color, box_color, custom_prompt, retry_count - 1)

    return None




def generate_plantuml_diagram(background_color, arrow_color, box_color, custom_prompt=None, retry_count=3):
    """Generates a PlantUML diagram with dynamic colors and a custom prompt, with retry logic for syntax errors."""
    print(f"Prompt in generate plantuml{custom_prompt}")
    try:
        if retry_count <= 0:
            raise RuntimeError("Max retry attempts reached. Aborting diagram generation.")

        timestamp = get_timestamp()
        puml_path = MERMAID_FILE_PATH / f"plantuml_{timestamp}.puml"
        img_path = IMAGE_FILE_PATH / f"plantuml_{timestamp}.png"
        
        
        # Use custom prompt if provided, otherwise use default
        plantuml_prompt = f"Generate a Plantuml diagram. {custom_prompt} Background Color: {background_color}, Arrows: {arrow_color}, Boxes: {box_color}."
        

        response = generate_diagram(plantuml_prompt)
        print(f"Response from generate_diagram: {response}")

        if "```plantuml" in response:
            plantuml_code = response.split("```plantuml")[1].split("```")[0].strip()
        else:
            raise ValueError("Error: No PlantUML diagram found in the response.")

        save_diagram_code(plantuml_code, puml_path)

        # Ensure the PlantUML file is created
        if not puml_path.exists():
            raise FileNotFoundError(f"Error: PlantUML file was not created at {puml_path}")

        # Generate the image
        try:
            generate_image_from_kroki("plantuml", plantuml_code, img_path)
        except Exception as e:
            error_message = str(e)
            if "Syntax error" in error_message or "Error 400" in error_message:
                print(f"Syntax error detected: {error_message}")
                print("Retrying with adjusted prompt...")
                return generate_plantuml_diagram(background_color, arrow_color, box_color, custom_prompt, retry_count - 1)
            else:
                raise  # Raise unexpected errors

        # Ensure the image file is created
        if not img_path.exists():
            raise FileNotFoundError(f"Error: Image file was not created at {img_path}")

        return img_path  # Return image path

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except RuntimeError as e:
        print(e)
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Retrying diagram generation...")

        return generate_plantuml_diagram(background_color, arrow_color, box_color, custom_prompt, retry_count - 1)

    return None





def apply_background_color(image_path, background_color, diagram_type):
    try:
        if diagram_type.lower() != "mermaid":
            return image_path  # No need to apply background for PlantUML

        if not image_path.exists():
            raise FileNotFoundError(f"Error: File {image_path} not found.")

        # Convert HEX color to BGR
        bg_color_bgr = hex_to_bgr(background_color)

        # Load the diagram with unchanged mode to check alpha channel
        diagram = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if diagram is None:
            raise ValueError(f"Error: Failed to load image from {image_path}")

        h, w = diagram.shape[:2]

        if diagram.shape[2] == 4:  # RGBA image (has transparency)
            # Extract alpha channel
            alpha_channel = diagram[:, :, 3] / 255.0
            rgb_channels = diagram[:, :, :3]

            # Create solid background
            background = np.full((h, w, 3), bg_color_bgr, dtype=np.uint8)

            # Blend images using alpha channel
            blended = (rgb_channels * alpha_channel[:, :, None] + background * (1 - alpha_channel[:, :, None])).astype(np.uint8)

        else:  # If no transparency, overlay the image on solid background
            background = np.full((h, w, 3), bg_color_bgr, dtype=np.uint8)
            blended = background.copy()
            cv2.copyTo(diagram, None, blended)  # Direct copy

        # Save the modified image
        updated_image_path = image_path.parent / f"colored_{image_path.name}"
        cv2.imwrite(str(updated_image_path), blended)

        print(f"Updated image saved at: {updated_image_path}")
        return updated_image_path

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except AttributeError as e:
        print(f"Error: Unexpected image format or empty image - {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None
