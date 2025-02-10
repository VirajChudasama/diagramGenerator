import os
import json
import requests
from openai import OpenAI
from pathlib import Path
from datetime import datetime
import sys

# Define Directories
PNG_DIR = Path(r"C:\Diagramgenerator\prompts_png")
MMD_DIR = Path(r"C:\Diagramgenerator\prompts_mmd")

# Ensure directories exist
PNG_DIR.mkdir(parents=True, exist_ok=True)
MMD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_timestamp():
    """Returns the current timestamp in a safe filename format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_diagram(prompt):
    """Generates a diagram using OpenAI."""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini"
    )
    return chat_completion.choices[0].message.content

def save_diagram_code(diagram_code, file_path):
    """Saves diagram code to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(diagram_code)
    print(f"Diagram saved at: {file_path}")

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
    mmd_path = MMD_DIR / f"mermaid_{timestamp}.mmd"
    img_path = PNG_DIR / f"mermaid_{timestamp}.png"

    # Use custom prompt if provided, otherwise use default
    mermaid_prompt = f"{custom_prompt} Background Color: {background_color}, Arrows: {arrow_color}, Boxes: {box_color}. " \
                 "Ensure the diagram is **fully correct**, avoiding invalid syntax. " \
                 "Use `\\n` for multi-line text (no `<br>`). Format it as a **flowchart** and apply proper styling."


    response = generate_diagram(mermaid_prompt)

    if "```mermaid" in response:
        mermaid_code = response.split("```mermaid")[1].split("```")[0].strip()
    else:
        print("Error: No Mermaid diagram found in the response.")
        return None

    save_diagram_code(mermaid_code, mmd_path)
    generate_image_from_kroki("mermaid", mermaid_code, img_path)
    return img_path  # Return image path

def generate_plantuml_diagram(background_color, arrow_color, box_color, custom_prompt=None):
    """Generates a PlantUML diagram with dynamic colors and a custom prompt."""
    timestamp = get_timestamp()
    puml_path = MMD_DIR / f"plantuml_{timestamp}.puml"
    img_path = PNG_DIR / f"plantuml_{timestamp}.png"

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


# Example usage
background_color = "#F5F5DC"  # Beige
arrow_color = "#FF4500"  # OrangeRed
box_color = "#87CEEB"  # SkyBlue


# Read command-line arguments
diagram_type = sys.argv[1]  # "mermaid" or "plantuml"
background_color = sys.argv[2]
arrow_color = sys.argv[3]
box_color = sys.argv[4]
custom_prompt = sys.argv[5]  # This should now be correctly passed

print(f"Using custom prompt: {custom_prompt}")  # Debugging

if diagram_type == "mermaid":
    generate_mermaid_diagram(background_color, arrow_color, box_color, custom_prompt)
elif diagram_type == "plantuml":
    generate_plantuml_diagram(background_color, arrow_color, box_color, custom_prompt)
else:
    print("Error: Unsupported diagram type")

#generate_mermaid_diagram(background_color, arrow_color, box_color)
#generate_plantuml_diagram(background_color, arrow_color, box_color)
#print(background_color)
