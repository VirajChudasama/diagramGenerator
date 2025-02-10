import os
import json
import requests
from openai import OpenAI
from pathlib import Path

# Define Directories
PNG_DIR = Path(r"C:\Diagramgenerator\prompts_png")
MMD_DIR = Path(r"C:\Diagramgenerator\prompts_mmd")

# Ensure directories exist
PNG_DIR.mkdir(parents=True, exist_ok=True)
MMD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_next_counter(directory, prefix):
    """Find the next available counter for the given file type in the specified directory."""
    existing_files = list(directory.glob(f"{prefix}_*.png"))
    numbers = [int(f.stem.split("_")[-1]) for f in existing_files if f.stem.split("_")[-1].isdigit()]
    return max(numbers, default=0) + 1

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

# **1. Generate Mermaid Diagram**
mermaid_counter = get_next_counter(PNG_DIR, "mermaid")
mermaid_mmd_path = MMD_DIR / f"mermaid_{mermaid_counter}.mmd"
mermaid_img_path = PNG_DIR / f"mermaid_{mermaid_counter}.png"

mermaid_prompt = ("Generate a valid Mermaid diagram for an AI-powered document processing pipeline. "
                  "Ensure the diagram is **fully correct**, avoiding invalid syntax. "
                  "Use `\\n` for multi-line text (no `<br>`). Format it as a **flowchart**.")

mermaid_response = generate_diagram(mermaid_prompt)

if "```mermaid" in mermaid_response:
    mermaid_code = mermaid_response.split("```mermaid")[1].split("```")[0].strip()
else:
    print("Error: No Mermaid diagram found in the response.")
    exit(1)

# Save Mermaid diagram
save_diagram_code(mermaid_code, mermaid_mmd_path)
generate_image_from_kroki("mermaid", mermaid_code, mermaid_img_path)

# **2. Generate PlantUML Diagram**
plantuml_counter = get_next_counter(PNG_DIR, "plantuml")
plantuml_puml_path = MMD_DIR / f"plantuml_{plantuml_counter}.puml"
plantuml_img_path = PNG_DIR / f"plantuml_{plantuml_counter}.png"

plantuml_prompt = ("Generate a valid PlantUML diagram for an AI-powered document processing pipeline. "
                   "Ensure the syntax is fully correct and follows standard PlantUML conventions.")

plantuml_response = generate_diagram(plantuml_prompt)

if "```plantuml" in plantuml_response:
    plantuml_code = plantuml_response.split("```plantuml")[1].split("```")[0].strip()
else:
    print("Error: No PlantUML diagram found in the response.")
    exit(1)

# Save PlantUML diagram
save_diagram_code(plantuml_code, plantuml_puml_path)
generate_image_from_kroki("plantuml", plantuml_code, plantuml_img_path)


