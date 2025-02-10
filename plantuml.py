import os
import json
import requests
from openai import OpenAI

# Define Paths
MERMAID_FILE_PATH = r"C:\Diagramgenerator\diagram.mmd"
MERMAID_IMAGE_PATH = r"C:\Diagramgenerator\diagram.png"
PLANTUML_FILE_PATH = r"C:\Diagramgenerator\diagram.puml"
PLANTUML_IMAGE_PATH = r"C:\Diagramgenerator\plantuml.png"

# Initialize OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_diagram(prompt):
    """Generates a diagram using OpenAI."""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini"
    )
    return chat_completion.choices[0].message.content

# **1. Generate Mermaid Diagram**
mermaid_prompt = ("Generate a valid Mermaid diagram for an AI-powered document processing pipeline. "
                  "Ensure the diagram is **fully correct**, avoiding invalid syntax. "
                  "Use `\\n` for multi-line text (no `<br>`). Format it as a **flowchart**.")

mermaid_response = generate_diagram(mermaid_prompt)

if "```mermaid" in mermaid_response:
    mermaid_code = mermaid_response.split("```mermaid")[1].split("```")[0].strip()
else:
    print("Error: No Mermaid diagram found in the response.")
    exit(1)

# Save Mermaid code to a file
with open(MERMAID_FILE_PATH, "w", encoding="utf-8") as f:
    f.write(mermaid_code)

print(f"Mermaid diagram saved at: {MERMAID_FILE_PATH}")

def generate_image_from_kroki(diagram_type, diagram_code, output_path):
    """ Converts Mermaid or PlantUML code into an image using the Kroki API. """
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

# Generate and save Mermaid image
generate_image_from_kroki("mermaid", mermaid_code, MERMAID_IMAGE_PATH)

# **2. Generate PlantUML Diagram**
plantuml_prompt = ("Generate a valid PlantUML diagram for an AI-powered document processing pipeline. "
                   "Ensure the syntax is fully correct and follows standard PlantUML conventions.")

plantuml_response = generate_diagram(plantuml_prompt)

if "```plantuml" in plantuml_response:
    plantuml_code = plantuml_response.split("```plantuml")[1].split("```")[0].strip()
else:
    print("Error: No PlantUML diagram found in the response.")
    exit(1)

# Save PlantUML code to a file
with open(PLANTUML_FILE_PATH, "w", encoding="utf-8") as f:
    f.write(plantuml_code)

print(f"PlantUML diagram saved at: {PLANTUML_FILE_PATH}")

# Generate and save PlantUML image
generate_image_from_kroki("plantuml", plantuml_code, PLANTUML_IMAGE_PATH)
