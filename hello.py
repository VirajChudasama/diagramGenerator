import os
from openai import OpenAI
import requests
import json

# Define paths
MERMAID_FILE_PATH = r"C:\Diagramgenerator\diagram.mmd"
IMAGE_FILE_PATH = r"C:\Diagramgenerator\diagram.png"

# Initialize the OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Generate Mermaid diagram using OpenAI
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Generate a valid Mermaid diagram for an AI-powered document processing pipeline. "
                       "Ensure the diagram is **fully correct**, avoiding invalid syntax. "
                       "Use `\\n` for multi-line text (no `<br>`). Format it as a **flowchart**."
        }
    ],
    model="gpt-4o-mini"
)

# Extract the Mermaid code from response
response_message = chat_completion.choices[0].message.content

if "```mermaid" in response_message:
    mermaid_code = response_message.split("```mermaid")[1].split("```")[0].strip()
else:
    print("Error: No Mermaid diagram found in the response.")
    exit(1)

# Ensure the Mermaid code is **not empty**
if not mermaid_code.strip():
    print("Error: Mermaid diagram is empty!")
    exit(1)

# Save Mermaid code to a file
with open(MERMAID_FILE_PATH, "w", encoding="utf-8") as f:
    f.write(mermaid_code)

print(f"Mermaid diagram saved at: {MERMAID_FILE_PATH}")

def generate_image_from_mermaid(mermaid_code, output_path):
    """ Converts Mermaid code into an image using the Kroki API. """
    kroki_url = "https://kroki.io/mermaid/png"

    # Prepare JSON payload correctly
    payload = {"diagram_source": mermaid_code}  
    headers = {"Content-Type": "application/json"}

    # Send POST request
    response = requests.post(kroki_url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        with open(output_path, "wb") as img_file:
            img_file.write(response.content)
        print(f"Diagram image saved at: {output_path}")
    else:
        print(f"Failed to generate image. HTTP Status: {response.status_code}, Response: {response.text}")

# Generate and save the image
generate_image_from_mermaid(mermaid_code, IMAGE_FILE_PATH)



