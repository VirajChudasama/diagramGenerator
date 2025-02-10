from flask import Flask, request, jsonify, send_file
import os
import json
import requests
from openai import OpenAI
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# Define Directories
PNG_DIR = Path(r"C:\Diagramgenerator\prompts_png")
MMD_DIR = Path(r"C:\Diagramgenerator\prompts_mmd")

# Ensure directories exist
PNG_DIR.mkdir(parents=True, exist_ok=True)
MMD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_diagram(prompt):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini"
    )
    return chat_completion.choices[0].message.content

def save_diagram_code(diagram_code, file_path):
    print(f"Saving to: {file_path}")  # Debug print
    print(f"Diagram Code:\n{diagram_code}")  # Debug print

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(diagram_code)

    if not file_path.exists():  # Check if file was created
        print(f"Error: {file_path} was not created!")
        return None

    print(f"Diagram saved at: {file_path}")
    return file_path

def generate_image_from_kroki(diagram_type, diagram_code, output_path):
    kroki_url = f"https://kroki.io/{diagram_type}/png"
    payload = {"diagram_source": diagram_code}  # Directly send the string
    headers = {"Content-Type": "application/json"}

    print(f"Sending to Kroki ({diagram_type}):\n{diagram_code}")  # Debug print

    response = requests.post(kroki_url, headers=headers, data=json.dumps(payload))

    print("Kroki Response Code:", response.status_code)
    print("Kroki Response Text:", response.text)

    if response.status_code == 200:
        with open(output_path, "wb") as img_file:
            img_file.write(response.content)
        print(f"Diagram image saved at: {output_path}")
        return output_path
    else:
        print(f"Error: Failed to generate {diagram_type} diagram.")
        return None

def generate_mermaid_diagram(background_color, arrow_color, box_color):
    timestamp = get_timestamp()
    mmd_path = MMD_DIR / f"mermaid_{timestamp}.mmd"
    img_path = PNG_DIR / f"mermaid_{timestamp}.png"

    mermaid_prompt = (f"Generate a valid Mermaid diagram for an AI-powered document processing pipeline. "
                      f"Use background color `{background_color}`, arrow color `{arrow_color}`, and box color `{box_color}`. "
                      "Ensure the diagram is **fully correct**, avoiding invalid syntax. "
                      "Use `\\n` for multi-line text (no `<br>`). Format it as a **flowchart** and apply proper styling.")

    response = generate_diagram(mermaid_prompt)

    if "```mermaid" in response:
        mermaid_code = response.split("```mermaid")[1].split("```")[0].strip()
        save_diagram_code(mermaid_code, mmd_path)  # Save for debugging
        return generate_image_from_kroki("mermaid", mermaid_code, img_path)  # Directly send to Kroki
    else:
        print("Error: No Mermaid diagram found in the response.")
        return None

def generate_plantuml_diagram(background_color, arrow_color, box_color):
    timestamp = get_timestamp()
    puml_path = MMD_DIR / f"plantuml_{timestamp}.puml"
    img_path = PNG_DIR / f"plantuml_{timestamp}.png"

    plantuml_prompt = (f"Generate a valid PlantUML diagram for an AI-powered document processing pipeline. "
                       f"Use background color `{background_color}`, arrow color `{arrow_color}`, and box color `{box_color}`. "
                       "Ensure the syntax is fully correct and follows standard PlantUML conventions. Apply styling to match the colors.")

    response = generate_diagram(plantuml_prompt)

    if "```plantuml" in response:
        plantuml_code = response.split("```plantuml")[1].split("```")[0].strip()
        save_diagram_code(plantuml_code, puml_path)  # Save for debugging
        return generate_image_from_kroki("plantuml", plantuml_code, img_path)  # Directly send to Kroki
    else:
        print("Error: No PlantUML diagram found in the response.")
        return None

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    diagram_type = data.get("type", "mermaid").lower()
    background_color = data.get("background_color", "#FFFFFF")
    arrow_color = data.get("arrow_color", "#000000")
    box_color = data.get("box_color", "#AAAAAA")

    print(f"Received request: {data}")  # Debug print

    if diagram_type == "mermaid":
        img_path = generate_mermaid_diagram(background_color, arrow_color, box_color)
    elif diagram_type == "plantuml":
        img_path = generate_plantuml_diagram(background_color, arrow_color, box_color)
    else:
        return jsonify({"error": "Invalid diagram type. Use 'mermaid' or 'plantuml'."}), 400

    if img_path:
        return send_file(img_path, mimetype='image/png')
    else:
        return jsonify({"error": "Failed to generate diagram."}), 500

if __name__ == "__main__":
    app.run(debug=True)
