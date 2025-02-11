from flask import Flask, request, send_file, jsonify
import subprocess
import os
import glob

app = Flask(__name__)

# Define the folder where images are saved
OUTPUT_DIR = "C:/Diagramgenerator/prompts_png"

def get_latest_image():
    """Finds the most recently generated diagram with a logo."""
    files = glob.glob(os.path.join(OUTPUT_DIR, "diagram_with_logo_*.png"))
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)  # Get the newest file
    return latest_file

def run_script(script_name, *args):
    """Runs a given Python script with optional arguments."""
    try:
        subprocess.run(["python", script_name, *args], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        return False

@app.route("/generate", methods=["POST"])
def generate_diagram():
    """
    API endpoint to generate a diagram and optionally add a logo.
    Returns the image instead of a JSON response.
    """
    data = request.json  # Get JSON request data

    # Extract values from JSON request
    diagram_type = data.get("type")
    background_color = data.get("background_color")
    arrow_color = data.get("arrow_color")
    box_color = data.get("box_color")
    add_logo = data.get("add_logo", False)
    prompt = data.get("prompt")

    print(f"Received prompt: {prompt}")  # Debugging

    # Convert prompt to a single-line argument
    prompt_arg = prompt.replace("\n", " ")

    # Step 1: Generate the diagram
    diagram_success = run_script("diagram.py", diagram_type, background_color, arrow_color, box_color, prompt_arg)
    if not diagram_success:
        return jsonify({"error": "Failed to generate diagram"}), 500

    # Step 2: Overlay logo if requested
    if add_logo:
        logo_success = run_script("logo.py", background_color)
        if not logo_success:
            return jsonify({"error": "Failed to add logo"}), 500

    # Find the latest generated image
    latest_image = get_latest_image()
    
    if latest_image and os.path.exists(latest_image):
        print(f"Returning image: {latest_image}")
        return send_file(latest_image, mimetype="image/png")

    print("ERROR: Image not found")
    return jsonify({"error": "Image not found"}), 500

if __name__ == "__main__":
    app.run(debug=True)
