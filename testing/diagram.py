from flask import Flask, request, jsonify, send_file
import os
from diagram_service import generate_mermaid_diagram, generate_plantuml_diagram, addLogo

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        print(f"Received request: {data}")  # Debug print

        diagram_type = data.get("diagram_type", "").lower()
        background_color = data.get("background_color", "#FFFFFF")
        arrow_color = data.get("arrow_color", "#000000")
        box_color = data.get("box_color", "#000000")
        add_logo = data.get("add_logo", False)
        prompt_arg = data.get("prompt", "")
        
        img_path = None
        if diagram_type == "mermaid":
            img_path = generate_mermaid_diagram(background_color, arrow_color, box_color, prompt_arg)
        elif diagram_type == "plantuml":
            img_path = generate_plantuml_diagram(background_color, arrow_color, box_color, prompt_arg)
        else:
            return jsonify({"error": "Invalid diagram type. Use 'mermaid' or 'plantuml'."}), 400
        
        if not img_path or not os.path.exists(img_path):
            return jsonify({"error": "Failed to generate diagram."}), 500
        
        if add_logo and os.path.exists(img_path):
            img_path = addLogo(img_path, background_color)
        
        if not os.path.exists(img_path):
            return jsonify({"error": "Generated image not found"}), 500
        
        return send_file(img_path, mimetype='image/png')
    
    except Exception as e:
        print(f"Error in generate_image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
