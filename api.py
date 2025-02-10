import os
from openai import OpenAI
import requests
import zlib
import base64
import cv2
import numpy as np

# Define paths
BPMN_FILE_PATH = r"C:\Diagramgenerator\diagram.bpmn"
IMAGE_FILE_PATH = r"C:\Diagramgenerator\diagram.svg"

# Initialize OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Create a chat completion
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Generate a Netflix workflow using BPMN XML format. Provide only the BPMN code inside triple backticks, like this: ```bpmn ... ```"
        }
    ],
    model="gpt-4o-mini"
)

# Extract the BPMN code from the response
response_message = chat_completion.choices[0].message.content

# Extract BPMN code from response
if "```bpmn" in response_message:
    bpmn_code = response_message.split("```bpmn")[1].split("```")[0].strip()
else:
    print("Error: No BPMN diagram found in the response.")
    exit(1)

# Save BPMN code to a file
with open(BPMN_FILE_PATH, "w") as f:
    f.write(bpmn_code)

print(f"BPMN diagram saved as text at: {BPMN_FILE_PATH}")


def encode_for_kroki(diagram_code):
    """ Compresses and encodes BPMN code for Kroki API """
    compressed = zlib.compress(diagram_code.encode("utf-8"))
    return base64.urlsafe_b64encode(compressed).decode("utf-8")


def generate_bpmn_image(bpmn_code, output_path):
    """ Converts BPMN code into an image using Kroki API and saves it. """
    encoded_bpmn = encode_for_kroki(bpmn_code)
    kroki_url = f"https://kroki.io/bpmn/svg/{encoded_bpmn}"

    response = requests.get(kroki_url)

    if response.status_code == 200:
        with open(output_path, "wb") as img_file:
            img_file.write(response.content)
        print(f"BPMN diagram image saved at: {output_path}")
    else:
        print(f"Failed to generate BPMN image. HTTP Status: {response.status_code}, Response: {response.text}")


# Generate and save the BPMN image
generate_bpmn_image(bpmn_code, IMAGE_FILE_PATH)



diagram_path = r"C:\Diagramgenerator\diagram.png"
logo_path = r"C:\Diagramgenerator\logo.jpg"
output_path = r"C:\Diagramgenerator\diagram_with_logo.png"

# Load the main diagram image
diagram = cv2.imread(diagram_path, cv2.IMREAD_UNCHANGED)
if diagram is None:
    print("Error: Could not load diagram image.")
    exit(1)

# Load the logo image
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
if logo is None:
    print("Error: Could not load logo image.")
    exit(1)

# Get dimensions of the main image and logo
(h_img, w_img, _) = diagram.shape
(h_logo, w_logo, _) = logo.shape

# Define position (bottom-right corner)
x_offset = w_img - w_logo - 10  # 10px padding from the right
y_offset = h_img - h_logo - 10  # 10px padding from the bottom

# Overlay the logo on the diagram
roi = diagram[y_offset:y_offset + h_logo, x_offset:x_offset + w_logo]

# Convert logo to grayscale and create mask
logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(logo_gray, 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Black-out the area of the logo in the ROI
bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Extract logo region
fg = cv2.bitwise_and(logo, logo, mask=mask)

# Add logo to the main image
combined = cv2.add(bg, fg)
diagram[y_offset:y_offset + h_logo, x_offset:x_offset + w_logo] = combined

# Save the result
cv2.imwrite(output_path, diagram)
print(f"Diagram with logo saved at: {output_path}")
