import subprocess

def run_script(script_name, *args):
    """Runs a given Python script with optional arguments."""
    try:
        subprocess.run(["python", script_name, *args], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")

if __name__ == "__main__":
    print("Generating diagrams...")
    run_script("mermaid-plantuml-dynamic.py")
    
    add_logo = input("Do you want to add a logo? (yes/no): ").strip().lower()
    if add_logo == "yes":
        background_color = "#F5F5DC"  # Beige
        print("Processing diagram with OpenCV and adding logo...")
        run_script("overlaytext.py", background_color)
    
    print("Process completed successfully.")
