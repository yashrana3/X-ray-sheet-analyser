from PIL import Image

def validate_xray_image(image_path):
    try:
        img = Image.open(image_path)
        
        # Check if image is grayscale (mode should be "L" for grayscale)
        if img.mode != "L":
            return "Error: Only grayscale lung X-rays are allowed."

        # Check image dimensions (modify as per dataset)
        if img.size[0] < 96 or img.size[1] < 96:
            return "Error: Image resolution is too low."

        return "Valid X-ray image."
    
    except Exception as e:
        return f"Invalid image file: {str(e)}"

# Example usage
print(validate_xray_image("assets/robot.jpg"))