
import os
from PIL import Image
import numpy as np

def load_image(path):
    # Load the image using your preferred method
    # Example: Using PIL
    image = Image.open(path)
    return image

def concatenate_images(image1, image2, axis=0):
    # Concatenate the images using your preferred method
    # Example: Using PIL
    
    print(np.array(image1).shape, np.array(image2).shape, image1.mode, image2.mode)
    
    if image2.mode == 'RGBA':
        image2 = image2.convert('RGB')

    if image1.mode == 'RGBA':
        image1 = image1.convert('RGB')

    if axis == 0:

        return Image.fromarray(np.concatenate((np.array(image1), np.array(image2)), axis=0))
    elif axis == 1:
        
        return Image.fromarray(np.concatenate((np.array(image1), np.array(image2)), axis=1))
    else:
        raise ValueError("Invalid axis value")
    
def save_image(image, path):
    # Save the image using your preferred method
    # Example: Using PIL
    image.save(path)

def form_image(string):
    image_path = "/home/vivek/Downloads/Letters/"
    image_extension = ".png"
    image_files = [f"img_{char}{image_extension}" for char in string]
    image_paths = [os.path.join(image_path, file) for file in image_files]
    
    # Join the images here using your preferred method
    
    # Example: Concatenate the images horizontally
    joined_image = None
    
    for path in image_paths:
        image = load_image(path)
        if joined_image is None:
            joined_image = image
        else:
            joined_image = concatenate_images(joined_image, image, axis=1)
    
    # Save the joined image
    save_image(joined_image, "test-case.png")
    
    # Return the joined image path
    return "test-case.png"


# Example usage
input_string = "Vivek  Hriswitha  Siddhu  SOHITH"
joined_image_path = form_image(input_string)
print(f"Joined image path: {joined_image_path}")
