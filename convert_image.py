from PIL import Image
import sys

def resize_image(input_path, output_path, size=(32, 32)):
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGBA")  # Ensure consistent color mode
            resized_img = img.resize(size, Image.NEAREST)  # NEAREST for pixel-art style
            resized_img.save(output_path)
            print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python resize_to_32x32.py <input_image_path> <output_image_path>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        resize_image(input_file, output_file)
