
import cv2
from pathlib import Path

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "images" / "sample_text.png"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)


def preprocess_image(image_path):

    # Step 1: Read image
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("Original image shape:", image.shape)

    # Step 2: Convert BGR to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Resize image
    resized = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Step 4: Remove noise using Median Blur
    denoised = cv2.medianBlur(resized, 3)

    # Step 5: Apply Otsu thresholding
    _, thresholded = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

       # Save processed images
    files = {
        "grayscale.png": gray,
        "resized.png": resized,
        "denoised.png": denoised,
        "thresholded.png": thresholded
    }

    for filename, processed_image in files.items():

        output_path = OUTPUT_DIR / filename

        saved = cv2.imwrite(
            str(output_path),
            processed_image
        )

        print(f"{filename}: {'Saved' if saved else 'FAILED'}")

    return thresholded


if __name__ == "__main__":
    processed_image = preprocess_image(INPUT_PATH)

    print("Final processed image shape:", processed_image.shape)
    print("Output folder:", OUTPUT_DIR)
