
import cv2
import pytesseract
from pathlib import Path
from pytesseract import Output

# Configure Tesseract executable
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "outputs" / "thresholded.png"


def extract_text(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Extract complete text
    config = "--oem 3 --psm 6"

    extracted_text = pytesseract.image_to_string(
        image,
        config=config
    )

    print("\nEXTRACTED TEXT:")
    print("-------------------------")
    print(extracted_text)

    # Extract word-level information
    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=Output.DICT
    )

    print("\nWORD-LEVEL RESULTS:")
    print("-------------------------")

    for i, word in enumerate(data["text"]):

        word = word.strip()
        confidence = float(data["conf"][i])

        if word and confidence >= 0:
            print(
                f"Word: {word} | "
                f"Confidence: {confidence:.2f}%"
            )

    return extracted_text


if __name__ == "__main__":
    extract_text(IMAGE_PATH)
