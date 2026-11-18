import cv2
import pytesseract
from pathlib import Path
from jiwer import cer, wer

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "outputs" / "thresholded.png"


def evaluate_ocr():

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError("Processed image not found!")

    # Step 1: Extract text using Tesseract
    predicted_text = pytesseract.image_to_string(
        image,
        config="--oem 3 --psm 6"
    )

    # Step 2: Define the correct reference text
    reference_text = (
        "Hello! Welcome to OCR "
        "Tesseract reads printed text "
        "Computer Vision Practical"
    )

    # Step 3: Normalize whitespace
    reference_text = " ".join(reference_text.split())
    predicted_text = " ".join(predicted_text.split())

    # Step 4: Calculate evaluation metrics
    character_error = cer(reference_text, predicted_text)
    word_error = wer(reference_text, predicted_text)

    print("\nOCR EVALUATION RESULTS")
    print("-------------------------")
    print("Reference:", reference_text)
    print("Predicted:", predicted_text)

    print(f"CER: {character_error:.4f}")
    print(f"WER: {word_error:.4f}")

    print(f"CER Percentage: {character_error * 100:.2f}%")
    print(f"WER Percentage: {word_error * 100:.2f}%")


if __name__ == "__main__":
    evaluate_ocr()