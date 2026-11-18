from pathlib import Path

from src.preprocess import preprocess_image, INPUT_PATH
from src.ocr_engine import extract_text
from src.text_detection import detect_text_boxes, IMAGE_PATH
from src.evaluate import evaluate_ocr


def main():

    print("\n================================")
    print(" INTELLIGENT OCR TEXT EXTRACTION")
    print("================================")

    # Step 1: Image preprocessing
    print("\n[1] PREPROCESSING IMAGE")
    preprocess_image(INPUT_PATH)

    # Step 2: OCR text extraction
    print("\n[2] EXTRACTING TEXT")
    extract_text(IMAGE_PATH)

    # Step 3: Text detection
    print("\n[3] DETECTING TEXT")
    detect_text_boxes(IMAGE_PATH)

    # Step 4: OCR evaluation
    print("\n[4] EVALUATING OCR")
    evaluate_ocr()

    print("\n================================")
    print(" OCR PIPELINE COMPLETED!")
    print("================================")


if __name__ == "__main__":
    main()