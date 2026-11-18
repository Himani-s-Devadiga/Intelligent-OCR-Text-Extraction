
import cv2
import pytesseract
from pathlib import Path
from pytesseract import Output

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "outputs" / "thresholded.png"
OUTPUT_PATH = BASE_DIR / "outputs" / "detected_text.png"


def detect_text_boxes(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError("Image not found!")

    data = pytesseract.image_to_data(
        image,
        config="--oem 3 --psm 6",
        output_type=Output.DICT
    )

    detected_count = 0

    for i, word in enumerate(data["text"]):

        word = word.strip()
        confidence = float(data["conf"][i])

        if word and confidence >= 0:

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            # Draw bounding box
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display detected word
            cv2.putText(
                image,
                word,
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

            detected_count += 1

    cv2.imwrite(str(OUTPUT_PATH), image)

    print("Text detection completed!")
    print("Words detected:", detected_count)
    print("Saved output:", OUTPUT_PATH)


if __name__ == "__main__":
    detect_text_boxes(IMAGE_PATH)
