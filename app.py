
import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output
from jiwer import cer, wer

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Page configuration
st.set_page_config(
    page_title="Intelligent OCR Text Extraction",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# Application title
st.title("📄 Intelligent OCR Text Extraction")

st.write(
    "Upload an image and extract readable text "
    "using Computer Vision and OCR."
)

# Image upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Extract text button
    if st.button("Extract Text", type="primary"):

        # Image preprocessing
        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        resized = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        denoised = cv2.medianBlur(resized, 3)

        _, thresholded = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        st.session_state.thresholded = thresholded

        # OCR extraction
        config = "--oem 3 --psm 6"

        extracted_text = pytesseract.image_to_string(
            thresholded,
            config=config
        )

        st.session_state.extracted_text = extracted_text

        # Word-level detection and confidence
        data = pytesseract.image_to_data(
            thresholded,
            config=config,
            output_type=Output.DICT
        )

        words = []
        confidences = []

        annotated = cv2.cvtColor(
            thresholded,
            cv2.COLOR_GRAY2BGR
        )

        for i, word in enumerate(data["text"]):

            word = word.strip()
            confidence = float(data["conf"][i])

            if word and confidence >= 0:

                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]

                words.append(word)
                confidences.append(round(confidence, 2))

                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

        st.session_state.annotated = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        st.session_state.words = words
        st.session_state.confidences = confidences

    # Display saved OCR results
    if st.session_state.extracted_text:

        st.subheader("Preprocessed Image")
        st.image(
            st.session_state.thresholded,
            use_container_width=True
        )

        st.subheader("Extracted Text")
        st.text_area(
            "Recognized text",
            value=st.session_state.extracted_text,
            height=150
        )

        st.subheader("Detected Words and Bounding Boxes")
        st.image(
            st.session_state.annotated,
            caption="Detected text regions",
            use_container_width=True
        )

        st.subheader("Word-Level Confidence Scores")

        st.dataframe(
            {
                "Word": st.session_state.words,
                "Confidence (%)": st.session_state.confidences
            },
            use_container_width=True
        )

        # Manual reference text
        st.subheader("Reference Text")

        reference_text = st.text_area(
            "Enter the correct text manually",
            placeholder="Type the actual text present in the image...",
            height=150
        )

        # OCR evaluation
        if st.button("Evaluate OCR"):

            if not reference_text.strip():

                st.warning(
                    "Please enter the reference text first."
                )

            else:

                reference = " ".join(
                    reference_text.split()
                )

                predicted = " ".join(
                    st.session_state.extracted_text.split()
                )

                character_error = cer(
                    reference,
                    predicted
                )

                word_error = wer(
                    reference,
                    predicted
                )

                st.subheader("OCR Evaluation Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Character Error Rate (CER)",
                        f"{character_error * 100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Word Error Rate (WER)",
                        f"{word_error * 100:.2f}%"
                    )

                st.success(
                    "OCR evaluation completed!"
                )
