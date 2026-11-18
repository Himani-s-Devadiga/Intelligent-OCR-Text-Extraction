# Intelligent OCR Text Extraction

An Optical Character Recognition (OCR) project that extracts readable text from images using image preprocessing and Tesseract OCR.

## Project Overview

This project explores how image processing techniques can improve text recognition. It includes image preprocessing, OCR-based text extraction, text detection, and recognition evaluation.

## Features

- Image upload and display.
- Image preprocessing using OpenCV.
- Grayscale conversion and image resizing.
- Noise removal and thresholding.
- Text extraction using Tesseract OCR.
- Text detection with bounding boxes.
- Word-level confidence scores.
- Character Error Rate (CER) and Word Error Rate (WER) evaluation.
- Greedy and Beam Search CTC decoding implementations.

## Technologies Used

- Python
- OpenCV
- NumPy
- Pillow
- Tesseract OCR
- Pytesseract
- Streamlit
- Jiwer

## Project Structure

```text
Intelligent-OCR-Text-Extraction/
│
├── images/
│   └── sample_text.png
│
├── outputs/
│   ├── denoised.png
│   ├── detected_text.png
│   ├── grayscale.png
│   ├── resized.png
│   └── thresholded.png
│
├── src/
│   ├── beam_decoder.py
│   ├── compare_decoders.py
│   ├── ctc_decoder.py
│   ├── evaluate.py
│   ├── ocr_engine.py
│   ├── preprocess.py
│   └── text_detection.py
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Himani-s-Devadiga/Intelligent-OCR-Text-Extraction.git
cd Intelligent-OCR-Text-Extraction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Tesseract OCR

Install Tesseract OCR separately and ensure its executable is accessible to Pytesseract.

## Run the Application

```bash
streamlit run app.py
```

## Learning Outcomes

- Understanding OCR fundamentals.
- Applying image preprocessing techniques.
- Performing text detection and recognition.
- Evaluating OCR output using CER and WER.
- Understanding CTC decoding, Greedy Search, and Beam Search.
- Building an interactive OCR application using Streamlit.

## Author

**Himani S K**

GitHub: [Himani-s-Devadiga](https://github.com/Himani-s-Devadiga)