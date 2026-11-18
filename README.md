\# Intelligent OCR Text Extraction



An Optical Character Recognition (OCR) project that extracts readable text from images using image preprocessing and Tesseract OCR.



\## Project Overview



This project explores how image processing techniques can improve text recognition. It includes image preprocessing, OCR-based text extraction, text detection, and recognition evaluation.



\## Features



\- Image upload and display.

\- Image preprocessing using OpenCV.

\- Grayscale conversion and resizing.

\- Noise removal and thresholding.

\- Text extraction using Tesseract OCR.

\- Text detection and word-level bounding boxes.

\- Word-level confidence scores.

\- OCR evaluation using Character Error Rate (CER) and Word Error Rate (WER).

\- Greedy and Beam Search CTC decoder learning implementations.



\## Technologies Used



\- Python

\- OpenCV

\- NumPy

\- Pillow

\- Tesseract OCR

\- Pytesseract

\- Streamlit

\- Jiwer



\## Project Structure



```text

Intelligent-OCR-Text-Extraction/

│

├── images/

│   └── sample\_text.png

│

├── outputs/

│   ├── grayscale.png

│   ├── resized.png

│   ├── denoised.png

│   ├── thresholded.png

│   └── detected\_text.png

│

├── src/

│   ├── preprocess.py

│   ├── ocr\_engine.py

│   ├── text\_detection.py

│   ├── evaluate.py

│   ├── ctc\_decoder.py

│   ├── beam\_decoder.py

│   └── compare\_decoders.py

│

├── app.py

├── main.py

├── requirements.txt

└── README.md

```



\## Installation



Clone the repository:



```bash

git clone https://github.com/Himani-s-Devadiga/Intelligent-OCR-Text-Extraction.git

cd Intelligent-OCR-Text-Extraction

```



Create and activate a virtual environment:



```bash

python -m venv venv

venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Install Tesseract OCR separately and configure its executable path if required.



\## Run the Application



```bash

streamlit run app.py

```



\## Learning Outcomes



\- Understanding image preprocessing techniques for OCR.

\- Learning text detection and recognition.

\- Evaluating OCR performance using CER and WER.

\- Understanding CRNN, RNN, LSTM and CTC decoding concepts.

\- Implementing basic Greedy and Beam Search CTC decoders.



\## Author



Himani S. Devadiga



GitHub: https://github.com/Himani-s-Devadiga

