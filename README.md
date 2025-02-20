# Document Processing and Retrieval System

## Overview
This project is a document processing and retrieval system that enables users to upload PDFs, extract text using PaddleOCR, index the extracted text using FAISS, and perform intelligent querying with Groq's Llama-3.3-70b-versatile model. The system separates titles and sentences from the extracted text to improve retrieval efficiency.

## Features
- **PDF Upload**: Users can upload PDF files for processing.
- **Text Extraction**: Uses PaddleOCR to extract text and bounding box information from PDF images.
- **Title & Sentence Separation**: Identifies titles and sentences separately for better indexing.
- **FAISS Indexing**: Creates FAISS indices for efficient nearest-neighbor search.
- **Semantic Search**: Retrieves the most relevant titles and sentences for a given query.
- **Groq Llama-3 Integration**: Uses Groq's API to generate responses based on retrieved context.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Virtual environment (optional but recommended)

### Dependencies
Install the required dependencies using pip:
```sh
pip install flask numpy faiss-cpu pickle-mixin sentence-transformers pdf2image paddleocr groq flask-cors
```
Additionally, install Tesseract OCR (for PaddleOCR) and Poppler:
For UBUNTU
```sh
sudo apt install tesseract-ocr poppler-utils
```
>-for Windows:<br>
<ol>
  <li>Install Tesseract OCR</li>
  <li>Install Poppler</li>
</ol>

