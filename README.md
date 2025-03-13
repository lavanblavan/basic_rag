
# RAG with simple libraries with paddleocr

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
>-For UBUNTU
```sh
sudo apt install tesseract-ocr poppler-utils
```
>-for Windows:<br>
<ol>
  <li>Install Tesseract OCR</li>
  <li>Install Poppler</li>
</ol>
<h2> <em><strong>Usage</strong></em></h2>
<em><strong>Running the Server</strong></em>
<ol>
  <li>Clone the repository and navigate to the project folder.</li>
  <li>Run the Flask server:
  </li>
<pre><code class="language-bash">$ python app.py</code></pre>
  <li>The API will start at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).</li>
</ol>
<h2> <em><strong>API End Points</strong></em></h2>
<ol>
  <li> Upload pdf
    <ul>
  <li><strong>Endpoint: /upload</strong></li>
      <li><strong> Method: POST</strong></li>
      <li><strong>Description: Uploads and processes a PDF file.</strong></li>
      <li><strong>Request: Form-data with a file field named file.</strong></li>
      <li><strong>Response:
      <pre><code>{"message": "File processed successfully."}</code></pre></strong></li>
</ul>
  </li>
</ol>
<h3>2. Query the Processed Document</h3>
    <ul>
        <li><strong>Endpoint:</strong> <code>/query</code></li>
        <li><strong>Method:</strong> <code>POST</code></li>
        <li><strong>Description:</strong> Searches the FAISS index and retrieves relevant text using Groq.</li>
    </ul>
<p><strong>Request:</strong></p>
    <pre><code>{
    "query": "What is the content about?"
}</code></pre>
    <p><strong>Response:</strong></p>
    <pre><code>{
    "response": "Generated answer based on the document."
}</code></pre>

  <h2>File Structure</h2>
 <pre><code>.
├── app.py                # Main Flask application
├── find_words.py         # PaddleOCR text extraction module
├── uploads/              # Folder for uploaded PDFs
├── output_images/        # Folder for OCR output images
├── title_index.faiss     # FAISS index for titles
├── sentence_index.faiss  # FAISS index for sentences
├── titles.pkl            # Pickle file storing extracted titles
├── sentences.pkl         # Pickle file storing extracted sentences
├── README.md             # Project documentation
</code></pre>
<h2>How it Works</h2>
    <ol>
        <li><strong>PDF Upload</strong>: The user uploads a PDF.</li>
        <li><strong>Text Extraction</strong>: PaddleOCR extracts text and bounding boxes from images.</li>
        <li><strong>Title & Sentence Separation</strong>: Titles and sentences are identified using heuristics.</li>
        <li><strong>FAISS Indexing</strong>: Titles and sentences are embedded and indexed separately.</li>
        <li><strong>Query Processing</strong>:
            <ul>
                <li>FAISS retrieves relevant text based on a query.</li>
                <li>Groq generates a response using the retrieved context.</li>
            </ul>
        </li>
        <li><strong>Response</strong>: The system returns an AI-generated answer based on document content.</li>
    </ol>
    <h2>Environment Variables</h2>
    <p>Set the following environment variable before running the application:</p>
    <pre><code>export GROQ_API_KEY="your_api_key"</code></pre>

  <h2>Future Improvements</h2>
    <ul>
        <li>Improve title and sentence classification heuristics.</li>
        <li>Add support for multiple languages in OCR.</li>
        <li>Enhance indexing with hierarchical FAISS structures.</li>
    </ul>


https://github.com/user-attachments/assets/a9865f75-7a8a-4a16-9a35-a69bb003802e


   <h2>License</h2>
   
    <p>This project is licensed under the MIT License.</p>
