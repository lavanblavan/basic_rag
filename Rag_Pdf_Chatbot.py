from flask import Flask, request, jsonify
import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
from find_words import txt_find
from groq import Groq
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def save_data_to_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def separate_titles_and_sentences(texts, boxes):
    titles = []
    sentences = []

    for i, (text, box) in enumerate(zip(texts, boxes)):
        width = box[1][0] - box[0][0]  # Calculate width (x1 - x0)

        # Clean the extracted text
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        cleaned_text = cleaned_text.replace('\n', ' ')

        if width <= 400:  # Title heuristic (adjust 200 as needed)
            titles.append(cleaned_text)
        else:
            # Check if the text ends with punctuation
            if re.search(r'[.?!]$', cleaned_text):
                sentences.append(cleaned_text)
            else:
                # Handle incomplete sentences by merging consecutive short texts
                combined_text = cleaned_text
                combined_box_width = width

                for j in range(i + 1, len(texts)):
                    next_text = texts[j]
                    next_box = boxes[j]
                    next_width = next_box[1][0] - next_box[0][0]

                    cleaned_next_text = re.sub(r'\s+', ' ', next_text).strip()
                    cleaned_next_text = cleaned_next_text.replace('\n', ' ')

                    if combined_box_width + next_width <= 200:
                        # Merge if width is still within threshold
                        combined_text += " " + cleaned_next_text
                        combined_box_width += next_width
                    elif re.search(r'[.?!]$', cleaned_next_text):
                        # If punctuation is found, finalize sentence
                        combined_text += " " + cleaned_next_text
                        sentences.append(combined_text)
                        break
                    else:
                        # If next word is too wide and no punctuation, finalize current combined text
                        sentences.append(combined_text)
                        break

                # If the last sentence in the loop had no punctuation, add it
                if not re.search(r'[.?!]$', combined_text):
                    sentences.append(combined_text)

    return titles, sentences

def process_pdf(pdf_path):
    texts, boxes, _, _ = txt_find(convert_from_path(pdf_path))
    titles, sentences = separate_titles_and_sentences(texts, boxes)

    # Create embeddings for titles and sentences
    title_embeddings = model.encode(titles)
    sentence_embeddings = model.encode(sentences)

    # Create FAISS indices for titles and sentences
    title_index = create_faiss_index(title_embeddings)
    sentence_index = create_faiss_index(sentence_embeddings)

    # Save the FAISS indices to files
    faiss.write_index(title_index, "title_index.faiss")
    faiss.write_index(sentence_index, "sentence_index.faiss")

    # Save titles and sentences to .pkl files
    save_data_to_pkl(titles, "titles.pkl")
    save_data_to_pkl(sentences, "sentences.pkl")

    return "Indexing completed."

def search_faiss_index(query):
    query_embedding = model.encode([query])[0]

    # Load FAISS indices
    title_index = faiss.read_index("title_index.faiss")
    sentence_index = faiss.read_index("sentence_index.faiss")

    # Load titles and sentences
    titles = pickle.load(open("titles.pkl", "rb"))
    sentences = pickle.load(open("sentences.pkl", "rb"))

    # Search titles and sentences
    _, title_indices = title_index.search(np.array([query_embedding]).astype("float32"), 5)
    _, sentence_indices = sentence_index.search(np.array([query_embedding]).astype("float32"), 5)

    retrieved_titles = [titles[i] for i in title_indices[0]]
    retrieved_sentences = [sentences[i] for i in sentence_indices[0]]

    return retrieved_titles, retrieved_sentences

def get_groq_response(query, context):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Use the provided context to answer accurately."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content.strip()

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    process_pdf(filepath)
    return jsonify({"message": "File processed successfully."})

@app.route("/query", methods=["POST"])
def query_pdf():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    retrieved_titles, retrieved_sentences = search_faiss_index(query)
    context = f"Relevant Titles: {', '.join(retrieved_titles)}\nRelevant Sentences: {', '.join(retrieved_sentences)}"
    response = get_groq_response(query, context)
    print(response)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)