import os
import requests
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load FAISS index
def load_faiss_index(filename):
    return faiss.read_index(filename)

# Load titles and sentences from pickle files
def load_titles_from_index(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_sentences_from_index(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Search FAISS index
def search_faiss_index(query_embedding, index, k=5):
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), k)
    return distances, indices

# Create embeddings
def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return embeddings

# Generate response using Groq
def get_groq_response(query, context, api_key):
    client = Groq(api_key=api_key)

    full_prompt = f"""
    You are a helpful assistant. Check the provided question and use the given context to generate an accurate answer.

    Context:
    {context}

    Question:
    {query}

    Correct Answer:
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained for accurate responses."},
            {"role": "user", "content": full_prompt},
        ],
    )

    return response.choices[0].message.content.strip()

# Retrieve context and use it for response
def get_retrieved_response(query, title_index, sentence_index, titles, sentences, api_key):
    query_embedding = create_embeddings([query])[0]

    _, title_indices = search_faiss_index(query_embedding, title_index)
    _, sentence_indices = search_faiss_index(query_embedding, sentence_index)

    retrieved_titles = " ".join([titles[i] for i in title_indices[0]]) if len(title_indices[0]) > 0 else "No relevant titles found."
    retrieved_sentences = " ".join([sentences[i] for i in sentence_indices[0]]) if len(sentence_indices[0]) > 0 else "No relevant sentences found."

    retrieved_context = f"Relevant Titles: {retrieved_titles}\nRelevant Sentences: {retrieved_sentences}"

    return get_groq_response(query, retrieved_context, api_key)

# Example usage
if __name__ == "__main__":
    # Get API key securely from environment variables
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("API key not found! Set the GROQ_API_KEY environment variable.")

    # Load FAISS indices and data
    title_index = load_faiss_index("title_index.faiss")
    sentence_index = load_faiss_index("sentence_index.faiss")

    titles = load_titles_from_index("titles_data.pkl")
    sentences = load_sentences_from_index("sentences_data.pkl")

    # Query example
    query = "what does experimental setup consist?"
    response = get_retrieved_response(query, title_index, sentence_index, titles, sentences, api_key)

    print("Generated Answer:", response)
