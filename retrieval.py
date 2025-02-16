import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# Load titles and sentences from pickle files
def load_titles_from_index(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_sentences_from_index(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Function to generate response from OpenLlama via API
def get_openllama_response(query, api_key):
    """Generate a response from OpenLLaMA using the query and API key."""
    url = "https://api.openllama.com/generate"  # Update the URL with the correct API endpoint
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Create the payload with the query
    payload = {
        "input": query,
        "parameters": {
            "max_length": 200,  # Adjust response length if needed
            "num_return_sequences": 1
        }
    }

    # Send request to OpenLlama API
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data['generated_text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Function to create embeddings for text using SentenceTransformer
def create_embeddings(texts):
    """Create embeddings for the given texts."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models as well
    embeddings = model.encode(texts)
    return embeddings

# Load FAISS index for titles and sentences
def load_faiss_index(filename="index.faiss"):
    """Load the FAISS index from a file."""
    return faiss.read_index(filename)

# Search the FAISS index to find the top-k most similar items
def search_faiss_index(query_embedding, index, k=5):
    """Search the FAISS index to find the top-k most similar items."""
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    return distances, indices

# Retrieve context based on the query (utilizing both OpenLlama response and FAISS)
def get_retrieved_context(query, title_index, sentence_index, titles, sentences, api_key):
    """Retrieve the most relevant titles and sentences from the FAISS index based on a query."""
    # Generate OpenLlama response for the query
    openllama_response = get_openllama_response(query, api_key)
    
    # Create embeddings for the query
    query_embedding = create_embeddings([query])[0]

    # Search the FAISS index for the most similar titles and sentences
    title_distances, title_indices = search_faiss_index(query_embedding, title_index)
    sentence_distances, sentence_indices = search_faiss_index(query_embedding, sentence_index)

    # Retrieve the context from FAISS indices (titles and sentences)
    retrieved_titles = " ".join([titles[i] for i in title_indices[0]]) if len(title_indices[0]) > 0 else "No relevant titles found."
    retrieved_sentences = " ".join([sentences[i] for i in sentence_indices[0]]) if len(sentence_indices[0]) > 0 else "No relevant sentences found."

    # Combine OpenLlama response and FAISS results into final context
    context = f"OpenLlama Response: {openllama_response}\n\nSimilar Titles: {retrieved_titles}\n\nSimilar Sentences: {retrieved_sentences}"

    return context

# Example usage of the functions (for testing)
if __name__ == "__main__":
    # Your OpenLlama API key
    api_key = "LA-18da46f05ef74a44b06b21c33b0d270fbe54cbc6946942be8164c1f5e2f31020"  # Replace with your OpenLlama API key
    
    # Load the FAISS indices for titles and sentences
    title_index = load_faiss_index("title_index.faiss")
    sentence_index = load_faiss_index("sentence_index.faiss")

    # Load the titles and sentences from the .pkl files
    titles = load_titles_from_index("titles_data.pkl")
    sentences = load_sentences_from_index("sentences_data.pkl")

    # Example query (you can replace this with any input query)
    query = "What is incirinated"

    # Retrieve context based on the query
    context = get_retrieved_context(query, title_index, sentence_index, titles, sentences, api_key)

    # Print the retrieved context (reply from OpenLLaMA and FAISS results)
    print("Retrieved Context:", context)
