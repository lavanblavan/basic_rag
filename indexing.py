import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from find_words import txt_find
from pdf2image import convert_from_path
import pickle  # Import pickle for saving the data
import numpy as np
def separate_titles_and_sentences(texts, boxes):
    """Separates titles and sentences based on width and punctuation."""
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


def create_embeddings(texts):
    """Create embeddings for the given texts."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models as well
    embeddings = model.encode(texts)
    return embeddings


def create_faiss_index(embeddings):
    """Create a FAISS index for fast similarity search."""
    # Convert the embeddings to a FAISS-compatible format (float32)
    embeddings = np.array(embeddings).astype('float32')
    
    # Initialize the FAISS index (L2 distance)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance is typically used
    index.add(embeddings)  # Add embeddings to the index
    
    return index


def save_faiss_index(index, filename="index.faiss"):
    """Save the FAISS index to a file."""
    faiss.write_index(index, filename)


def save_data_to_pkl(data, filename):
    """Save data to a .pkl file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")


def process_pdf_and_create_index(pdf_path):
    """Process the PDF, separate titles and sentences, create embeddings and FAISS index."""
    # Extract text and bounding boxes from the PDF
    texts, boxes, x_max, coord_txt = txt_find(convert_from_path(pdf_path))

    # Separate titles and sentences
    titles, sentences = separate_titles_and_sentences(texts, boxes)

    # Create embeddings for titles and sentences
    title_embeddings = create_embeddings(titles)
    sentence_embeddings = create_embeddings(sentences)

    # Create FAISS indices for titles and sentences
    title_index = create_faiss_index(title_embeddings)
    sentence_index = create_faiss_index(sentence_embeddings)

    # Save the FAISS indices to files
    save_faiss_index(title_index, "title_index.faiss")
    save_faiss_index(sentence_index, "sentence_index.faiss")

    # Save titles and sentences to .pkl files
    save_data_to_pkl(titles, "titles_data.pkl")
    save_data_to_pkl(sentences, "sentences_data.pkl")

    return titles, sentences


# Example usage
if __name__ == "__main__":
    pdf_path = '191025X-ASS2.pdf'

    # Process the PDF and create indices
    titles, sentences = process_pdf_and_create_index(pdf_path)

    print("Indexing complete. Titles and Sentences indexed.")
