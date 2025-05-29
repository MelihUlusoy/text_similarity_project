import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def get_top_similar_tfidf(model_path, query_text, top_n=5):
    with open(model_path, 'rb') as f:
        vectorizer, tfidf_matrix, texts = pickle.load(f)

    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = [(texts[i], similarities[i]) for i in top_indices]
    
    return results
