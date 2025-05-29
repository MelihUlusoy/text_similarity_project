from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def get_average_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0).reshape(1, -1)

def get_top_similar_word2vec(model_path, data_path, query_text, top_n=5):
    model = Word2Vec.load(model_path)
    df = pd.read_csv(data_path)
    texts = df["text"].astype(str).tolist()

    query_tokens = word_tokenize(query_text.lower())
    query_vector = get_average_vector(query_tokens, model)

    if query_vector is None:
        print("Giriş metnindeki hiçbir kelime modelde bulunamadı.")
        return []

    similarities = []
    for text in texts:
        tokens = word_tokenize(text)
        avg_vector = get_average_vector(tokens, model)
        if avg_vector is not None:
            score = cosine_similarity(query_vector, avg_vector)[0][0]
            similarities.append(score)
        else:
            similarities.append(-1)

    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = [(texts[i], similarities[i]) for i in top_indices if similarities[i] >= 0]
    return results
