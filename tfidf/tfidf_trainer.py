import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def train_and_save_tfidf(input_csv_path, output_pkl_path):
    df = pd.read_csv(input_csv_path)
    
    if 'text' not in df.columns:
        raise ValueError("CSV dosyasında 'text' adında bir sütun yok.")

    texts = df['text'].astype(str).tolist()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    with open(output_pkl_path, 'wb') as f:
        pickle.dump((vectorizer, tfidf_matrix, texts), f)

    print(f"Model kaydedildi: {output_pkl_path}")
