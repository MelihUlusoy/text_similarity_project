import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import os

def train_word2vec_model(texts, sg, window, vector_size, save_path):
    tokenized = [word_tokenize(text) for text in texts]
    
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        sg=sg,
        workers=4,
        min_count=1
    )

    model.save(save_path)
    print(f"Model kaydedildi: {save_path}")

def train_all_models():
    settings = {
        "lemmatized": "data/lemmatized.csv",
        "stemmed": "data/stemmed.csv"
    }

    for name, path in settings.items():
        df = pd.read_csv(path)
        texts = df["text"].astype(str).tolist()

        for sg in [0, 1]:  # CBOW=0, SkipGram=1
            for window in [3, 5]:
                for vector_size in [50, 100]:
                    model_name = f"{name}_sg{sg}_win{window}_dim{vector_size}.model"
                    save_path = os.path.join("models/word2vec", model_name)
                    train_word2vec_model(texts, sg, window, vector_size, save_path)
