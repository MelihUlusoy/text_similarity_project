import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    return text

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens, language='english'):
    stops = set(stopwords.words(language))
    return [word for word in tokens if word not in stops]

def stem_tokens(tokens):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(word) for word in tokens]

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_and_save():
    df = pd.read_csv("data/urun_adlari_large_dataset.csv")

    lemmatized_texts = []
    stemmed_texts = []

    for title in df['product_title']:
        cleaned = clean_text(title)
        tokens = tokenize(cleaned)
        tokens_no_stop = remove_stopwords(tokens)

        lemmatized = lemmatize_tokens(tokens_no_stop)
        stemmed = stem_tokens(tokens_no_stop)

        lemmatized_texts.append(" ".join(lemmatized))
        stemmed_texts.append(" ".join(stemmed))

    pd.DataFrame({"text": lemmatized_texts}).to_csv("data/lemmatized.csv", index=False)
    pd.DataFrame({"text": stemmed_texts}).to_csv("data/stemmed.csv", index=False)

    print("Veriler başarıyla işlendi ve kaydedildi.")
