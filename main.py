from tfidf.tfidf_trainer import train_and_save_tfidf
from similarity.word2vec_trainer import train_all_models
from similarity.tfidf_similarity import get_top_similar_tfidf
from similarity.word2vec_similarity import get_top_similar_word2vec
from evaluation.subjective_eval import simulate_subjective_scores
from evaluation.jaccard_eval import generate_jaccard_matrix
import os
import pandas as pd

if __name__ == "__main__":
    train_and_save_tfidf("data/lemmatized.csv", "tfidf/tfidf_lemmatized.pkl")
    train_and_save_tfidf("data/stemmed.csv", "tfidf/tfidf_stemmed.pkl")

    train_all_models()

    query = "Redmi Note 11 Ryzen 7"
    data_paths = {
        "lemmatized": "data/lemmatized.csv",
        "stemmed": "data/stemmed.csv"
    }

    results = []
    for fname in os.listdir("models/word2vec"):
        if fname.endswith(".model"):
            model_path = os.path.join("models/word2vec", fname)
            data_key = "lemmatized" if "lemmatized" in fname else "stemmed"
            data_path = data_paths[data_key]
            top5 = get_top_similar_word2vec(model_path, data_path, query)
            for rank, (text, score) in enumerate(top5, 1):
                results.append({"model": fname, "rank": rank, "text": text, "score": score})

    pd.DataFrame(results).to_csv("reports/output_results.csv", index=False)
    simulate_subjective_scores()
    generate_jaccard_matrix()
    print("✅ Tüm işlemler tamamlandı.")
