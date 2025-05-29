import pandas as pd
import random

def simulate_subjective_scores(csv_path="reports/output_results.csv"):
    df = pd.read_csv(csv_path)

    # Simülasyon: rasgele ama kontrollü skorlar atama (gerçek kullanımda elle girilebilir)
    def generate_score(text):
        # örnek olarak keyword içeriyorsa daha yüksek puan verelim
        keywords = ["redmi", "ryzen", "note"]
        score = 1
        for kw in keywords:
            if kw in text.lower():
                score += 1
        return min(score + random.randint(0, 2), 5)

    df['subjective_score'] = df['text'].apply(generate_score)

    avg_scores = df.groupby("model")['subjective_score'].mean().reset_index()
    avg_scores = avg_scores.sort_values(by="subjective_score", ascending=False)

    avg_scores.to_csv("reports/subjective_avg_scores.csv", index=False)
    print("🎯 Anlamsal skorlar raporu kaydedildi: reports/subjective_avg_scores.csv")

    return avg_scores
