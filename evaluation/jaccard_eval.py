import pandas as pd
import itertools

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def generate_jaccard_matrix(input_csv="reports/output_results.csv", output_csv="reports/jaccard_matrix.csv"):
    df = pd.read_csv(input_csv)

    # Model -> Top-5 metinler (küme)
    model_results = {}
    for model in df['model'].unique():
        top5_texts = df[df['model'] == model].sort_values(by='score', ascending=False).head(5)['text']
        model_results[model] = set(top5_texts)

    models = list(model_results.keys())
    matrix = []

    for m1 in models:
        row = []
        for m2 in models:
            score = jaccard_similarity(model_results[m1], model_results[m2])
            row.append(round(score, 2))
        matrix.append(row)

    result_df = pd.DataFrame(matrix, index=models, columns=models)
    result_df.to_csv(output_csv)
    print("📊 Jaccard matrisi kaydedildi:", output_csv)
    return result_df
