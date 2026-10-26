import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

sns.set_style("whitegrid")


URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text_basic(s: str) -> str:
    s = str(s).lower()
    return s

def clean_text_medium(s: str) -> str:
    s = str(s).lower()
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    s = re.sub(r"[^a-z0-9\s!?']", " ", s) 
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def run_analysis():
    DATA_PATH = "archive/go_emotions_dataset.csv"
    df = pd.read_csv(DATA_PATH)

    emotion_cols = [c for c in df.columns if c not in ["id", "text", "example_very_unclear"]]

    print("DATA INFO")
    print("Number of samples:", len(df))
    print("Number of emotion labels:", len(emotion_cols))

    df["text_len"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    df["num_labels"] = df[emotion_cols].sum(axis=1).astype(int)

    print("\nText length statistics:")
    print(df["text_len"].describe())


    q1, q3 = df["text_len"].quantile([0.25, 0.75])
    iqr = q3 - q1
    long_thr = int(q3 + 1.5 * iqr)

    label_thr = int(df["num_labels"].quantile(0.99))  
    print(f"\nOutlier thresholds:")
    print(f" - Long text threshold (IQR rule): >= {long_thr} words")
    print(f" - Extreme many-labels threshold (99th pct): >= {label_thr} labels")

    print("\nExamples: 5 longest texts (showing length + first 120 chars):")
    for _, r in df.nlargest(5, "text_len")[["text_len", "text"]].iterrows():
        print(f"  len={int(r['text_len'])} | {str(r['text'])[:120]}...")

    print("\nExamples: 5 with most labels (showing #labels + first 120 chars):")
    for _, r in df.nlargest(5, "num_labels")[["num_labels", "text"]].iterrows():
        print(f"  labels={int(r['num_labels'])} | {str(r['text'])[:120]}...")




    plt.figure(figsize=(6,4))
    sns.histplot(df["text_len"], bins=40, kde=True)
    plt.title("1. Distribution of text lengths (words)")
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6,4))
    sns.boxplot(x=df["text_len"])
    plt.title("2. Boxplot of text lengths (outliers)")
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6,4))
    sns.histplot(df["num_labels"], bins=15)
    plt.title("3. Distribution of number of labels per example")
    plt.tight_layout()
    plt.show()


    label_counts = df[emotion_cols].sum().sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=label_counts.head(15).index, y=label_counts.head(15).values)
    plt.xticks(rotation=45, ha="right")
    plt.title("4. Top 15 most frequent emotions")
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10,5))
    sns.barplot(x=label_counts.tail(15).index, y=label_counts.tail(15).values)
    plt.xticks(rotation=45, ha="right")
    plt.title("5. Least 15 frequent emotions")
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(12,10))
    sns.heatmap(df[emotion_cols].corr(), cmap="coolwarm", center=0)
    plt.title("6. Correlation heatmap between emotions")
    plt.tight_layout()
    plt.show()

   
    plt.figure(figsize=(6,4))
    sns.scatterplot(x="text_len", y="num_labels", data=df, alpha=0.3)
    plt.title("7. Text length vs number of labels")
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8,4))
    sns.histplot(label_counts.values, bins=30)
    plt.title("8. Distribution of label frequencies")
    plt.tight_layout()
    plt.show()


  
    plt.figure(figsize=(7,4))
    tmp = df.groupby("num_labels")["text_len"].mean().reset_index()
    sns.lineplot(data=tmp, x="num_labels", y="text_len", marker="o")
    plt.title("10. Average text length vs number of labels")
    plt.tight_layout()
    plt.show()

  

    X_raw = df["text"].astype(str)
    y = df[emotion_cols].values


    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42
    )


    cleaners = [
        ("basic", clean_text_basic),
        ("medium", clean_text_medium)
    ]


    vectorizers = []
    for vec_name, vec_class in [("Count", CountVectorizer), ("TF-IDF", TfidfVectorizer)]:
        for ngram in [(1,1), (1,2), (1,3)]:
            for stop in [None, "english"]:
                v = vec_class(
                    ngram_range=ngram,
                    stop_words=stop,
                    min_df=3,
                    max_df=0.9,
                    max_features=20000
                )
                vectorizers.append((f"{vec_name} {ngram} stop={'yes' if stop=='english' else 'no'}", v))


    nb_alphas = [0.1, 0.5, 1.0]

    results = []

    def eval_pred(y_true, y_pred):
        return (
            f1_score(y_true, y_pred, average="micro"),
            f1_score(y_true, y_pred, average="macro"),
        )

    for clean_name, clean_fn in cleaners:
        total_jobs = len(cleaners) * len(vectorizers)
        job = 0
        X_tr = X_train.apply(clean_fn)
        X_va = X_val.apply(clean_fn)
        X_te = X_test.apply(clean_fn)

        for vec_name, vec in vectorizers:
            job += 1
            print(f"\nProgress {job}/{total_jobs} | cleaning={clean_name} | vectorizer={vec_name}")
            X_tr_vec = vec.fit_transform(X_tr)
            X_va_vec = vec.transform(X_va)
            X_te_vec = vec.transform(X_te)


            best_micro = -1
            best_cfg = None
            best_model = None
            best_macro = None

            for a in nb_alphas:
                model = OneVsRestClassifier(MultinomialNB(alpha=a))
                model.fit(X_tr_vec, y_train)
                y_val_pred = model.predict(X_va_vec)
                micro, macro = eval_pred(y_val, y_val_pred)
                if micro > best_micro:
                    best_micro = micro
                    best_macro = macro
                    best_cfg = f"NB alpha={a}"
                    best_model = model


            y_test_pred = best_model.predict(X_te_vec)
            test_micro, test_macro = eval_pred(y_test, y_test_pred)
            results.append((clean_name, vec_name, best_cfg, best_micro, best_macro, test_micro, test_macro))

    results_df = pd.DataFrame(
        results,
        columns=["Cleaning", "Vectorizer", "Model", "Val Micro F1", "Val Macro F1", "Test Micro F1", "Test Macro F1"]
    )

    results_df = results_df.sort_values(by="Val Micro F1", ascending=False)

    print("\nTOP 20 sorted by Val Micro F1")
    print(results_df.head(20).to_string(index=False))


    top = results_df.head(15).copy()
    top["Name"] = top["Cleaning"] + " | " + top["Vectorizer"] + " | " + top["Model"]


    plt.figure(figsize=(12,6))
    sns.barplot(x="Name", y="Val Micro F1", data=top)
    plt.xticks(rotation=90)
    plt.title("11. Top 15 Validation Micro F1 comparison")
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(12,6))
    sns.barplot(x="Name", y="Test Micro F1", data=top)
    plt.xticks(rotation=90)
    plt.title("12. Test Micro F1 for top-15 (selected by validation)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_analysis()
