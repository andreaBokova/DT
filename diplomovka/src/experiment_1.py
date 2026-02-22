from preprocessing_helper import (
    load_or_preprocess,
    load_phrasebank_txt,
    add_sentence_key,
    remove_overlap,
    LABEL_ORDER,
)

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Vektorizacia - BoW, TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Model - Logistic Regression
from sklearn.linear_model import LogisticRegression

# Metriky
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from nltk.tokenize import word_tokenize

# __________________________
# SETUP
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# stanza.download("en")


# Cesty k datasetom
DATASETS = {
    "50": "data/financialphrasebank/raw/Sentences_50Agree.txt",
    "66": "data/financialphrasebank/raw/Sentences_66Agree.txt",
    "75": "data/financialphrasebank/raw/Sentences_75Agree.txt",
    "100": "data/financialphrasebank/raw/Sentences_AllAgree.txt",
}


RESULTS_DIR = "results/experiment1"
os.makedirs(RESULTS_DIR, exist_ok=True)

# _________________
# FUNKCIE NA VYPOCET A ULOZENIE STATISTIK O DATASETOCH DO TABULKY
# Tabulka ma informacie o datasetoch pocas roznych stadii 
# obsahuje stlpce: stage, note, total, count_negative, count_neutral, count_positive, prop_negative, prop_neutral, prop_positive
# bude ulozena ako dataset_stats_experiment1.csv

STATS_DIR = os.path.join(RESULTS_DIR, "stats")
os.makedirs(STATS_DIR, exist_ok=True)

def label_stats(df: pd.DataFrame, label_col: str = "label") -> dict:
    """
    Pomocna funkcia - vrati pocet a podiel tried v DF
    """
    counts = df[label_col].value_counts()
    total = int(counts.sum())
    props = (counts / total) if total > 0 else counts
    out = {"total": total}
    for lab in LABEL_ORDER:
        out[f"count_{lab}"] = int(counts.get(lab, 0))
        out[f"prop_{lab}"] = float(props.get(lab, 0.0))
    return out

def print_label_stats(title: str, df: pd.DataFrame):
    """
    Funkcia na vypis rozlozenia tried (counts aj percenta)
    """
    s = label_stats(df)
    print(f"\nStats>{title}")
    print(f"  total: {s['total']}")
    for lab in LABEL_ORDER:
        pct = s[f"prop_{lab}"] * 100
        print(f"  {lab:8s}: {s[f'count_{lab}']:5d}  ({pct:5.1f}%)")

def append_dataset_stats(rows: list, dataset_name: str, stage: str, df: pd.DataFrame, note: str = ""):
    """
    Ulozi jeden riadok do tabulky statistik 
    stage hodnota moze byt: raw, processed, train_no_overlap, train_balanced, test
    """
    s = label_stats(df)
    rows.append({
        "dataset": dataset_name,
        "stage": stage,
        "note": note,
        "total": s["total"],
        "count_negative": s["count_negative"],
        "count_neutral": s["count_neutral"],
        "count_positive": s["count_positive"],
        "prop_negative": s["prop_negative"],
        "prop_neutral": s["prop_neutral"],
        "prop_positive": s["prop_positive"],
    })

# ________________________________
# UNDERSAMPLING (iba trening)

def undersample_dataframe(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Vyvazenie tried undersamplingom - kazdu triedu zredukujeme na velkost najmensej triedy
    """
    groups = [g for _, g in df.groupby(label_col)]
    min_n = min(len(g) for g in groups)

    sampled = [
        resample(g, replace=False, n_samples=min_n, random_state=42)
        for g in groups
    ]
    out = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
    return out


# _______________________________
# VECTORIZERY (BoW vs TF-IDF)

def make_vectorizer(repr_name: str):
    """
    repr_name:
      - "bow"  -> CountVectorizer
      - "tfidf"-> TfidfVectorizer
    """
    if repr_name == "bow":
        # Bag-of-Words
        return CountVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2
        )
    elif repr_name == "tfidf":
        # TF-IDF
        return TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2
        )
    else:
        raise ValueError("repr_name must be bow or tfidf")


# ___________________________________
# MODEL (Logistic Regression)

def make_model():
    """
    - funkcia vzdy vytvori novu instanciu modelu,
    co zabezpeci, ze vsetky experimenty pouzivaju rovnake nastavenie
    - pouzivame Multiclass logistic regression, pre triedy negative, neutral, positive
    - logisticka regresia je velmi vhodna pre BoW a TF-ID
    - solver='lbfgs' je algoritmus, ktorym sa optimalizuje vahovy vektor modelu a minimalizuje log-loss
    a podporuje multi_class="multinomial"
    - multi_class="multinomial" mdel optimalizuje vsetky 3 triedy naraz
    - max_iter zvysime, aby model určite konvergoval, default 100 nestaci.
    - Ak by bolo malo iteracii, model nemusi konvergovat a sklearn vypise chybu
    - Pri použití solvera L-BFGS je multinomiálna logistická regresia v scikit-learn použitá by default
    """
    return LogisticRegression(
        max_iter=2000,  # Default je 100, co nestaci - ak by bolo malo iteracii, model nemusi konvergovat a sklearn vypise chybu
        solver="lbfgs"
    )

# ______________________________________________________________________
# JEDEN BEH EXPERIMENTU (train_variant - 50Agree/60Agree/75Agree/Combined, vyvazeny - ano/nie, reprezentacia - BoW/TF-IDF)

def run_single_setting(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    balance_train: bool,
    repr_name: str,
    setting_name: str,
    dataset_stats_rows: list = None,
    dataset_name_for_stats: str = "" 
) -> dict:
    """
    Spusti jeden konkretny setting:
    - train_df: predspracovany train dataset (napr. 50Agree)
    - test_df: predspracovany test dataset (AllAgree 20%)
    - balance_train: True/False
    - repr_name: 'bow' alebo 'tfidf'
    - setting_name: string pre ukladanie
    - dataset_stats_rows: aby sme mohli do tabulky statistik pridat stage "train_balanced"
    - dataset_name_for_stats: string pre nazov datasetu do tabulky statistik

    Vrati slovnik s metrikami, ktore ulozime do summary tabulky.
    """

    # 1) Priprav treningove data
    print_label_stats(f"{setting_name} | TRAIN (before balancing)", train_df)

    if balance_train:
        train_used = undersample_dataframe(train_df, "label")
    
        print_label_stats(f"{setting_name} | TRAIN (after undersampling)", train_used)
        print(f"Stats>{setting_name} | rows before={len(train_df)} -> after={len(train_used)} (undersampling removed {len(train_df)-len(train_used)})")

        # Uloz riadok do tabulky pre stage train_balanced
        if dataset_stats_rows is not None and dataset_name_for_stats:
            append_dataset_stats(
                dataset_stats_rows,
                dataset_name_for_stats,
                "train_balanced",
                train_used,
                note=f"undersampling | setting={setting_name}"
            )
    else:
        train_used = train_df

    # 2) Vektorizacia
    vectorizer = make_vectorizer(repr_name)
    X_train = vectorizer.fit_transform(train_used["text_clean"])
    X_test = vectorizer.transform(test_df["text_clean"])

    y_train = train_used["label"]
    y_test = test_df["label"]

    # 3) Model
    model = make_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 4) Metriky
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # detail pre kazdu triedu
    p, r, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=LABEL_ORDER, zero_division=0
    )

    # 5) Reporty a confusion matrix
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)

    # ulozenie reportu
    report_path = os.path.join(RESULTS_DIR, f"{setting_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(str(cm))

    # 6) Return do summary tabulky
    result = {
        "setting": setting_name,
        "repr": repr_name,
        "balanced_train": balance_train,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    # pridame metriky pre kazdu triedu (positive, negative, neutral)
    for i, label in enumerate(LABEL_ORDER):
        result[f"precision_{label}"] = p[i]
        result[f"recall_{label}"] = r[i]
        result[f"f1_{label}"] = f1[i]
        result[f"support_{label}"] = int(support[i])

    print(f"DONE {setting_name} | macro_f1={macro_f1:.4f} | repr={repr_name} | balanced={balance_train}")
    return result


# ____________________________________
# 7) MAIN - spustenie Experimentu 1

def main():

    # 1A) Nacitanie suborov + preprocessing
    # Treningove varianty:
    train_50 = load_or_preprocess("50", DATASETS["50"])
    train_66 = load_or_preprocess("66", DATASETS["66"])
    train_75 = load_or_preprocess("75", DATASETS["75"])

    # Nacitanie raw datasetov len na ucely pocitania viet
    raw_50 = load_phrasebank_txt(DATASETS["50"])
    raw_66 = load_phrasebank_txt(DATASETS["66"])
    raw_75 = load_phrasebank_txt(DATASETS["75"])
    raw_100 = load_phrasebank_txt(DATASETS["100"])


    dataset_stats_rows = []
    append_dataset_stats(dataset_stats_rows, "50Agree", "raw", raw_50, note="raw txt count")
    append_dataset_stats(dataset_stats_rows, "66Agree", "raw", raw_66, note="raw txt count")
    append_dataset_stats(dataset_stats_rows, "75Agree", "raw", raw_75, note="raw txt count")
    append_dataset_stats(dataset_stats_rows, "AllAgree", "raw", raw_100, note="raw txt count")

    append_dataset_stats(dataset_stats_rows, "50Agree", "processed", train_50, note="after preprocessing")
    append_dataset_stats(dataset_stats_rows, "66Agree", "processed", train_66, note="after preprocessing")
    append_dataset_stats(dataset_stats_rows, "75Agree", "processed", train_75, note="after preprocessing")

    # AllAgree pouzijeme ako zdroj pre testovaci set
    allagree_100 = load_or_preprocess("100", DATASETS["100"])
    append_dataset_stats(dataset_stats_rows, "AllAgree", "processed", allagree_100, note="after preprocessing")

    # 1B) Pridame kluc vety (__key) pre odstranenie overlapu train vs test
    train_50 = add_sentence_key(train_50)
    train_66 = add_sentence_key(train_66)
    train_75 = add_sentence_key(train_75)
    allagree_100 = add_sentence_key(allagree_100)

    # 2) Vytvorime testovaci set z AllAgree, ale zoberieme len 20%
    #    - Stratify zachova pomery tried v teste
    allagree_rest, test_allagree_20 = train_test_split(
        allagree_100,
        test_size=0.20,
        random_state=42,   # pri kazdom spusteni programu sa vyberu rovnake (nahodne) vety
        stratify=allagree_100["label"]
    )

    print("\n[TEST] AllAgree split:")
    print("AllAgree total:", len(allagree_100))
    print("Test (20%):", len(test_allagree_20))
    print("Test label counts:\n", test_allagree_20["label"].value_counts())

    # test stats
    append_dataset_stats(dataset_stats_rows, "AllAgree20", "test", test_allagree_20, note="fixed test (20% of AllAgree)")
    print_label_stats("TEST (AllAgree 20%)", test_allagree_20)

    # Test kluce - vety, ktore nesmu byt v trenovacej mnozine
    test_keys = set(test_allagree_20["__key"].tolist())

    # 3) Definujeme experimentalne konfigurácie
    train_variants = {
        "train50": train_50,
        "train66": train_66,
        "train75": train_75
    }

    repr_variants = ["bow", "tfidf"]
    balance_variants = [False, True]

    all_results = []

    # 4) 50/66/75 × (bow/tfidf) × (balanced/unbalanced)
    for train_name, train_df in train_variants.items():
        # vyhod overlap s testom
        train_no_overlap = remove_overlap(train_df, test_keys)

        if train_name == "train50":
            base_ds_name = "50Agree"
        elif train_name == "train66":
            base_ds_name = "66Agree"
        elif train_name == "train75":
            base_ds_name = "75Agree"
        else:
            base_ds_name = train_name

        # Uloz riadok do tabulky pre stage train_no_overlap
        append_dataset_stats(dataset_stats_rows, base_ds_name, "train_no_overlap", train_no_overlap, note="after removing overlap with test")
        print_label_stats(f"{train_name} | TRAIN after overlap removal", train_no_overlap)

        for repr_name in repr_variants:
            for balance in balance_variants:
                setting_name = f"{train_name}_{repr_name}_{'bal' if balance else 'unbal'}_testAllAgree20"
                res = run_single_setting(
                    train_df=train_no_overlap,
                    test_df=test_allagree_20,
                    balance_train=balance,
                    repr_name=repr_name,
                    setting_name=setting_name,
                    # posielame dataset_stats_rows, nech mozeme zalogovat stats pre kombinaciu nastaveni
                    dataset_stats_rows=dataset_stats_rows,
                    dataset_name_for_stats=base_ds_name
                )

                all_results.append(res)

    # 5) Kombinovany dataset 50+66+75 (shuffle)
    # Spojime datasety
    # ignore index vytvori novy cisty index
    combined = pd.concat([train_50, train_66, train_75], ignore_index=True) 

    # deduplikacia medzi datasetmi - rovnaka veta sa moze vyskytovat vo viacerych verziach datasetov (50/66/75)
    # deduplikujeme podľa __key, aby veta nedostala väčšiu váhu len preto, že bola vo viacerých súboroch
    combined = combined.drop_duplicates(subset="__key").reset_index(drop=True)

    # Shuffle
    # frac=1 znamena, ze vezme vsetky riadky ale v nahodnom poradi
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # odstranime overlap combined a test
    combined_no_overlap = remove_overlap(combined, test_keys)

    # Uloz riadok do tabulky pre stage train_no_overlap pre Combined DF
    append_dataset_stats(dataset_stats_rows, "Combined", "train_no_overlap", combined_no_overlap, note="50+66+75 dedup+shuffle, after removing overlap with test")
    print_label_stats("Combined | TRAIN after overlap removal", combined_no_overlap)

    for repr_name in repr_variants:
        for balance in balance_variants:
            setting_name = f"trainCOMBINED_{repr_name}_{'bal' if balance else 'unbal'}_testAllAgree20"
            res = run_single_setting(
                train_df=combined_no_overlap,
                test_df=test_allagree_20,
                balance_train=balance,
                repr_name=repr_name,
                setting_name=setting_name,
                dataset_stats_rows=dataset_stats_rows,
                dataset_name_for_stats="Combined"
            )
            all_results.append(res)

    # 6) Ulozenie summary tabulky do CSV
    results_df = pd.DataFrame(all_results)
    summary_path = os.path.join(RESULTS_DIR, "experiment1_summary.csv")
    results_df.to_csv(summary_path, index=False)

    # Uloz statisticke udaje o datasetoch 
    stats_df = pd.DataFrame(dataset_stats_rows)

    # Z tabulky statistik odstranime duplicity - kedze "train_balanced" sa zapisuje pre kazdy beh experimentu, mozu vznikat duplicitne riadky v tabulke
    stats_df = stats_df.drop_duplicates(subset=[
        "dataset", "stage", "total",
        "count_negative", "count_neutral", "count_positive"
    ]).reset_index(drop=True)

    stats_csv_path = os.path.join(STATS_DIR, "dataset_stats_experiment1.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"\nStats>Saved dataset stats to: {stats_csv_path}")

    # 7) Vypis top vysledkov (podla macro F1)
    print("\nTOP RESULTS (sorted by macro_f1)")
    print("\n_______________________________")
    print(results_df.sort_values("macro_f1", ascending=False).head(10)[
        ["setting", "macro_f1", "weighted_f1", "repr", "balanced_train"]
    ])

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved per-setting reports to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
