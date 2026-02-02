import os
import re
import joblib
import pandas as pd
import nltk
import stanza

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

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ============================================================
# SETUP - staci spustit raz
# ============================================================
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# stanza.download("en")

ALLOWED = {"positive", "negative", "neutral"}
LABEL_ORDER = ["negative", "neutral", "positive"]  # fixne poradie pre confusion matrix
STOP_WORDS = set(stopwords.words("english"))

# Stanza pipeline na spracovanie textu, inicializuje sa raz pri starte programu
# text prechadza cez tokenizaciu, 
# POS tagging 
# a lematizaciu (lematizacia v Stanza vyuziva informaciu o slovnom druhu)
NLP = stanza.Pipeline(
    "en",
    processors="tokenize,pos,lemma",
    tokenize_no_ssplit=True, # vstup je jedna veta (nie je potrebne dalsie rozdelovanie, lebo DS Financial PhraseBank je uz rozdeleny na vety)
    verbose=False # vypne vypis do konzoly
)

# Cesty k datasetom
DATASETS = {
    "50": "data/financialphrasebank/raw/Sentences_50Agree.txt",
    "66": "data/financialphrasebank/raw/Sentences_66Agree.txt",
    "75": "data/financialphrasebank/raw/Sentences_75Agree.txt",
    "100": "data/financialphrasebank/raw/Sentences_AllAgree.txt",
}

# Cache pre predspracovane data, aby sme nespustali Stanza pipeline dokola
CACHE_DIR = "data/financialphrasebank/processed"
os.makedirs(CACHE_DIR, exist_ok=True)

RESULTS_DIR = "results/experiment1"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1) NACITANIE DATASETU FINANCIAL PHRASEBANK - priprava DF
# ============================================================

def load_phrasebank_txt(path: str) -> pd.DataFrame:
    """
    Nacita .txt, kde kazdy riadok ma formu sentence @label
    Vrati dataframe so stlpcami sentence, label
    """
    rows, bad = [], 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "@" not in line:
                bad += 1
                continue

            # split podla @ na konci
            sentence, label = line.rsplit("@", 1)
            sentence = sentence.strip()
            label = label.strip().lower()

            if label not in ALLOWED or not sentence:
                bad += 1
                continue

            rows.append({"sentence": sentence, "label": label})

    df = pd.DataFrame(rows)
    print(f"[LOAD] {os.path.basename(path)} | rows={len(df)} | skipped={bad}")
    return df


# ============================================================
# 2) PREPROCESSING (cistenie, stopwords, lemmatizacia)
# ============================================================

def basic_clean(text: str) -> str:
    """Zakladne cistenie textu: lower, odstranenie URL, spec. znakov, cisiel, extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"&[a-z]+;", " ", text)              # HTML escape
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # URL
    text = re.sub(r"[^a-z\s]", " ", text)              # iba pismena a medzery
    text = re.sub(r"\s+", " ", text).strip()           # extra whitespaces
    return text

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def lemmatize(tokens):
    """
    Lematizacia cez Stanza, najpomalsia cast preprocessing pipeline.
    """
    if not tokens:
        return []
    doc = NLP(" ".join(tokens))
    lemmas = []
    for sent in doc.sentences:
        for w in sent.words:
            if w.lemma:
                lemmas.append(w.lemma)
    return lemmas

def preprocess_sentence(text: str) -> str:
    """Kompletny preprocessing jednej vety"""
    text = basic_clean(text)
    if not text:
        return ""
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)

def prepare_dataframe(df: pd.DataFrame, min_tokens: int = 3) -> pd.DataFrame:
    """
    - odstrani prazdne/kratke/duplicitne vety
    - vytvori finalny text (text_clean), ktory sa pouzije na vektorizaciu (BoW, TF-IDF).
    """
    out = df.copy() # vytvorenie kopie datasetu

    # Rychly filter na zaklade basic_clean (bez lematizacie), zistime ci veta stoji za spracovanie
    # pouzije nami definovanu funkciu basic_clean na kazdy riadok v stlpci sentence
    # out["sentence"] je stlpec s povodnymi vetami (raw text)
    out["__tmp"] = out["sentence"].astype(str).map(basic_clean)
    out = out[out["__tmp"].str.len() > 0]

    # odstrania sa velmi kratke vety
    out["__tok_cnt"] = out["__tmp"].map(lambda x: len(x.split()))
    out = out[out["__tok_cnt"] >= min_tokens]

    # odtrania sa duplicitne vety
    out = out.drop_duplicates(subset="__tmp").reset_index(drop=True)

    # najpomalsia cast pipeline, robi sa len na datach (text_clean), ktore presli filtrami
    # robi sa tu preprocessing - cistenie, tokenizacia, odstranenie stop slov, lematizacia (Stanza)
    out["text_clean"] = out["sentence"].map(preprocess_sentence)
    # niektore vety mozu po odstraneni stop slov ostat prazdne, preto ich odstranime
    out = out[out["text_clean"].str.len() > 0].reset_index(drop=True)

    # Upratanie pomocných stlpcov __tmp a __tok_cnt,
    # ostanu len povodne stplce - sentence (povodna veta), label (sentiment) a text_clean (vstup pre ML model)
    out = out.drop(columns=["__tmp", "__tok_cnt"])
    return out

def load_or_preprocess(dataset_key: str, path: str) -> pd.DataFrame:
    """
    Cache mechanizmus:
    - ak existuje CSV v cache, nacitame
    - inak spravime preprocessing a ulozime
    """
    cache_path = os.path.join(CACHE_DIR, f"phrasebank_{dataset_key}_processed.csv")

    if os.path.exists(cache_path):
        print(f"[CACHE] Loading {dataset_key} from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"[CACHE] Not found -> preprocessing {dataset_key} (can take minutes)...")
    df = load_phrasebank_txt(path)
    df = prepare_dataframe(df, min_tokens=3)
    df.to_csv(cache_path, index=False)
    print(f"[CACHE] Saved {dataset_key} to {cache_path}")
    return df

# ============================================================
# 2b) DEDUP/OVERLAP HELPER
# ============================================================

def add_sentence_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vytvori stabilny kluc vety pre porovnavanie overlapu medzi train/test.
    Pouzijeme basic_clean, aby sme ignorovali:
    - casing
    - interpunkciu
    - extra medzery
    """
    out = df.copy()
    out["__key"] = out["sentence"].astype(str).map(basic_clean)
    return out

def remove_overlap(train_df: pd.DataFrame, test_keys: set) -> pd.DataFrame:
    """
    Vyhodi z train_df vsetky vety, ktore sa nachadzaju v teste (podla __key).
    Tym zabezpecime, ze test obsahuje len vety, ktore model nevidel pri treningu.
    """
    before = len(train_df)
    out = train_df[~train_df["__key"].isin(test_keys)].copy().reset_index(drop=True)
    after = len(out)
    print(f"[OVERLAP] Removed {before - after} overlapping rows from train (before={before} rows, after={after} rows)")
    return out


# ============================================================
# 3) UNDERSAMPLING (iba trening)
# ============================================================

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


# ============================================================
# 4) VECTORIZERY (BoW vs TF-IDF)
# ============================================================

def make_vectorizer(repr_name: str):
    """
    repr_name:
      - "bow"  -> CountVectorizer
      - "tfidf"-> TfidfVectorizer
    """
    if repr_name == "bow":
        # Bag-of-Words - frekvencie slov
        return CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=5
        )
    elif repr_name == "tfidf":
        # TF-IDF - vahy slov podla informativnosti
        return TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=5
        )
    else:
        raise ValueError("repr_name must be 'bow' or 'tfidf'")


# ============================================================
# 5) MODEL (Logistic Regression)
# ============================================================

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
        max_iter=2000, #Default je 100, co nestaci - ak by bolo malo iteracii, model nemusi konvergovat a sklearn vypise chybu
        solver="lbfgs" 
    )

# ============================================================
# 6) JEDEN BEH EXPERIMENTU (train_variant - 50Agree/60Agree/75Agree/Combined, vyvazeny - ano/nie, reprezentacia - BoW/TF-IDF)
# ============================================================

def run_single_setting(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    balance_train: bool,
    repr_name: str,
    setting_name: str
) -> dict:
    """
    Spusti jeden konkretny setting:
    - train_df: predspracovany train dataset (napr. 50Agree)
    - test_df: predspracovany test dataset (AllAgree 20% split)
    - balance_train: True/False
    - repr_name: 'bow' alebo 'tfidf'
    - setting_name: string pre ukladanie

    Vrati slovnik s metrikami, ktore ulozime do summary tabulky.
    """

    # 1) Priprav treningove data
    if balance_train:
        train_used = undersample_dataframe(train_df, "label")
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

    print(f"[DONE] {setting_name} | macro_f1={macro_f1:.4f} | repr={repr_name} | balanced={balance_train}")
    return result


# ============================================================
# 7) MAIN - spustenie Experimentu 1
# ============================================================
 
def main():
    # 1) Nacitanie suborov + preprocessing 
    # Treningove varianty:
    train_50 = load_or_preprocess("50", DATASETS["50"])
    train_66 = load_or_preprocess("66", DATASETS["66"])
    train_75 = load_or_preprocess("75", DATASETS["75"])

    # AllAgree pouzijeme ako zdroj pre testovaci set
    allagree_100 = load_or_preprocess("100", DATASETS["100"])

    # 1b) Pridame kluc vety (__key) pre odstranenie overlapu train vs test
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

        for repr_name in repr_variants:
            for balance in balance_variants:
                setting_name = f"{train_name}_{repr_name}_{'bal' if balance else 'unbal'}_testAllAgree20"
                res = run_single_setting(
                    train_df=train_no_overlap,
                    test_df=test_allagree_20,
                    balance_train=balance,
                    repr_name=repr_name,
                    setting_name=setting_name
                )
                all_results.append(res)

    # 5) Kombinovany dataset 50+66+75 (shuffle)
    # Spojime datasety
    combined = pd.concat([train_50, train_66, train_75], ignore_index=True)\
    
    # deduplikacia medzi datasetmi - rovnaka veta sa moze vyskytovat vo viacerych verziach datasetov (50/66/75)
    # deduplikujeme podľa __key, aby veta nedostala väčšiu váhu len preto, že bola vo viacerých súboroch
    combined = combined.drop_duplicates(subset="__key").reset_index(drop=True)

    # Shuffle
    # frac=1 znamena, ze vezme vsetky riadky ale v nahodnom poradi
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # odstranime overlap combined a test
    combined_no_overlap = remove_overlap(combined, test_keys)

    for repr_name in repr_variants:
        for balance in balance_variants:
            setting_name = f"trainCOMBINED_{repr_name}_{'bal' if balance else 'unbal'}_testAllAgree20"
            res = run_single_setting(
                train_df=combined_no_overlap,
                test_df=test_allagree_20,
                balance_train=balance,
                repr_name=repr_name,
                setting_name=setting_name
            )
            all_results.append(res)

    # 6) Ulozenie summary tabulky do CSV 
    results_df = pd.DataFrame(all_results)
    summary_path = os.path.join(RESULTS_DIR, "experiment1_summary.csv")
    results_df.to_csv(summary_path, index=False)

    # 7) Vypis top vysledkov (podla macro F1)
    print("\n=== TOP RESULTS (sorted by macro_f1) ===")
    print(results_df.sort_values("macro_f1", ascending=False).head(10)[
        ["setting", "macro_f1", "weighted_f1", "repr", "balanced_train"]
    ])

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved per-setting reports to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
