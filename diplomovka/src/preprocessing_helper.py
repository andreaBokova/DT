# Tento script obsahuje pomocne funkcie na predspracovane textu
import os
import re
import pandas as pd
import stanza
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
    tokenize_no_ssplit=True,  # vstup je jedna veta (nie je potrebne dalsie rozdelovanie, lebo DS Financial PhraseBank je uz rozdeleny na vety)
    verbose=False  # vypne vypis do konzoly
)

# Cache pre predspracovane data, aby sme nespustali Stanza pipeline dokola
CACHE_DIR = "data/financialphrasebank/processed"
os.makedirs(CACHE_DIR, exist_ok=True)

# ________________________________________________________
# 1) NACITANIE DATASETU FINANCIAL PHRASEBANK - priprava DF

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
    print(f"Loading {os.path.basename(path)} | rows={len(df)} | skipped={bad}")
    return df


# ____________________________________________________
# 2A) PREPROCESSING (cistenie, stopwords, lemmatizacia)


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
    out = df.copy()  # vytvorenie kopie datasetu

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
        print(f"Loading {dataset_key} from {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"Stats>{dataset_key}: after preprocessing (from cache) = {len(df)}")
        return df

    print(f"Not found in CACHE -> preprocessing {dataset_key} ...")

    df = load_phrasebank_txt(path)
    df = prepare_dataframe(df, min_tokens=3)
    df.to_csv(cache_path, index=False)

    print(f"Stats>{dataset_key}: after preprocessing = {len(df)}")
    print(f"Saved {dataset_key} to {cache_path}")
    return df

# ________________________
# 2B) OVERLAP HELPER - na odstranenie prekryvov

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
    print(f"Removed {before - after} overlapping rows from train (before={before} rows, after={after} rows)")
    return out

