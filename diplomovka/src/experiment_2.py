from preprocessing_helper import (
    load_or_preprocess,
    add_sentence_key,
    remove_overlap,
    LABEL_ORDER,
)

import os
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from xgboost import XGBClassifier

# ______________________
# SETUP

DATASETS = {
    "75": "data/financialphrasebank/raw/Sentences_75Agree.txt",
    "100": "data/financialphrasebank/raw/Sentences_AllAgree.txt",
}

RESULTS_DIR = "results/experiment2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ____________________
# ZOSTAVENIE TRENOVACEJ A TESTOVACEJ MNOZINY

def build_train_and_test():
    """
    Train: 75Agree + 80% AllAgree (AllAgree po vycleneni testu 20%)
    Test:  AllAgree20 (fixny split random_state=42)
    """
    df_75 = load_or_preprocess("75", DATASETS["75"])
    df_all = load_or_preprocess("100", DATASETS["100"])

    df_75 = add_sentence_key(df_75)
    df_all = add_sentence_key(df_all)

    # Fixny test set z 20% AllAgree
    all_rest, test_set_allagree_20 = train_test_split(
        df_all,
        test_size=0.20,
        random_state=42,
        stratify=df_all["label"],
    )

    # trenovaciu mnozinu zostavime z 75Agree a 80% AllAgree(all_rest)
    train_set = pd.concat([df_75, all_rest], ignore_index=True)

    # deduplikujeme vety podla kluca, aby sa nestalo, ze sa budu rovnake opakovat
    train_set = train_set.drop_duplicates(subset="__key").reset_index(drop=True)

    # odstranime overlap trenovacej a testovacej mnoziny
    test_keys = set(test_set_allagree_20["__key"].tolist())
    train_set = remove_overlap(train_set, test_keys)

    return train_set, test_set_allagree_20

# _______________________
# MODELY
# modelom nastavujeme grid (mnozina hyperparametrov, ktore chceme vyskusat)

def get_models():
    # na zaklade vysledkov Experimentu 1 pouzivame pre vsetky BoW
    vectorizer = CountVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
    )

    models = {
         "MultinomialNB": (
            MultinomialNB(),
            {
                # clf__alpha je Laplace smoothing parameter, aby sa zabranilo nulovym pravdepodobnostiam
                "clf__alpha": [0.1, 0.5, 1.0, 2.0],
                # NB nepodporujeclass_weight
                # clf__fit_prior = False predpoklada,ze vsetky triedy su rovnako pravdepodobne
                # clf__fit_prior = True - model respektuje skutocne rozdelenie dat v triedach
                # v nasom pripade pri clf__fit_prior = True ma trieda neutral vyssiu pravdepodobnost este predtym ako sa zohladnia slova
                "clf__fit_prior": [True, False],
            },
        ),
        "ComplementNB": (
            ComplementNB(),
            {
                "clf__alpha": [0.1, 0.5, 1.0, 2.0],
                "clf__fit_prior": [True, False],
                # ComplementNB pocita vahy slov pre kazdu triedu, niektore slova mozu mat prilis velke vahy(riziko preucenia)
                # clf__norm:True normalizuje vahy (casto lepsia generalizacia)
                "clf__norm":[ False, True],
                # 4×2×2=16 kombinaciix5 = 80 trenovani
            },
        ),
        "LinearSVC": (
            LinearSVC(max_iter=10000),
            {
                # GridSearchCV bude skusat vsetky kombinacie tychto hodnot
                # 0.1 znamena, ze silna regulacia, 5 znamena slabsia regulacia
                "clf__C": [0.1, 0.5, 1, 2, 5], 
                # clf__class_weight meni vahu chyby pocas ucenia
                # clf__class_weight=None znamena,ze vsetky triedy maju rovnaku vahu - chyba na netral ma rovnaky vyznam ako chyba na positive
                # clf__class_weight=None - model sa bude viac snazit minimalizovat chyby na vacsinovej triede
                # clf__class_weight=balanced znamena,ze vahy sa upravia podla velkosti tried
                # clf__class_weight=balanced - mensinova trieda ma vacsiu vahu - model viac tresta chyby na mensinovej triede
                "clf__class_weight": [None, "balanced"],
                # 5×2=10 kombinacii
                # 10×5=50 trenovani
            },
        ),
        "LogReg": (
            LogisticRegression(max_iter=2000),
            {
                "clf__C": [0.1, 0.5, 1, 2, 5],
                # algoritmus na optimalizaciu modelu,default je lbfgs-slaby, 
                # saga je casto lepsi pri BoW reprezentacii
                "clf__solver": ["lbfgs", "saga"], 
                "clf__class_weight": [None, "balanced"],
                # 5×2×2=20 kombinacii
                # 20×5=100 trenovani
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_jobs=-1,
                random_state=42,
            ),
            {
                "clf__n_estimators": [300, 600],
                # mac_depth None znamena,ze strom rastie, kym moze - riziko preucenia
                "clf__max_depth": [None, 30, 60],
                # kolko vzoriek v liste stromu - 1=strom je velmi specificky
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__class_weight": [None, "balanced"],
                # kolko [priznakov sa nahodne vyberie pri kazdom deleni stromu
                "clf__max_features": ["sqrt", "log2"],
                #2×3×3×2×2=72 komb.x5=360 trenovani
            },
        ),    
        "XGBoost": (
            XGBClassifier(
                #  model vracia pravdepodobnosti pre kazdu triedu
                objective="multi:softprob",
                # mame tri triedy
                num_class=3,
                # metoda, podla ktorej model hodnoti ucenie - multiclass log-loss
                eval_metric="mlogloss",
                # histogramova metoda trenovania stromov na CPU
                tree_method="hist",
                # pouziju sa vsetky dostupne jadra CPU 
                n_jobs=-1,
                # random state zabezpecuje reprodukovatelnost experimentu
                random_state=42,
            ),
            {
                # pocet stromov - viac stromov znamena silnejsi ale pomalsi model
                "clf__n_estimators": [300, 600, 900],
                # maximalna hlbka jedneho stromu (hlbsi model je komplexnejsi, ale riziko preucenia)
                "clf__max_depth": [3, 5, 7],
                # mensi leraning rate znamena pomalsie ucenie, ale casto lepsiu generalizaciu
                "clf__learning_rate": [0.05, 0.1],
                # kolko vzoriek sa nahodne pouzije pre kazdy strom
                # < 1.0 = menej preucenia
                "clf__subsample": [0.8, 1.0],
                # kolko priznakov (features) sa pouzije pre kazdy strom
                # < 1.0 = menej preucenia
                "clf__colsample_bytree": [0.8, 1.0],
                # minimalna vaha dat v liste, pravidlo na rast stromu
                # vyssia vaha - menej deleni
                "clf__min_child_weight": [1, 5],
                # L2 regularizacia vah v listoch
                # vyssia hodnota znamena silnejsiu regularizaciu - menej preucenia
                "clf__reg_lambda": [1.0, 3.0],
                #2x3x2x2x2x2x2= 192komb.x5= 960trenovani
            },
        ),        
        "AdaBoost": (
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                random_state=42,
            ),
            {
                "clf__n_estimators": [200, 400, 600],
                "clf__learning_rate": [0.5, 1.0],
            },
            # 3×2=6 kombinaciix5=30 trenovani

        )
    }

    return models, vectorizer

# ________________________
# Spustenie EXPERIMENTU 2

def run_experiment2():
    # priprava trenovacej a testovacej mnoziny
    train_df, test_df = build_train_and_test()
    # vrati slovnik modelov a BoW vectorizer
    models, vectorizer = get_models()

    # texty po preprocessingu 
    X_train = train_df["text_clean"]
    X_test = test_df["text_clean"]

    # triedy sentimentu
    y_train = train_df["label"]
    y_test = test_df["label"]

    # label encoding - LabelEncoder spravi mapovanie napr.:negative → 0,neutral → 1,positive → 2
    # fit_transform() si zapamata mapovanie na treningu
    # transform() ho potom pouzije aj na test
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # tu sa bude ukladat najlepsie CV macro F1, test macro F1, najlepsie hyperparametre
    all_summary_rows = []

    # hlavna slucka pre kazdy model - nazov, konkretna instancia, grid hyperparametrov
    for name, (clf, grid) in models.items():
        print("\n" + "=" * 70)
        print(f"MODEL: {name}")

        pipe = Pipeline([
            ("vec", vectorizer),
            ("clf", clf),
        ])

        # tu prebieha ladenie hyperparametrov
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,    # skusaju sa vsetky komb. hyperpar.
            scoring="f1_macro", # vyberame komb. podla macro F1
            cv=5, # 5-fold (5x trenovanie pre kazdu komb.) krizova validacia na treningovych datach
            n_jobs=-1,
            verbose=1, # vypis priebeh
        )

        #GridSearch urobi pre kazdu kombinaciu 5x trenovanie a vypocet macro F1 a vyberie najlepsiu kombinaciu
        search.fit(X_train, y_train_enc)

        best_model = search.best_estimator_ # najlepsia kombinacia parametrov
        best_params = search.best_params_ # ktore parametre vyhrali
        cv_best = float(search.best_score_) # najlepsi priemerny CV macro F1

        # finalne testovanie na AllAgree20
        y_pred_enc = best_model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

        print(f"Best CV macro-F1: {cv_best:.4f}")
        print(f"Best params: {best_params}")
        print(f"TEST macro-F1 (AllAgree20): {test_macro_f1:.4f}")

        # pre kazdu triedu bude mat report precision,recall, F1
        report = classification_report(y_test, y_pred, digits=4, labels=LABEL_ORDER)

        cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)

        out_path = os.path.join(RESULTS_DIR, f"{name}_best_report.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL: {name}\n")
            f.write(f"BEST CV macro-F1: {cv_best:.6f}\n")
            f.write(f"BEST PARAMS: {best_params}\n\n")
            f.write("Classification report (TEST)\n")
            f.write(report)
            f.write("\n\nConfusion matrix (rows=true, cols=pred)\n")
            f.write(str(cm))

        all_summary_rows.append({
            "model": name,
            "cv_best_macro_f1": cv_best,
            "test_macro_f1": test_macro_f1,
            "best_params": str(best_params),
        })

    summary_df = pd.DataFrame(all_summary_rows).sort_values("test_macro_f1", ascending=False)
    summary_path = os.path.join(RESULTS_DIR, "experiment2_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved summary to: {summary_path}")

if __name__ == "__main__":
    run_experiment2()
