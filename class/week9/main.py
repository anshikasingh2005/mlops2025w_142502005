from datasets import load_dataset
import pandas as pd
import re
from snorkel.labeling import labeling_function, LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ABSTAIN, NEG, POS = -1, 0, 1
positive_words ={"great","excellent","amazing","wonderful","best","fantastic"}
negative_words = {"bad","terrible","awful","worst","boring","poor"}

@labeling_function()
def lf_positive(x):
    return POS if any(w in x.text.split() for w in positive_words) else ABSTAIN
@labeling_function()
def lf_negative(x):
    return NEG if any(w in x.text.split() for w in negative_words) else ABSTAIN
@labeling_function()
def lf_exclaim(x):
    return POS if x.text.count("!") > 2 else ABSTAIN
def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^\w\s']", "", text)
    return text.lower()


def main():
    # Load 2000 training and 500 test examples for speed
    imdb = load_dataset("imdb")
    #train = pd.DataFrame(imdb["train"].select(range(2000)))
    #test = pd.DataFrame(imdb["test"].select(range(500)))
    

    imdb_train_df = pd.DataFrame(imdb["train"])
    imdb_test_df = pd.DataFrame(imdb["test"])

    train, _ = train_test_split(imdb_train_df, train_size=2000, stratify=imdb_train_df["label"], random_state=42)
    test, _ = train_test_split(imdb_test_df, train_size=500, stratify=imdb_test_df["label"], random_state=42)

    
    
    print("Train size:", len(train), "Test size:", len(test))
    print(train.head())
    train["text"] = train["text"].apply(clean_text)
    test["text"] = test["text"].apply(clean_text)
    lfs = [lf_positive, lf_negative, lf_exclaim]
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(train)
    LFAnalysis(L_train, lfs).lf_summary()
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)
    # Get probabilistic labels
    train_probs = label_model.predict_proba(L_train)
    train_preds = label_model.predict(L_train)
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5_000)
    X_train = vectorizer.fit_transform(train["text"])
    y_train = train_preds
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    X_test = vectorizer.transform(test["text"])
    y_test = test["label"]
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, target_names=["neg","pos","neutral"]))

    print("Unique classes in train labels:", train["label"].unique())
    print("Counts:", train["label"].value_counts())

    clf_fs = LogisticRegression(max_iter=200)
    clf_fs.fit(X_train, train["label"])
    fs_preds = clf_fs.predict(X_test)
    print("Fully supervised performance:")
    print(classification_report(y_test, fs_preds, target_names=["neg","pos"]))


if __name__ == "__main__":
    main()
