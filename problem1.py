import numpy as np
import pandas as pd
import os
from pprint import pprint
import sklearn.pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import expon, uniform
from scipy.sparse import spmatrix
import string
import matplotlib.pyplot as plt

def main():
    x, y, x_test = get_all_data_from_dir("data_reviews")
    x = np.array(x)
    y = np.array(y, dtype=np.int32)
    x_test = np.array(x_test)

    # bow_preprocessor = CountVectorizer(ngram_range=(1,1), strip_accents="ascii")
    # res = bow_preprocessor.fit_transform(x)
    # sums = np.sum(res.toarray(), axis=0)
    # sorted_indices = np.argsort(-sums)

    # sorted_sums = sums[sorted_indices]
    # sorted_wrds = bow_preprocessor.get_feature_names_out()[sorted_indices]
    # x_coords = np.arange(0, len(sorted_sums))
    # print(sorted_wrds)
    # print(sorted_sums)

    # preprocessed_x = np.char.lower(x)

    # for row in preprocessed_x:
    #     if np.char.find(row, "the") == -1:
    #         print(row)

    # for term, count in list(sorted(bow_preprocessor.vocabulary_.items())):
    #     print("%4d %s" % (count, term))

    pipeline = sklearn.pipeline.Pipeline([
        ("bow_feature_extractor", CountVectorizer(ngram_range=(1,1), stop_words="english", strip_accents="ascii")),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])

    distributions = {
        "classifier__C": np.logspace(-2, 2, 5), 
        # "bow_feature_extractor__min_df": np.linspace(0, 10, 5), 
        "bow_feature_extractor__max_df": np.linspace(0.05, 0.8, 21)
    }

    clf = GridSearchCV(pipeline, distributions, verbose=1, n_jobs=-1, cv=5, scoring=lambda e, x, y: roc_auc_score(y, e.predict(x)))

    clf.fit(x, y)
    # yhat = clf.predict_proba(x)
    # score = sklearn.metrics.roc_auc_score(y, yhat[:,1])
    # print(f"score = {score}")

    pprint(clf.best_params_)
    pprint(clf.best_score_)
    pprint(clf.cv_results_.keys())

    yhat_test = clf.predict_proba(x_test)[:,1]
    with open("yproba1_test.txt", "w") as f:
        f.writelines([f"{str(v)}\n" for v in yhat_test])

def get_all_data_from_dir(dirname: str) -> tuple[list[str], list[str], list[str]]:
    x_train = load_data_from_file(os.path.join(dirname, "x_train.csv"), "text")
    y_train = load_data_from_file(os.path.join(dirname, "y_train.csv"), "is_positive_sentiment")
    x_test = load_data_from_file(os.path.join(dirname, "x_test.csv"), "text")
    return remove_punctuation(x_train), y_train, remove_punctuation(x_test)

def remove_punctuation(strings: list[str]) -> list[str]:
    s1 = [s.translate(str.maketrans('', '', string.punctuation)) for s in strings]
    s2 = [s.lower() for s in s1]
    return s2

def load_data_from_file(filename: str, col: str) -> list[str]:
    csv_data = pd.read_csv(filename)
    # pprint(csv_data)
    list_of_sentences = csv_data[col].values.tolist()
    return list_of_sentences

if __name__ == "__main__":
    main()
