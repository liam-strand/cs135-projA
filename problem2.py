import numpy as np
import pandas as pd
import os
from pprint import pprint
import sklearn.pipeline
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neural_network
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import expon, uniform
import string


def main():

    (x_amazon, x_imdb, x_yelp), (y_amazon, y_imdb, y_yelp) = get_split_train_data_from_dir("data_reviews")
    x_test_amazon, x_test_imdb, x_test_yelp = get_split_data_from_file("data_reviews/x_test.csv")

    clf_amazon = fit_model(x_amazon, y_amazon)
    clf_imdb = fit_model(x_imdb, y_imdb)
    clf_yelp = fit_model(x_yelp, y_yelp)


    yhat_test_amazon = clf_amazon.predict_proba(x_test_amazon)[:,1]
    yhat_test_imdb = clf_imdb.predict_proba(x_test_imdb)[:,1]
    yhat_test_yelp = clf_yelp.predict_proba(x_test_yelp)[:,1]

    with open("yproba1_test.txt", "w") as f:
        f.writelines([f"{str(v)}\n" for v in yhat_test_amazon])
        f.writelines([f"{str(v)}\n" for v in yhat_test_imdb])
        f.writelines([f"{str(v)}\n" for v in yhat_test_yelp])

def fit_model(x: np.ndarray, y: np.ndarray) -> sklearn.model_selection.GridSearchCV:
    pipeline = sklearn.pipeline.Pipeline([
        ("bow_feature_extractor", TfidfVectorizer(ngram_range=(1,2), strip_accents="ascii")),
        ("classifier", sklearn.neural_network.MLPClassifier(solver="lbfgs", max_iter=2000)),
    ])
    distributions = {
        "classifier__alpha": np.logspace(-5, 2, 8), 
        "classifier__hidden_layer_sizes": [(100), (50, 25), (50, 50), (100, 25), (100, 50), (100, 100)]
        # "bow_feature_extractor__min_df": range(0, 100, 10), 
        # "bow_feature_extractor__max_df": np.arange(0.8, 1.0, 0.01),
    }
    clf = sklearn.model_selection.GridSearchCV(pipeline, distributions, n_jobs=-1, verbose=3)
    clf.fit(x, y)

    pprint(clf.best_params_)

    return clf


def get_split_train_data_from_dir(dirname: str):
    x_amazon = list()
    x_imdb = list()
    x_yelp = list()

    y_data = pd.read_csv(f"{dirname}/y_train.csv")
    # also need to split y data
    y_amazon = list()
    y_imdb = list()
    y_yelp = list()

    for i, row in enumerate(pd.read_csv(f"{dirname}/x_train.csv").iterrows()):
        # split data into 3 lists, also get the ith value of y and put in correponding list
        if row[1][0] == "amazon":
            x_amazon.append(row[1][1])
            y_amazon.append(y_data.iloc[i][0])
        elif row[1][0] == "imdb":
            x_imdb.append(row[1][1])
            y_imdb.append(y_data.iloc[i][0])
        elif row[1][0] == "yelp":
            x_yelp.append(row[1][1])
            y_yelp.append(y_data.iloc[i][0])
    
    return ((np.array(x_amazon), np.array(x_imdb), np.array(x_yelp)), 
            (np.array(y_amazon), np.array(y_imdb), np.array(y_yelp)))

def get_split_data_from_file(filename: str):
    x_amazon = list()
    x_imdb = list()
    x_yelp = list()

    for row in pd.read_csv(filename).iterrows():
        # split data into 3 lists, also get the ith value of y and put in correponding list
        if row[1][0] == "amazon":
            x_amazon.append(row[1][1])
        elif row[1][0] == "imdb":
            x_imdb.append(row[1][1])
        elif row[1][0] == "yelp":
            x_yelp.append(row[1][1])
    
    return np.array(x_amazon), np.array(x_imdb), np.array(x_yelp)

def get_all_data_from_dir(dirname: str) -> tuple[list[str], list[str], list[str]]:
    x_train = load_data_from_file(os.path.join(dirname, "x_train.csv"), "text")
    y_train = load_data_from_file(os.path.join(dirname, "y_train.csv"), "is_positive_sentiment")
    x_test = load_data_from_file(os.path.join(dirname, "x_test.csv"), "text")
    return remove_punctuation(x_train), y_train, remove_punctuation(x_test)

def remove_punctuation(strings: list[str]) -> list[str]:
    return [s.translate(str.maketrans('', '', string.punctuation)) for s in strings]

def load_data_from_file(filename: str, col: str) -> list[str]:
    csv_data = pd.read_csv(filename)
    # pprint(csv_data)
    list_of_sentences = csv_data[col].values.tolist()
    return list_of_sentences

if __name__ == "__main__":
    main()
