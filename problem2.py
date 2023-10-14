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
    pprint(pd.read_csv(os.path.join("data_reviews", "x_train.csv")))
    x_amazon = list()
    x_imdb = list()
    x_yelp = list()
    for row in pd.read_csv(os.path.join("data_reviews", "x_train.csv")).iterrows():
        for col1, col2 in row:
            if col1 == "amazon":
                x_amazon.append(col2)
            elif col1 == "imdb":
                x_imdb.append(col2)
            else: # col1 == "yelp"
                x_yelp.append(col2)
            
    
    # , (y_amazon, y_imdb, y_yelp), (x_test_amazon, x_test_imdb, x_test_yelp) = get_all_data_from_dir("data_reviews")

    # x = np.array(x)
    # y = np.array(y, dtype=np.int32)
    # x_test = np.array(x_test)

    # pipeline = sklearn.pipeline.Pipeline([
    #     ("bow_feature_extractor", TfidfVectorizer(ngram_range=(1,1), stop_words="english", strip_accents="ascii")),
    #     ("classifier", sklearn.neural_network.MLPClassifier([100, 50, 25],solver="lbfgs", max_iter=2000)),
    # ])

    # distributions = {
    #     # "classifier__C": np.logspace(-5, 5, 20), 
    #     # "bow_feature_extractor__min_df": range(0, 100, 10), 
    #     # "bow_feature_extractor__max_df": np.arange(0.8, 1.0, 0.01),
    # }

    # clf = sklearn.model_selection.RandomizedSearchCV(pipeline, distributions, n_iter=100, verbose=2)

    # clf.fit(x, y)
    # yhat = clf.predict_proba(x)
    # score = sklearn.metrics.roc_auc_score(y, yhat[:,1])
    # print(f"score = {score}")

    # yhat_test = clf.predict_proba(x_test)[:,1]
    # with open("yproba1_test.txt", "w") as f:
    #     f.writelines([f"{str(v)}\n" for v in yhat_test])

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
