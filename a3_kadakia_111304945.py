import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.decomposition import PCA
import scipy.stats as ss
from gensim.models import Word2Vec
import torch
import sys
import pandas as pd
from nltk.tokenize import word_tokenize
import time
import multiprocessing


def readCSV(fileName):
    """
    Reads in all the files and stores them for future use.

    Attributes
    ----------
    fileName : str
        The name of the csv file to open.

    Returns
    -------
    ratings : dict
        A list containing all the reviews and ratings.

    Steps
    -----
    Step 1.1: Read the reviews and ratings from the file.
    Step 1.2: Tokenize the file. You may now use any existing tokenizer.
    """

    dataFile = pd.read_csv(fileName, sep=',')
    item_ids = []
    ratings = []
    reviews = []

    dataFile["reviewText"] = dataFile["reviewText"].replace("", np.nan)
    dataFile = dataFile.dropna(subset=["reviewText"])

    for _, line in dataFile.iterrows():

        item_ids.append(int(line[0]))
        ratings.append(int(line[1]))
        reviews.append(word_tokenize(
            line[5].lower()) if type(line[5]) == str else "")

        # print("\nItem id:", line[0], type(line[0]), "Tokenized", item_id_tokenized)
        # print("Item rating:", line[1], type(line[1]), "Tokenized", rating_tokenized)
        # print("Item review:", line[5], type(line[5]), "Tokenized", review_tokenized)

    return [item_ids, reviews, ratings]


def trainWord2VecModel(reviews):
    cores = multiprocessing.cpu_count()

    w2v_model = Word2Vec(min_count=3,
                         window=3,
                         size=128,
                         alpha=0.03,
                         seed=42,
                         iter=10,
                         workers=cores-1)

    t = time.time()

    w2v_model.build_vocab(reviews, progress_per=1000)

    print('Time to build vocab: {} mins'.format((time.time() - t) / 60))

    t = time.time()

    w2v_model.train(reviews, total_examples=w2v_model.corpus_count,
                    epochs=30, report_delay=0)

    print('Time to train the model: {} mins'.format((time.time() - t) / 60))

    return w2v_model


def getFeatures(w2v_model, reviews):
    ridge_train_x = []

    for review in reviews:
        temp = [0] * 128

        for word in review:
            if word in w2v_model.wv.index2word:
                temp += w2v_model.wv[word]

        temp = np.asarray(temp)
        ridge_train_x.append(temp/len(review))

    return np.asarray(ridge_train_x)


def buildRatingPredictor(train_x, test_x, train_y, test_y):

    listC = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    maxAccuracy = -1
    bestModel = None

    for C in listC:

        model = Ridge(random_state=42, alpha=C).fit(train_x, train_y)
        pred_y_test = model.predict(test_x)

        acc = MAE(test_y, pred_y_test)

        if acc >= maxAccuracy and acc < 1:
            print("Current Best Model acc:", acc, "C:", C)
            maxAccuracy = acc
            bestModel = model

    return bestModel

##################################################################
##################################################################
# Main:


if __name__ == '__main__':

    print("Welcome to the project")

    now = time.time()

    print("Stage 1.1: Read the reviews and ratings from the file.\n")

    if(len(sys.argv) < 3 or len(sys.argv) > 3):
        print("Please enter the right amount of arguments in the following manner:\npython3 a3_kadakia_111304945 '*_train.csv' '*_trial.csv'")
        # sys.exit(0)
        training_file = 'food_train.csv'
        trial_file = 'food_trial.csv'
    else:
        training_file = sys.argv[1]
        trial_file = sys.argv[2]

    training_data = readCSV(training_file)

    trial_data = readCSV(trial_file)

    train_w2v_model = trainWord2VecModel(training_data[1])
    test_w2v_model = trainWord2VecModel(trial_data[1])

    train_x = getFeatures(train_w2v_model, training_data[1])
    test_x = getFeatures(test_w2v_model, trial_data[1])

    train_y = np.asarray(training_data[2], dtype=np.int)
    test_y = np.asarray(trial_data[2], dtype=np.int)

    rating_model = buildRatingPredictor(train_x, test_x, train_y, test_y)
    pred_y_test = rating_model.predict(test_x)
    pred_y_train = rating_model.predict(train_x)

    print(len(train_w2v_model.wv.vocab))

    print()
    print("Accuracy for the Ridge model train:", MAE(train_y, pred_y_train))
    print("Accuracy for the Ridge model test:", MAE(test_y, pred_y_test))

    print()
    print("Accuracy for the Pearson train:",
          ss.pearsonr(train_y, pred_y_train))
    print("Accuracy for the Pearson test:", ss.pearsonr(test_y, pred_y_test))
    print()
    print("Total time:", time.time() - now)
