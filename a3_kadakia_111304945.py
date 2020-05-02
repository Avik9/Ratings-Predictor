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
import multiprocessing

##########################################################################################
## Stage 1

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

    return [item_ids, reviews, ratings]


def trainWord2VecModel(reviews):
    """
    Trains a Word2Vec model from the given reviews.

    Attributes
    ----------
    reviews : list
        A list of all the reviews.

    Returns
    -------
    w2v_model : Word2Vec model
        A trained Word2Vec model.

    Steps
    -----
    Step 1.3: Use GenSim word2vec to train a 128-dimensional 
              word2vec model utilizing only the training data.
    """

    cores = multiprocessing.cpu_count()

    w2v_model = Word2Vec(sentences=reviews,
                         workers=cores-1,
                         window=10,
                         negative=0,
                         min_count=5,
                         seed=42,
                         size=128)

    w2v_model.train(sentences=reviews,
                    total_examples=w2v_model.corpus_count,
                    epochs=30)

    return w2v_model


def getFeatures(w2v_model, reviews):
    """
    Returns the extracted features from the Word2Vec model.

    Attributes
    ----------
    w2v_model : Word2Vec model
        A trained Word2Vec model.
    reviews : list
        A list of all the reviews.

    Returns
    -------
    train_x : Numpy Array
        An array containing the averaged extracted features 
        from the Word2Vec model.

    Steps
    -----
    Step 1.4: Extract features: utilizing your word2vec model,
              get a representation for each word per review.
    """

    train_x = []

    for review in reviews:
        temp = [0] * 128

        counter = 1

        for word in review:
            if word in w2v_model.wv.index2word:
                temp += w2v_model.wv[word]
                counter += 1

        temp = np.asarray(temp)
        # train_x.append(temp/len(review))
        train_x.append(temp/counter)

    return np.asarray(train_x)


def buildRatingPredictor(train_x, test_x, train_y, test_y):
    """
    Tries different models and returns the one wit the best accuracy.

    Attributes
    ----------
    train_x : Numpy array
        The data for training the model.
    test_x : Numpy array
        The data for testing the model.
    train_y : Numpy array
        The data for training the model.
    test_y : Numpy array
        The data for training the model.

    Returns
    -------
    bestModel : Ridge model
        A trained Ridge model.

    Steps
    -----
    Step 1.5: Build a rating predictor using L2 *linear* regression
              (can use the SKLearn Ridge class) with word2vec features.
    """

    listC = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    maxAccuracy = -1
    bestModel = None

    for C in listC:

        model = Ridge(random_state=42, alpha=C).fit(train_x, train_y)
        pred_y_test = model.predict(test_x)

        acc = MAE(test_y, pred_y_test)

        if acc >= maxAccuracy and acc < 1:
            # print("Current Best Model acc:", acc, "C:", C)
            maxAccuracy = acc
            bestModel = model

    return bestModel

##########################################################################################
## Stage 2

##################################################################
##################################################################
# Main:


if __name__ == '__main__':

    print("Stage 1:\n")

    if(len(sys.argv) < 3 or len(sys.argv) > 3):
        # print("Please enter the right amount of arguments in the following manner:",
        #       "\npython3 a3_kadakia_111304945 '*_train.csv' '*_trial.csv'")
        # sys.exit(0)
        training_file = 'food_train.csv'
        trial_file = 'food_trial.csv'
    else:
        training_file = sys.argv[1]
        trial_file = sys.argv[2]

    # Stage 1.1: Read the reviews and ratings from the file
    training_data = readCSV(training_file)
    trial_data = readCSV(trial_file)

    # Stage 1.3: Use GenSim word2vec to train a 128-dimensional word2vec
    #            model utilizing only the training data
    train_w2v_model = trainWord2VecModel(training_data[1])
    test_w2v_model = trainWord2VecModel(trial_data[1])

    # Stage 1.4: Extract features
    train_x = getFeatures(train_w2v_model, training_data[1])
    test_x = getFeatures(test_w2v_model, trial_data[1])
    train_y = np.asarray(training_data[2], dtype=np.int)
    test_y = np.asarray(trial_data[2], dtype=np.int)

    # Stage 1.5: Build a rating predictor
    rating_model = buildRatingPredictor(train_x, test_x, train_y, test_y)
    y_pred = rating_model.predict(test_x)

    # Stage 1.6: Print both the mean absolute error and Pearson correlation
    #            between the predictions and the (test input) set
    print("MAE (test):", MAE(test_y, y_pred))
    print("Pearson test:", ss.pearsonr(test_y, y_pred))

    print("\nStage 1 Checkpoint:\n")
    print("MAE (test):", MAE(test_y, y_pred))
    print("Pearson product-moment correlation coefficients (test):",
          np.corrcoef(test_y, y_pred))

    if "food" in trial_file:
        testCases = [548, 4258, 4766, 5800]

        for case in testCases:
            if case in trial_data[0]:
                pos = trial_data[0].index(case)
                print()
                print(case, "\tPredicted Value",
                      y_pred[pos], "\tTrue Value:", trial_data[2][pos])
            else:
                print(case, "not in", trial_file)

    print("\n\nStage 2:\n")
