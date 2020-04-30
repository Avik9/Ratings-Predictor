import numpy as np
from sklearn.linear_model import Ridge
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

    for _, line in dataFile.iterrows():

        item_ids.append(int(line[0]))
        ratings.append(word_tokenize(str(line[1])))
        reviews.append(word_tokenize(line[5].lower()) if type(line[5]) == str else "")

        # print("\nItem id:", line[0], type(line[0]), "Tokenized", item_id_tokenized)
        # print("Item rating:", line[1], type(line[1]), "Tokenized", rating_tokenized)
        # print("Item review:", line[5], type(line[5]), "Tokenized", review_tokenized)

    return [item_ids, reviews, ratings]

def trainWord2VecModel(reviews):
    cores = multiprocessing.cpu_count()
    # print("You have", cores, "cores")

    w2v_model = Word2Vec(min_count=1,
                     window=10,
                     size=128,
                     sample=6e-5, 
                     alpha=0.03,  
                     negative=0,
                     workers=cores-1)

    t = time.time()

    w2v_model.build_vocab(reviews, progress_per=1000)

    print('Time to build vocab: {} mins'.format((time.time() - t) / 60))

    t = time.time()

    w2v_model.train(reviews, total_examples=w2v_model.corpus_count, epochs=30, report_delay=0)

    print('Time to train the model: {} mins'.format((time.time() - t) / 60))

    return w2v_model

def getFeatures(w2v_model, reviews):
    ridge_train_x = np.array([])

    for review in reviews:
        temp = np.array([])

        for word in review:
            vector = w2v_model.wv[word]
            print(type(vector))
            print(vector)

            # if len(temp) == 0:
            #     temp = vector
            # else:
            temp = np.append(temp, np.array(vector), axis=0)
        
            print("Temp:", temp, "\n")
            print("Temp average:", np.mean(temp, axis=0, dtype=np.float64), "\n")
            print("Temp shape:", temp.shape, "\n")
        print("Final Temp average:", np.mean(temp, axis=0, dtype=np.float64), "\n")

        ridge_train_x = np.append(ridge_train_x, np.array([np.mean(temp, axis=0, dtype=np.float64)]))

    return ridge_train_x

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

    # print("The following files will be opened:", sys.argv[1], sys.argv[2])

    test_array = np.array([[1, 2], [2, 4], [3, 6]])
    print("Axis 0:", np.mean(test_array, axis=0, dtype=np.float64))
  
    training_data = readCSV(training_file)
    # training_item_ids = training_data[0]
    # training_reviews = training_data[1]
    # training_ratings = training_data[2]
    
    # trial_data = readCSV(trial_file)
    # trial_item_ids = trial_data[0]
    # trial_reviews = trial_data[1]
    # trial_ratings = trial_data[2]

    train_w2v_model = trainWord2VecModel(training_data[1])

    train_x = getFeatures(train_w2v_model, training_data[1])
    train_y = np.array(training_data[2])


    # print(w2v_model.wv['apple'])

    # word_vectors = w2v_model.wv.vectors
    # print(type(word_vectors))

    # print(len(training_ratings))
    # print(len(word_vectors))
    # print(word_vectors)
    # print(len(w2v_model.wv.vocab.keys()))
    # print(w2v_model.wv.vocab.keys())

    # for ls in word_vectors:
    #     print(ls, ":", word_vectors[ls])

    print("Total time:", time.time() - now)