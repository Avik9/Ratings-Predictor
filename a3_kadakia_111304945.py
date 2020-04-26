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
        reviews.append(word_tokenize(line[5]) if type(line[5]) == str else "")

        # print("\nItem id:", line[0], type(line[0]), "Tokenized", item_id_tokenized)
        # print("Item rating:", line[1], type(line[1]), "Tokenized", rating_tokenized)
        # print("Item review:", line[5], type(line[5]), "Tokenized", review_tokenized)

    return [item_ids, reviews, ratings]

##################################################################
##################################################################
# Main:


if __name__ == '__main__':

    print("Welcome to the project")

    now = time.time()

    print("Stage 1.1: Read the reviews and ratings from the file.\n")

    if(len(sys.argv) < 3 or len(sys.argv) > 3):
        print("Please enter the right amount of arguments in the following manner:\npython3 a3_kadakia_111304945 '*_train.csv' '*_trial.csv'")
        sys.exit(0)

    # print("The following files will be opened:", sys.argv[1], sys.argv[2])

    training_data = readCSV(sys.argv[1])
    training_item_ids = training_data[0]
    training_reviews = training_data[1]
    training_ratings = training_data[2]
    
    trial_data = readCSV(sys.argv[2])
    trial_item_ids = trial_data[0]
    trial_reviews = trial_data[1]
    trial_ratings = trial_data[2]

    cores = multiprocessing.cpu_count()
    print("You have", cores, "cores")

    w2v_model = Word2Vec(min_count=2,
                     window=3,
                     size=128,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=0,
                     workers=cores-1)

    t = time.time()

    w2v_model.build_vocab(training_reviews, progress_per=1000)

    print('Time to build vocab: {} mins'.format((time.time() - t) / 60))

    t = time.time()

    w2v_model.train(training_reviews, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    print('Time to train the model: {} mins'.format((time.time() - t) / 60))



    print("Total time:", time.time() - now)
