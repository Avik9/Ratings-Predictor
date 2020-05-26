# Avik Kadakia
# akadakia
# 111304945
# Kaggle shared task username: Avik Kadakia

import csv
import multiprocessing
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import scipy.stats as ss
from gensim.models import Word2Vec
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import pandas as pd
import nltk
nltk.download('punkt')

##########################################################################################
# Stage 1


def readCSV(fileName, test=False):
    """
    Reads in the files and stores them for future use.

    Attributes
    ----------
    fileName : str
        The name of the csv file to open.

    Returns
    -------
    ratings : list
        A list containing all the reviews and ratings.
    test: boolean
        The boolean stating if the data is for testing.

    Steps
    -----
    Step 1.1: Read the reviews and ratings from the file.
    Step 1.2: Tokenize the file. You may now use any existing tokenizer.
    """

    dataFile = pd.read_csv(fileName, sep=',')
    item_ids = []
    ratings = []
    reviews = []
    user_ids = []
    
    for _, line in dataFile.iterrows():

        if test:
            item_ids.append(int(line[0]))
            user_ids.append(line[4])

            # Tokenizes the reviews
            reviews.append(word_tokenize(
                line[5].lower()) if type(line[5]) == str else "")
        else:
            item_ids.append(int(line[0]))
            ratings.append(int(line[1]))
            user_ids.append(line[4])
            reviews.append(word_tokenize(
                line[5].lower()) if type(line[5]) == str else "")

    return [item_ids, reviews, ratings, user_ids]


def trainWord2VecModel(reviews, cores=2, min_count=3):
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

    w2v_model = Word2Vec(sentences=reviews,
                         workers=cores-1,
                         window=5,
                         alpha=0.03,
                         negative=0,
                         min_count=min_count,
                         seed=42,
                         size=128)

    w2v_model.train(sentences=reviews,
                    total_examples=w2v_model.corpus_count,
                    epochs=60)

    w2v_model.init_sims()

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

        for word in review:
            if word in w2v_model.wv.index2word:
                temp += w2v_model.wv[word]

        temp = np.asarray(temp)

        if(len(review) > 0):
            train_x.append(temp/len(review))
        else:
            train_x.append(temp)

    return np.asarray(train_x)


def buildRatingPredictor(train_x, train_y):
    """
    Tries different models and returns the one wit the best accuracy.

    Attributes
    ----------
    train_x : Numpy array
        The data for training the model.
    train_y : Numpy array
        The data for training the model.

    Returns
    -------
    bestAlpha : int
        The best alpha value.

    Steps
    -----
    Step 1.5: Build a rating predictor using L2 *linear* regression
              (can use the SKLearn Ridge class) with word2vec features.
    """

    listAlpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    minAccuracy = 1
    bestModel = None
    best_pearsonr = 0.35

    X_train, X_test, Y_train, Y_test = train_test_split(
        train_x, train_y, test_size=0.20, random_state=42)

    for alpha in listAlpha:

        model = Ridge(random_state=42, alpha=alpha).fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        acc = MAE(Y_test, Y_pred)
        pearsonr = ss.pearsonr(Y_test, Y_pred)

        if acc < 1 and acc < minAccuracy:

            minAccuracy = acc
            bestModel = model
            best_pearsonr = pearsonr[0]
            bestAlpha = alpha

    return bestAlpha

##########################################################################################
# Stage 2


def getUserBackground(files):
    """
    Gets the user background for each user.

    Attributes
    ----------
    files : list
        A list containing the training file path and trial file path.

    Returns
    -------
    users : list
        A list containing all the reviews and ratings grouped by users.

    Steps
    -----
    Step 2.1: Grab the user_ids for both datasets.
    """

    ratings = []
    reviews = []
    item_ids = []
    user_ids = []

    for file in files:
        dataFile = pd.read_csv(file, sep=',')

        for _, line in dataFile.iterrows():

            user = line[4]

            if user in user_ids:
                position = user_ids.index(user)

                item_ids[position].append(int(line[0]))  # Item ids
                ratings[position].append(int(line[1]))  # ratings
                reviews[position].append(word_tokenize(line[5].lower())
                                         if type(line[5]) == str else "")  # Reviews

            else:
                user_ids.append(user)
                position = user_ids.index(user)

                item_ids.append([int(line[0])])  # Item ids
                ratings.append([int(line[1])])  # ratings
                reviews.append([word_tokenize(line[5].lower())
                                if type(line[5]) == str else ""])  # Reviews

    return [user_ids, item_ids, ratings, reviews]


def getUserLangRepresentation(model, users):
    """
    Returns the user features from the Word2Vec model and the user's reviews.

    Attributes
    ----------
    w2v_model : Word2Vec model
        A trained Word2Vec model.

    users : list
        A list of all the users.

    Returns
    -------
    user_lang_rep : Numpy Array
        An array containing the averaged extracted features 
        for each user.

    Steps
    -----
    Step 2.2: For each user, treat their training data as "background" 
              (i.e. the data from which to learn the user factors) in 
              order to learn user factors: average all of their word2vec 
              features over the training data to treat as 128-dimensional 
              "user-language representations".
    """

    user_lang_rep = []

    for reviews in users[3]:
        temp_features = getFeatures(model, reviews)
        temp_array = [0] * 128

        for row in temp_features:
            temp_array += row

        user_lang_rep.append(temp_array/len(temp_features))

    return user_lang_rep


def runPCAMatrix(user_reviews):
    """
    Converts the user review's 128 dimensional features to 3 dimensional 
    transformation matrix.

    Attributes
    ----------
    user_reviews : list
        Reviews per user.

    Returns
    -------
    v_matrix : Numpy Array
        An array containing the 3 dimensional transformation matrix.

    Steps
    -----
    Step 2.3: Run PCA the matrix of user-language representations to 
              reduce down to just three factors. Save the 3 dimensional 
              transformation matrix (V) so that you may apply it to new 
              data (i.e. the trial or test set when predicting -- when 
              predicting you should not run PCA again; only before 
              training).
    """

    return (PCA(n_components=3, random_state=42).fit(user_reviews)).transform(user_reviews)


def PCA_feature_vector(X_train, X_test, review_training_data, review_trial_data,
                       review_testing_data, user_data, file, v_matrix):
    """
    Converts the user review's 128 dimensional features to 3 dimensional 
    transformation matrix.

    Attributes
    ----------
    X_train : list
        The data for training the model.

    X_test : list
        The data for testing the model.

    review_training_data : list
        A list containing all the reviews and ratings for training.

    review_trial_data : list
        A list containing all the reviews and ratings for trial.

    review_testing_data : list
        A list containing all the reviews and ratings for testing.

    user_data : list
        Reviews per user.

    file : str
        File name.

    v_matrix : list
        An array containing the 3 dimensional transformation matrix.

    Returns
    -------
    feature_vector : Numpy Array
        An array containing the user-factor adaptated review embedding.

    Steps
    -----
    Step 2.4: Use the first three factors from PCA as user factors 
              in order to run user-factor adaptation, otherwise 
              using the same approach as stage 1.
    """

    feature_vector = []

    dataFile = pd.read_csv(file, sep=',')
    
    for _, line in dataFile.iterrows():

        temp_feature_vector = []
        item_id = line[0]
        user_id = line[4]

        if item_id in review_training_data[0]:
            position = review_training_data[0].index(item_id)
            review_embedding = X_train[position]

        elif item_id in review_trial_data[0]:
            position = review_trial_data[0].index(item_id)
            review_embedding = X_test[position]

        elif review_testing_data != None and item_id in review_testing_data[0]:
            position = review_testing_data[0].index(item_id)
            review_embedding = X_test[position]

        if user_id in user_data[0]:
            position = user_data[0].index(user_id)
            user_factor = v_matrix[position]

        else:
            user_factor = [1] * 3

        for vector in user_factor:
            temp_feature_vector.append(review_embedding * vector)

        temp_feature_vector.append(review_embedding)
        flattened_array = np.ndarray.flatten(np.array(temp_feature_vector))
        feature_vector.append(flattened_array)

    feature_vector = np.array(feature_vector)
    return feature_vector

##########################################################################################
# Stage 3


def build_dataloader(embeddings, ratings, bs, shfle, workers):
    """
    Converts the user review's 128 dimensional features to 3 dimensional 
    transformation matrix.

    Attributes
    ----------
    embeddings : list
        The embeddings for the model.

    ratings : list
        The ratings for the model.

    bs : int
        Batch size.

    shfle : boolean
        If the data should be shuffled.

    workers : list
        Number of workers to use for training.

    Returns
    -------
    dataset : DataLoader
        A DataLoader containing the torch.tensor of the embeddings and reviews passed in.

    Steps
    -----
    Step 3.1: Start with word2vec embeddings *per word* -- these may already be in memory 
              from stage 1.
    """

    embeddings = list(embeddings)
    embeddings = [list(item) for item in embeddings]

    dataset = TensorDataset(torch.tensor(embeddings), torch.tensor(ratings))

    return DataLoader(dataset, batch_size=bs, shuffle=shfle, num_workers=workers)


class LSTM_RNN(nn.Module):
    """
    An LTSM RNN to predict better results. DOES NOT WORK. Just tried to implement it.

    Steps
    -----
    Step 3.1: Start with word2vec embeddings *per word* -- these may already be in memory 
              from stage 1.
    Step 3.2: Input the word2vec embeddings as the input sequence.
    Step 3.3: Add a linear regression output layer on top of your RNN or TSN to output a 
              continuous valued ratings score.
    Step 3.4: Add the user factors from stage 2 to your deep learning module.

    """

    embedding_dim = 128
    hidden_dim = 10
    num_epochs = 30

    def __init__(self, train_w2v_model):
        super(LSTM_RNN, self).__init__()

        self.embed = torch.nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = len(train_w2v_model.wv.vocab)

        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim)
        self.linear = torch.nn.Linear(self.hidden_dim, self.vocab_size) # Stage 3.3
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(list(self.lstm.parameters())
                                          + list(self.linear.parameters()), lr=0.001)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, encrypted):
        encrypted = encrypted.unsqueeze(1)
        lstm_out, lstm_hidden = self.lstm(encrypted, self.init_hidden())
        scores = self.linear(lstm_out)
        scores = scores.transpose(1, 2)
        predictions = self.softmax(scores, dim=2)
        _, batch_out = predictions.max(dim=2)
        batch_out = batch_out.squeeze(1)
        return scores, batch_out


def train(model, training_data, optimizer, loss_func, num_epochs=15):
    model.train()

    for i in range(num_epochs):
        epoch_loss = []
        matches, total = 0, 0

        for data, labels in training_data:

            optimizer.zero_grad()
            preds, batch_out = model(data)
            labels = labels.unsqueeze(1)

            loss = loss_func(preds, labels)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        matches += torch.eq(batch_out, labels).sum().item()
        total += torch.numel(batch_out)
        accuracy = matches / total

        print('Accuracy: {:4.2f}%'.format(accuracy * 100))
        print(f'Loss for epoch #{i+1}: {np.mean(epoch_loss)}')


def test(model, test_data, loss_func):
    model.eval()
    with torch.no_grad():
        test_loss = []
        matches, total = 0, 0
        for data, labels in test_data:

            data = data.transpose(0, 1).cuda()

            preds, batch_out = model(data)
            labels = labels.unsqueeze(1)

            loss = loss_func(preds, labels.cuda())
            test_loss.append(loss.item())

        matches += torch.eq(batch_out, labels).sum().item()
        total += torch.numel(batch_out)
        accuracy = matches / total
        print(f'Loss for test: {np.mean(test_loss)}')


##################################################################
##################################################################
# Main:


if __name__ == '__main__':

    # Stage I: Basic Sentiment Analysis with Word2Vec

    if(len(sys.argv) < 3 or len(sys.argv) > 3):
        print("Please enter the right amount of arguments in the following manner:",
              "\npython3 a3_kadakia_111304945 '*_train.csv' '*_trial.csv'")
        sys.exit(0)
    else:
        training_file = sys.argv[1]
        trial_file = sys.argv[2]

    print("\nStage 1 Checkpoint:\n")

    # Stage 1.1: Read the reviews and ratings from the file

    training_data = readCSV(training_file)
    trial_data = readCSV(trial_file)

    # Stage 1.3: Use GenSim word2vec to train a 128-dimensional word2vec
    #            model utilizing only the training data

    cores = multiprocessing.cpu_count()

    if "food_" in trial_file:
        train_w2v_model = trainWord2VecModel(training_data[1], cores)
        trial_w2v_model = trainWord2VecModel(trial_data[1], cores)

    if "music_" in trial_file:
        train_w2v_model = trainWord2VecModel(training_data[1], cores, 5)
        trial_w2v_model = trainWord2VecModel(trial_data[1], cores, 5)

    if "musicAndPetsup_" in trial_file:
        train_w2v_model = trainWord2VecModel(training_data[1], cores, 10)
        trial_w2v_model = trainWord2VecModel(trial_data[1], cores, 10)

    # Stage 1.4: Extract features

    train_x = getFeatures(train_w2v_model, training_data[1])
    test_x = getFeatures(trial_w2v_model, trial_data[1])
    train_y = np.asarray(training_data[2], dtype=np.int)
    test_y = np.asarray(trial_data[2], dtype=np.int)

    # Stage 1.5: Build a rating predictor

    bestALpha = buildRatingPredictor(train_x, train_y)
    rating_model = Ridge(
        random_state=42, alpha=bestALpha).fit(train_x, train_y)
    y_pred = rating_model.predict(test_x)

    # Threshold
    for pos in range(len(y_pred)):
        if y_pred[pos] <= 1:
            y_pred[pos] = 1
        if y_pred[pos] >= 5:
            y_pred[pos] = 5

    # Stage 1.6: Print both the mean absolute error and Pearson correlation
    #            between the predictions and the (test input) set

    print("Mean Absolute Error (test):", MAE(test_y, y_pred))
    print("Pearson product-moment correlation coefficients (test):",
          ss.pearsonr(test_y, y_pred))

    # Stage I: Checkpoint

    if "food_" in trial_file:
        testCases = [548, 4258, 4766, 5800]
        for case in testCases:
            if case in trial_data[0]:
                pos = trial_data[0].index(case)
                print()
                print(case, "\tPredicted Value",
                      y_pred[pos], "\tTrue Value:", trial_data[2][pos])
            else:
                print(case, "not in", trial_file)

    if "music_" in trial_file:
        testCases = [329, 11419, 14023, 14912]

        for case in testCases:
            if case in trial_data[0]:
                pos = trial_data[0].index(case)
                print()
                print(case, "\tPredicted Value",
                      y_pred[pos], "\tTrue Value:", trial_data[2][pos])
            else:
                print(case, "not in", trial_file)
    
    
    # Stage II: User-Factor Adaptation

    print("\n\nStage 2 Checkpoint:\n")

    # Stage 2.1: Grab the user_ids for both datasets.
    users = getUserBackground([training_file, trial_file])

    # Stage 2.2 For each user, treat their training data as "background" in order
    #           to learn user factors: average all of their word2vec features over
    #           the training data to treat as 128-dimensional
    #           "user-language representations".
    user_lang_rep = getUserLangRepresentation(train_w2v_model, users)

    # Stage 2.3: Run PCA the matrix of user-language representations to reduce down
    #            to just three factors. Save the 3 dimensional transformation matrix
    #            (V) so that you may apply it to new data (i.e. the trial or test set
    #            when predicting -- when predicting you should not run PCA again;
    #            only before training).
    v_matrix = runPCAMatrix(user_lang_rep)

    # Stage 2.4: Use the first three factors from PCA as user factors in order to run
    #            user-factor adaptation, otherwise using the same approach as stage 1.
    train_x_2 = PCA_feature_vector(
        train_x, test_x, training_data, trial_data, None, users, training_file, v_matrix)
    test_x_2 = PCA_feature_vector(
        train_x, test_x, training_data, trial_data, None, users, trial_file, v_matrix)

    bestALpha = buildRatingPredictor(train_x_2, train_y)
    rating_model = Ridge(random_state=42, alpha=bestALpha).fit(
        train_x_2, train_y)
    y_pred = rating_model.predict(test_x_2)

    # Threshold
    for pos in range(len(y_pred)):
        if y_pred[pos] <= 1:
            y_pred[pos] = 1
        if y_pred[pos] >= 5:
            y_pred[pos] = 5

    print("Mean Absolute Error (test):", MAE(test_y, y_pred))
    print("Pearson product-moment correlation coefficients (test):",
          ss.pearsonr(test_y, y_pred))

    # Stage II: Checkpoint

    if "food_" in trial_file:
        testCases = [548, 4258, 4766, 5800]

        for case in testCases:
            if case in trial_data[0]:
                pos = trial_data[0].index(case)
                print()
                print(case, "\tPredicted Value",
                      y_pred[pos], "\tTrue Value:", trial_data[2][pos])
            else:
                print(case, "not in", trial_file)

    if "music_" in trial_file:
        testCases = [329, 11419, 14023, 14912]

        for case in testCases:
            if case in trial_data[0]:
                pos = trial_data[0].index(case)
                print()
                print(case, "\tPredicted Value",
                      y_pred[pos], "\tTrue Value:", trial_data[2][pos])
            else:
                print(case, "not in", trial_file)
    
    
    # Stage III: Deep Learning

    print("\n\nStage 3 Checkpoint:\n")

    # Stage 3.1: Start with word2vec embeddings *per word* -- these may already be in memory from stage 1.
    training_data = build_dataloader(train_x, train_y, 128, False, cores-1)
    testing_data = build_dataloader(test_x, test_y, 128, False, cores-1)

    # model = LSTM_RNN(train_w2v_model)