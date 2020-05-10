import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as MAE
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
from nltk.tokenize import word_tokenize
import multiprocessing

##########################################################################################
# Stage 1


def readCSV(fileName):
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

        item_ids.append(int(line[0]))
        ratings.append(int(line[1]))
        user_ids.append(line[4])
        reviews.append(word_tokenize(
            line[5].lower()) if type(line[5]) == str else "")

    return [item_ids, reviews, ratings, user_ids]


def trainWord2VecModel(reviews, cores=2, min_count=1):
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
                         window=10,
                         alpha=0.03,
                         negative=0,
                         min_count=min_count,
                         seed=42,
                         size=128)

    w2v_model.train(sentences=reviews,
                    total_examples=w2v_model.corpus_count,
                    epochs=30)

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

    listAlpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    minAccuracy = 1
    bestModel = None
    best_pearsonr = 0.35

    for alpha in listAlpha:

        model = Ridge(random_state=42, alpha=alpha).fit(train_x, train_y)
        y_pred = model.predict(test_x)

        acc = MAE(test_y, y_pred)
        pearsonr = ss.pearsonr(test_y, y_pred)

        if acc < 1 and acc <= minAccuracy:
            # print("Current Best Model acc:", acc, "C:", C)
            minAccuracy = acc
            bestModel = model
            best_pearsonr = pearsonr[0]

    return bestModel

##########################################################################################
# Stage 2


def getUserBackground(files):
    """
    Reads in all the files and stores them for future use.

    Attributes
    ----------
    files : list
        A list containing the training file path and trial file path.

    Returns
    -------
    users : dict
        A dictionary containing all the reviews and ratings grouped by users.

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

    print("user_ids:", len(user_ids))
    print("item_ids:", len(item_ids))
    print("ratings:", len(ratings))
    print("reviews:", len(reviews))

    return [user_ids, item_ids, ratings, reviews]


def getUserLangRepresentation(model, users):
    user_lang_rep = []

    for reviews in users[3]:
        temp_features = getFeatures(model, reviews)
        temp_array = [0] * 128

        for row in temp_features:
            temp_array += row

        user_lang_rep.append(temp_array/len(temp_features))

    return user_lang_rep


def runPCAMatrix(user_reviews):

    # user_PCA = PCA(n_components=3).fit(user_reviews).transform(user_reviews)
    # v_matrix = user_PCA.transform(user_reviews)

    return (PCA(n_components=3).fit(user_reviews)).transform(user_reviews)

                       # Part 1  Part 1  ReadCSV               ReadCSV              2.1        CLI   2.3
def PCA_feature_vector(X_train, X_test, review_training_data, review_testing_data, user_data, file, v_matrix):

    feature_vector = []

    dataFile = pd.read_csv(file, sep=',')

    for _, line in dataFile.iterrows():

        temp_feature_vector = []
        item_id = line[0]
        user_id = line[4]

        if item_id in review_training_data[0]:
            position = review_training_data[0].index(item_id)
            review_embedding = X_train[position]

        elif item_id in review_testing_data[0]:
            position = review_testing_data[0].index(item_id)
            review_embedding = X_test[position]

        if user_id in user_data[0]:
            position = user_data[0].index(user_id)
            user_factor = v_matrix[position]

        elif user_id in user_data[0]:
            position = user_data[0].index(user_id)
            user_factor = v_matrix[position]

        for vector in user_factor:
            temp_feature_vector.append(review_embedding * vector)
        
        temp_feature_vector.append(review_embedding)
        flattened_array = np.ndarray.flatten(np.array(temp_feature_vector))
        feature_vector.append(flattened_array)

    feature_vector = np.array(feature_vector)
    return feature_vector

def build_dataloader(embeddings, ratings, bs, shfle, nworkers):

    dataset = TensorDataset(torch.Tensor(embeddings), torch.Tensor(ratings))
    print("Embeddings:", torch.Tensor(embeddings).dim())
    print("Ratings", torch.Tensor(ratings).ndim)
    

    return DataLoader(dataset, batch_size=bs, shuffle=shfle, num_workers=nworkers)


class my_lstm_regressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(my_lstm_regressor, self).__init__()
    
        self.hs = hidden_size # internal hidden size
        self.inp = input_size # input feature space

        # define our LSTM model and linear decoder
        self.lstm_cell = nn.LSTMCell(self.inp, self.hs)
        self.decoder = nn.Linear(self.hs, 1)

        # tracks if hidden states should be moved to GPU on init
        self.run_cuda = torch.cuda.is_available()

        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, text):
    
        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
        
        #embedded = [sent len, batch size, emb dim]
        
        output, hidden = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        # init hidden state and cell state to tensor of 0s, both of size self.hidden_size
        hidden, cell = (weight.new_zeros(batch_size, self.hs), weight.new_zeros(batch_size, self.hs))

        # confirm they are on GPU
        if self.run_cuda:
            hidden.cuda()
            cell.cuda()

        return hidden, cell

def train(train_data, epochs=5):
    model.train()

    for i in range(epochs):
        epoch_loss = []
        for data, labels in train_data:
            optimizer.zero_grad()
            hidden = model.init_hidden(data.shape[0])
            
            # LSTMCell expects batch at dim=1
            data = data.transpose(0,1)#.cuda()
            
            preds = model(data, hidden)
            
            # flatten labels to match with predictions
            loss = loss_func(preds, labels.view(-1).cuda())
            
            # don't need to flatten for MSE
            loss = loss_func(preds, labels.cuda())

            loss.backward()

            optimizer.step()

            epoch_loss.append(loss.item())
        
        print(f'Loss for epoch #{i}: {np.mean(epoch_loss)}')


def test(test_data):
    model.eval()
    with torch.no_grad():
        test_loss = []
        for data, labels in test_data:
            hidden = model.init_hidden(data.shape[0])
            data = data.transpose(0,1).cuda()

            preds = model(data, hidden)
            #loss = loss_func(preds, labels.view(-1).cuda())
            loss = loss_func(preds, labels.cuda())
            test_loss.append(loss.item())

        print(f'Loss for test: {np.mean(test_loss)}')


##################################################################
##################################################################
# Main:


if __name__ == '__main__':

    ## Stage I: Basic Sentiment Analysis with Word2Vec

    if(len(sys.argv) < 3 or len(sys.argv) > 3):
        # print("Please enter the right amount of arguments in the following manner:",
        #       "\npython3 a3_kadakia_111304945 '*_train.csv' '*_trial.csv'")
        # sys.exit(0)
        training_file = 'food_train.csv'
        trial_file = 'food_trial.csv'
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
        train_w2v_model = trainWord2VecModel(training_data[1], cores, 5)
        test_w2v_model = trainWord2VecModel(trial_data[1], cores, 5)

    if "music_" in trial_file:
        train_w2v_model = trainWord2VecModel(training_data[1], cores, 10)
        test_w2v_model = trainWord2VecModel(trial_data[1], cores, 10)

    if "musicAndPetsup_" in trial_file:
        train_w2v_model = trainWord2VecModel(training_data[1], cores, 20)
        test_w2v_model = trainWord2VecModel(trial_data[1], cores, 20)

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



    ## Stage II: User-Factor Adaptation
    
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
    train_x_2 = PCA_feature_vector(train_x, test_x, training_data, trial_data, users, training_file, v_matrix)
    test_x_2 = PCA_feature_vector(train_x, test_x, training_data, trial_data, users, trial_file, v_matrix)

    rating_model = buildRatingPredictor(train_x_2, test_x_2, train_y, test_y)
    y_pred = rating_model.predict(test_x_2)

    print("Mean Absolute Error (test):", MAE(test_y, y_pred))
    print("Pearson product-moment correlation coefficients (test):",
          ss.pearsonr(test_y, y_pred))

    # Stage II: Checkpoint

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

    if "music" in trial_file:
        testCases = [329, 11419, 14023, 14912]

        for case in testCases:
            if case in trial_data[0]:
                pos = trial_data[0].index(case)
                print()
                print(case, "\tPredicted Value",
                      y_pred[pos], "\tTrue Value:", trial_data[2][pos])
            else:
                print(case, "not in", trial_file)



    ## Stage III: Deep Learning
    
    print("\n\nStage 3 Checkpoint:\n")

    # Stage 3.1: Start with word2vec embeddings *per word* -- these may already be in memory from stage 1.

    training_data = build_dataloader(train_x, train_y, 128, False, cores)
    testing_data = build_dataloader(test_x, test_y, 128, False, cores)

    print(training_data)

    # Stage 3.2:

    # instantiate model, optimizer, and loss function
    model = my_lstm_regressor(128, 128)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    #loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss()

    # move model to GPU if possible
    if torch.cuda.is_available():
        model.cuda()

    train(training_data)
    test(testing_data)