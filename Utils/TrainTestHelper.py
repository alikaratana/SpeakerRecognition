import numpy
import time
import config
from sklearn.model_selection import StratifiedKFold


# function for splitting Dataset as features(x) and labels(y)
def split_dframe_x_y(dframe):
    # create a numpy array from DataFrame
    dnumpy = numpy.array(dframe)
    # get the size of the array
    size = dnumpy.shape[1]
    # get features as x
    x_dnumpy = dnumpy[:, :size - 1]
    # get labels as y
    y_dnumpy = dnumpy[:, size - 1]
    return x_dnumpy, y_dnumpy


# function for splitting Dataset as train and test
def split_dnumpy_train_test(dnumpy_x, dnumpy_y):
    # create kfold item for splitting the Dataset into n folds
    kfold = StratifiedKFold(n_splits=config.Model.n_splits)
    # create an array that will keep all folds
    folds = []
    # for each train and test indexes in folds
    for train_index, test_index in kfold.split(dnumpy_x, dnumpy_y):
        # get train part
        x_train, x_test = dnumpy_x[train_index], dnumpy_x[test_index]
        # get test part
        y_train, y_test = dnumpy_y[train_index], dnumpy_y[test_index]
        # add the fold to the array
        folds.append([x_train, y_train, x_test, y_test])
    return folds


def train_model(model, data_x, data_y):
    model.fit(data_x, data_y)


# function for applying KFold
def apply_kfold(model, folds):
    # get the current time
    start_time = time.time()
    # train and test the model to get the test score
    scores = []
    # for each fold
    for fold in folds:
        # train the model
        model.fit(fold[0], fold[1])
        # test the model
        score = model.score(fold[2], fold[3])
        print("Train size:", fold[1].size, "Test Size:", fold[3].size)
        print("Score:", score)
        # add the score to the array
        scores.append(score)

    avg_score = numpy.mean(scores)

    # compute elapsed time for training and testing the model
    elasped_time = time.time() - start_time
    print("-----------------------------------------------------------------------------")
    print("Average score for the model with ", config.Model.n_splits, "folds:", avg_score)
    print("Elapsed Time:", elasped_time)
