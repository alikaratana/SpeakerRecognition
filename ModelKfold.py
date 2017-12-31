import pandas
from sklearn.externals import joblib
from Utils import TrainTestHelper
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# main function
def main():
    # get Dataset as DataFrame
    data_set = pandas.read_csv('dataset.csv', index_col=False)
    print("Shape of Dataset:", data_set.shape)
    print("---------------------------------")
    # split Dataset into features(x) and labels(y)
    dnumpy_x, dnumpy_y = TrainTestHelper.split_dframe_x_y(data_set)
    # apply kfold cross validation and get folds
    folds = TrainTestHelper.split_dnumpy_train_test(dnumpy_x, dnumpy_y)
    # create the model
    model = SVC(C=1, kernel='linear')
    # model = RandomForestClassifier(n_estimators=23)
    # model = KNeighborsClassifier(n_neighbors=1)

    # apply kfold cross validation
    TrainTestHelper.apply_kfold(model, folds)
    # save the model for future use
    joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    main()
