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

    # create the model
    # model = SVC(C=1, kernel='linear')
    model = RandomForestClassifier(n_estimators=23)
    # model = KNeighborsClassifier(n_neighbors=1)

    # train the model
    TrainTestHelper.train_model(model, dnumpy_x, dnumpy_y)
    # save the model for future use
    joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    main()
