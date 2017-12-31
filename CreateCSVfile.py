import numpy
import pandas
import config
from Utils import FeatureExtractionHelper
from Utils import SignalProcessingHelper
from Utils import FileHelper
from sklearn.preprocessing import StandardScaler


def main():
    # set necessary properties
    samp_rate = config.Audio.samp_rate
    flen = config.Audio.flen
    hlen = config.Audio.hlen

    # set the path of voice record database
    DATASET_PATH = config.Paths.DATASET_PATH

    # get subfolders
    train_folders = FileHelper.get_subdirectories(DATASET_PATH)
    # create a flag to control if dataset is created or not
    is_created = False
    # create an array that will keep the labels for each row
    labels = []

    print("Creating the dataset.....")
    # for each folder
    for folder in train_folders:
        # get sample array for each audio in the folder
        sample_arrays = SignalProcessingHelper.get_sample_arrays(DATASET_PATH, folder, samp_rate)
        # for each sample array
        for sample_array in sample_arrays:
            # extract features from the sample array of the audio
            mfccs = FeatureExtractionHelper.extract_mfccs(sample_array, samp_rate, flen, hlen)
            # check if the dataset is created or not
            if not is_created:
                # if it's not, create it
                dataset_numpy = numpy.array(mfccs)
                is_created = True
            elif is_created:
                # if it is created, add new features to the existed dataset
                dataset_numpy = numpy.vstack((dataset_numpy, mfccs))

            # turn the final numpy array into a Pandas DataFrame
            dataset_pandas = pandas.DataFrame(dataset_numpy)
            # add the speaker's name to "labels" array
            for i in range(0, mfccs.shape[0]):
                labels.append(folder)

    # set the speaker's name for each row under the "Speaker" column
    dataset_pandas["speaker"] = labels
    # get the final Pandas DataFrame as .csv file
    dataset_pandas.to_csv("dataset.csv", index=False)

    print("Data set has been created and sent to the project folder !")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    main()
