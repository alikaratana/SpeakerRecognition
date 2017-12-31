import librosa
import numpy


# function for getting sample arrays for each audio in person's train audio folder
def get_sample_arrays(train_dir, folder_name, samp_rate):
    # get the path for each audio file in the folder
    path_of_audios = librosa.util.find_files(train_dir + "/" + folder_name)
    # create an array that will store sample arrays for each audio
    audios = []
    # for each audio in folder
    for audio in path_of_audios:
        # get sample array x
        x, sr = librosa.load(audio, sr=samp_rate, mono=True)
        # add it to the array of samples
        audios.append(x)
    # convert the array of songs into a numpy array
    audios_numpy = numpy.array(audios)
    return audios_numpy
