import numpy


class Paths:
    # Path of Dataset
    DATASET_PATH = "./Dataset/"

    # Path of test files
    TEST_PATH = " "


class Audio:
    # Sampling rate (Hz)
    samp_rate = 16000

    # Frame size (samples)
    fsize = 400
    # Frame length (seconds)
    flen = fsize / samp_rate

    # Hop size (samples)
    hsize = 160
    # Hop length (seconds)
    hlen = hsize / samp_rate


class Windowing:
    # hamming window
    hamming = lambda x: 0.54 - 0.46 * numpy.cos((2 * numpy.pi * x) / (400 - 1))


class Model:
    # n_splits of KFold
    n_splits = 1000

