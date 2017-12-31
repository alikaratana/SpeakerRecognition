import python_speech_features
import config
import numpy


# function for extracting all the features
def extract_features(sample_array, sampling_rate, flen, hlen):
    # compute MFCCs
    mfccs = extract_mfccs(sample_array, sampling_rate, flen, hlen)
    return mfccs


# function for extracting MFCCs from sample array
def extract_mfccs(sample_array, sampling_rate, flen, hlen):
    # compute MFCCs
    mfccs = python_speech_features.mfcc(sample_array, sampling_rate, winlen=flen, winstep=hlen,
                                        numcep=13, winfunc=config.Windowing.hamming)
    mfccs = numpy.array(mfccs)
    # do not include the 1st MFCC
    mfccs = mfccs[:, 1:]
    return mfccs

