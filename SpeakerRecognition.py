import time
import librosa
import config
from sklearn.externals import joblib
from Utils import LoggingHelper
from Utils import FeatureExtractionHelper

import itertools
import operator


def find_most_common(l):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(l))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(l)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


def find_the_speaker(sample_array, samp_rate, flen, hlen):
    # extract features from sample array
    features = FeatureExtractionHelper.extract_features(sample_array, samp_rate, flen, hlen)

    # get the model from pkl file
    model = joblib.load('model.pkl')

    # predict the speaker by using the model
    predicted_labels = model.predict(features)

    result = find_most_common(predicted_labels)
    return result


def main(audio_path):
    # set necessary properties
    samp_rate = config.Audio.samp_rate
    flen = config.Audio.flen
    hlen = config.Audio.hlen

    # get the audio as sample array
    sample_array, sr = librosa.load(audio_path, sr=config.Audio.samp_rate, mono=True)

    # get the current time
    start_time = time.time()

    # recognize the speaker
    result = find_the_speaker(sample_array, samp_rate, flen, hlen)

    # compute elapsed time for finding the speaker
    elapsed_time = time.time() - start_time

    LoggingHelper.log("The speaker is: " + result + " ( found in " + str(elapsed_time) + " )")

    return result


if __name__ == '__main__':
    main(config.Paths.TEST_PATH)
