from __future__ import division

import time

import numpy as np

from dsgd import DSGD

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import grid_search
from sklearn.base import clone

import cPickle as pkl

import cProfile
import pstats

np.random.seed(6789)

DATA_DIR = "example_data"

N_RUNS = 10

if __name__ == '__main__':

    print("=========== Running on cod-rna data ===========")
    # X in [-1, 1]
    X_train, y_train = load_svmlight_file(DATA_DIR + "/cod-rna.scale", n_features=8)
    X_train = X_train.toarray()

    score = np.zeros(N_RUNS)
    train_time = np.zeros(N_RUNS)
    test_time = np.zeros(N_RUNS)

    clf = [None] * N_RUNS

    for r in xrange(N_RUNS):
        idx = np.random.permutation(X_train.shape[0])

        c = DSGD(mode="online", loss="logit", maintain="k-merging",
                 lbd=3.3593684387335183e-05, gamma=1.0, k=20, max_size=100, D=200)

        c.fit(X_train[idx], y_train[idx])
        train_time[r] = c.last_train_time_
        score[r] = c.last_score_

        print("Mistake rate = %.4f" % score[r])
        print("Training time = %.4f" % train_time[r])
        print("Budget size = %d" % c.size_)
        clf[r] = c

    print("%.4f\t%.2f+-%.2f" % (train_time.mean(), 100 * score.mean(), 100 * score.std()))

    print("========= Running on ijcnn1 data ===========")

    # X in [-1, 1]
    X_train, y_train = load_svmlight_file(DATA_DIR + "/ijcnn1", n_features=22)
    X_train = X_train.toarray()

    score = np.zeros(N_RUNS)
    train_time = np.zeros(N_RUNS)
    test_time = np.zeros(N_RUNS)

    clf = [None] * N_RUNS

    for r in xrange(N_RUNS):
        idx = np.random.permutation(X_train.shape[0])

        c = DSGD(mode="online", loss="logit", maintain="k-merging",
                 lbd=5.7142857142857142e-05, gamma=1.0, k=20, max_size=100, D=200)

        c.fit(X_train[idx], y_train[idx])
        train_time[r] = c.last_train_time_
        score[r] = c.last_score_

        print("Mistake rate = %.4f" % score[r])
        print("Training time = %.4f" % train_time[r])
        print("Budget size = %d" % c.size_)
        clf[r] = c

    print("%.4f\t%.2f+-%.2f" % (train_time.mean(), 100 * score.mean(), 100 * score.std()))
