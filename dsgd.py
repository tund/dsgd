"""Dual Space Gradient Descent
"""

from __future__ import division

import time
import numpy as np
from numpy.linalg import norm
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error

MODE = {"online": 1, "batch": 2}
MAINTENANCE = {"merging": 1, "removal": 2, "k-merging": 3}
LOSS = {"hinge": 1, "l1": 2, "l2": 3, "logit": 4, "eps_insensitive": 5}
TASK = {"classification": 1, "regression": 2}
KERNEL = {"gaussian": 1}


class DSGD(BaseEstimator):

    def __init__(self, mode="batch", loss="hinge", eps=0.1, kernel="gaussian", gamma=0.1,
                 lbd=1.0, avg_weight=False, max_size=1000,
                 maintain="k-merging", k=10, D=100,
                 record=-1, verbose=0):
        self.mode = mode
        self.loss = loss
        self.eps = eps
        self.kernel = kernel
        self.gamma = gamma
        self.lbd = lbd
        self.avg_weight = avg_weight
        self.max_size = max_size
        self.maintain = maintain
        self.k = k
        self.D = D
        self.record = record
        self.verbose = verbose

    def init(self):
        try:
            self.mode = MODE[self.mode]
        except KeyError:
            raise ValueError("Learning mode % is not supported." % self.mode)

        try:
            self.loss = LOSS[self.loss]
        except KeyError:
            raise ValueError("Loss function %s is not supported." % self.loss)

        try:
            self.kernel = KERNEL[self.kernel]
        except KeyError:
            raise ValueError("Kernel %s is not supported." % self.kernel)

        if self.loss == LOSS["hinge"] or self.loss == LOSS["logit"]:
            self.task_ = TASK["classification"]
        else:
            self.task_ = TASK["regression"]

        try:
            self.maintain = MAINTENANCE[self.maintain]
        except KeyError:
            raise ValueError("Maintenance strategy %s is not supported." % self.maintain)

        self.n_classes_ = 0
        self.class_name_ = None
        self.size_ = 0
        self.w_ = None
        self.idx_ = None
        self.X_ = None
        self.z_ = None        # weights for random features

        self.train_time_ = []
        self.last_train_time_ = 0
        self.score_ = []
        self.score_idx_ = []
        self.last_score_= 0

    def get_wx(self, X, x, rx):
        if self.size_ == 0:
            return [0]
        else:
            if self.kernel == KERNEL["gaussian"]:
                xx = (X[self.idx_[:self.size_]]-x)
                return np.sum(self.w_[:self.size_]*np.exp(-self.gamma*(xx*xx).sum(axis=1, keepdims=True)), axis=0) + rx.dot(self.rw_)
            else:
                return [0]

    def get_wxy(self, X, x, rx, y, wx=None):
        if self.size_ == 0:
            return (0, -1)
        else:
            if self.kernel == KERNEL["gaussian"]:
                if wx is None:
                    wx = self.get_wx(X, x, rx)
                idx = np.ones(self.n_classes_, np.bool)
                idx[y] = False
                z = np.argmax(wx[idx])
                z += (z >= y)
                return (wx[y] - wx[z], z)
            else:
                return (0, -1)

    def get_grad(self, X, x, rx, y, wx=None):
        if self.n_classes_ > 2:
            wxy, z = self.get_wxy(X, x, rx, y, wx)
            if self.loss == LOSS["hinge"]:
                return (-1, z) if wxy <= 1 else (0, z)
            else:   # logit loss
                if wxy > 0:
                    return (-np.exp(-wxy) / (np.exp(-wxy) + 1), z)
                else:
                    return (-1 / (1 + np.exp(wxy)), z)
        else:
            wx = self.get_wx(X, x, rx)[0] if wx is None else wx[0]
            if self.loss == LOSS["hinge"]:
                return (-y, -1) if y*wx <= 1 else (0, -1)
            elif self.loss == LOSS["l1"]:
                return (np.sign(wx - y), -1)
            elif self.loss == LOSS["l2"]:
                return (wx-y, -1)
            elif self.loss == LOSS["logit"]:
                if y*wx > 0:
                    return (-y*np.exp(-y*wx) / (np.exp(-y*wx) + 1), -1)
                else:
                    return (-y / (1 + np.exp(y*wx)), -1)
            elif self.loss == LOSS["eps_insensitive"]:
                return (np.sign(wx - y), -1) if np.abs(y - wx) > self.eps else (0, -1)

    def add_to_core_set(self, t, w, y, z):
        self.idx_[self.size_] = t
        if self.n_classes_ > 2:
            self.w_[self.size_, y] = w
            if z >= 0:
                self.w_[self.size_, z] = -w
        else:
            self.w_[self.size_] = w
        self.size_ += 1

    def remove(self, idx):
        n = len(idx)
        mask = np.ones(self.max_size, np.bool)
        mask[idx] = False
        self.w_[:-n] = self.w_[mask]
        self.idx_[:-n] = self.idx_[mask]
        self.size_ -= n
        # self.w_ = np.roll(self.w_, self.size_ - idx - 1)
        # self.idx_ = np.roll(self.idx_, self.size_ - idx - 1)
        # self.size_ -= 1

    def maintain_budget(self, rX, w):
        if self.maintain == MAINTENANCE["k-merging"]:
            i = np.argsort(norm(self.w_[:self.size_], axis=1))
            # i = np.argpartition(norm(self.w_[:self.size_], axis=1), self.k)
            self.rw_ += rX[self.idx_[i[:self.k]]].T.dot(self.w_[i[:self.k]])
            self.remove(i[:self.k])

        elif self.maintain == MAINTENANCE["removal"]:
            i = np.argmin(norm(self.w_, axis=1))
            mask = np.ones(self.max_size, np.bool)
            mask[i] = False
            self.X_[:-1] = self.X_[mask]
            self.w_[:-1] = self.w_[mask]
            self.size_ -= 1

    def fit(self, X, y):
        self.init()

        if self.mode == MODE["online"]:
            y0 = y

        if self.task_ == TASK["classification"]:
            self.class_name_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.class_name_)
            if self.n_classes_ == 2:
                y[y == 0] = -1

        if self.n_classes_ > 2:
            self.w_ = np.zeros([self.max_size, self.n_classes_])
            self.rw_ = np.zeros([2*self.D, self.n_classes_])
        else:
            self.w_ = np.zeros([self.max_size, 1])
            self.rw_ = np.zeros([2*self.D, 1])

        if self.avg_weight:
            w_avg = np.zeros(self.w_.shape)
            rw_avg = np.zeros(self.rw_.shape)

        # self.X_ = np.zeros([self.max_size, X.shape[1]])
        # self.rX_ = np.zeros([self.max_size, 2*self.D])
        self.idx_ = np.zeros(self.max_size, dtype=np.int)

        score = 0.0
        start_time = time.time()

        # initialize mapping matrix for random features
        self.u_ = (2*self.gamma)*np.random.randn(X.shape[1], self.D)

        # pre-allocate (FASTEST)
        rX = np.zeros([X.shape[0], 2*self.D])
        rX[:, :self.D] = np.cos(X.dot(self.u_)) / np.sqrt(self.D)
        rX[:, self.D:] = np.sin(X.dot(self.u_)) / np.sqrt(self.D)

        # horizontal stack
        # rX = np.hstack([np.cos(X.dot(self.u_))/np.sqrt(self.D), np.sin(X.dot(self.u_))/np.sqrt(self.D)])


        # pre-allocate + sin-cos
        # rX = np.zeros([X.shape[0], 2*self.D])
        # sinx = np.sin(X.dot(self.u_))
        # cosx = np.abs((1-sinx**2)**0.5)
        # signx = np.sign(((X.dot(self.u_)-np.pi/2)%(2*np.pi))-np.pi)
        # rX[:, :self.D] = (cosx*signx) / np.sqrt(self.D)
        # rX[:, self.D:] = sinx / np.sqrt(self.D)

        for t in xrange(X.shape[0]):

            if self.mode == MODE["online"]:
                wx = self.get_wx(X, X[t], rX[t])
                if self.task_ == TASK["classification"]:
                    if self.n_classes_ == 2:
                        y_pred = self.class_name_[wx[0] >= 0]
                    else:
                        y_pred = self.class_name_[np.argmax(wx)]
                    score += (y_pred != y0[t])
                else:
                    score += (wx[0]-y0[t])**2
                alpha_t, z = self.get_grad(X, X[t], rX[t], y[t], wx=wx)   # compute \alpha_t
            else:
                alpha_t, z = self.get_grad(X, X[t], rX[t], y[t])          # compute \alpha_t

            self.w_ *= (1.0*t)/(t+1)
            self.rw_ *= (1.0*t)/(t+1)

            w = -alpha_t/(self.lbd*(t+1))

            if self.size_ == self.max_size:
                self.maintain_budget(rX, w)
            self.add_to_core_set(t, w, y=y[t], z=z)

            if self.avg_weight:
                w_avg += self.w_
                rw_avg += self.rw_

            if self.record > 0 and (not ((t+1) % self.record)):
                self.train_time_.append(time.time()-start_time)
                self.score_.append(score/(t+1))
                self.score_idx_.append(t+1)

            if self.verbose:
                print "[INFO] Data point: %d\tModel size: %d\tElapsed time: %.4f" % (t, self.size_, time.time()-start_time)

        if self.avg_weight:
            self.w_ = w_avg / X.shape[0]
            self.rw_ = rw_avg / X.shape[0]

        if self.mode == MODE["online"]:
            self.last_train_time_ = time.time() - start_time
            self.last_score_ = score / X.shape[0]
        else:
            self.train_time_ = time.time() - start_time

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i in xrange(X.shape[0]):
            wx = self.get_wx(X[i])
            if self.task_ == TASK["classification"]:
                if self.n_classes_ == 2:
                    y[i] = self.class_name_[wx[0] >= 0]
                else:
                    y[i] = self.class_name_[np.argmax(wx)]
            else:
                y[i] = wx[0]
        return y

    def score(self, X, y):
        if self.mode == MODE["online"]:
            return -self.last_score_
        else:
            if self.task_ == TASK["classification"]:
                return float(accuracy_score(self.predict(X), y))
            else:
                return -float(mean_squared_error(self.predict(X), y))
