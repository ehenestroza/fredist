# Mathieu Blondel, October 2010
#
# ** Added budgeting, ranking, sparse boolean vectors, and dev set tuning.
# ** Enrique Henestroza Anguiano, July 2012
#

import sys
import cPickle
import numpy as np
from collections import deque
from itertools import combinations

def sparse_dot(x, y):
    return len(x & y)

def polynomial_kernel(x, y, d=3):
    return sparse_dot(x, y) ** d

def polynomial_normdiff(x, y, d=3):
    if d < 1 or d > 3:
        print >> sys.stderr, "Normdiff implemented for d <= 3 only!"
        sys.exit(0)

    # Order 1
    nd_o1 = len((x | y) - (x & y))
    if d == 1:
        return nd_o1

    # Order 2
    x_o2 = set(combinations(sorted(x),2))
    y_o2 = set(combinations(sorted(y),2))
    nd_o2 = len((x_o2 | y_o2) - (x_o2 & y_o2))
    if d == 2:
        return nd_o1 + 2 * nd_o2

    # Order 3
    x_o3 = set(combinations(sorted(x),3))
    y_o3 = set(combinations(sorted(y),3))
    nd_o3 = len((x_o3 | y_o3) - (x_o3 & y_o3))
    if d == 3:
        return nd_o1 + 6 * (nd_o2 + nd_o3)

class KernelLBRankPerceptron(object):

    def __init__(self, kernel=polynomial_kernel, T=1, B=100):
        self.kernel = kernel
        self.normdiff = polynomial_normdiff
        self.T = T
        self.B = B

    def fit(self, XT1, XT1isd, XT2, XT2isd, XTidx, \
            XD1, XD1isd, XD2, XD2isd, XDidx, gm=False, bl=0, pa=0):
        n_t1_samples = len(XT1)
        n_t2_samples = len(XT2)
        n_d1_samples = len(XD1)
        n_d2_samples = len(XD2)

        # Gram matrices
        if gm:
            Kt1t1 = [[None]*n_t1_samples for _ in range(n_t1_samples)]
            Kt1t2 = [[None]*n_t2_samples for _ in range(n_t1_samples)]
            Kt2t1 = [[None]*n_t1_samples for _ in range(n_t2_samples)]
            Kt2t2 = [[None]*n_t2_samples for _ in range(n_t2_samples)]
            Kd1t1 = [[None]*n_t1_samples for _ in range(n_d1_samples)]
            Kd1t2 = [[None]*n_t2_samples for _ in range(n_d1_samples)]
            Kd2t1 = [[None]*n_t1_samples for _ in range(n_d2_samples)]
            Kd2t2 = [[None]*n_t2_samples for _ in range(n_d2_samples)]

        print >> sys.stderr, "Training..."
        print "Dev baseline: %d out of %d predictions correct " \
              % (bl,n_d1_samples)
        e_num = 0
        e_t1_idx = []
        e_t2_idx = []
        e_a = []
        bias = 0.0
        prev_dev_acc = 0.0
        for t in range(self.T):
            prev_a = e_a[:]
            prev_t1_idx = e_t1_idx[:]
            prev_t2_idx = e_t2_idx[:]
            prev_bias = bias
            print >> sys.stderr, "Iter:", t+1

            # Training.
            for t1 in range(n_t1_samples):
                print >> sys.stderr, t1,
                s1 = bias * XT1isd[t1] # 0
                for e in range(e_num):
                    e1 = e_t1_idx[e]
                    e2 = e_t2_idx[e]
                    if gm:
                        if Kt1t1[t1][e1] == None:
                            Kt1t1[t1][e1] = self.kernel(XT1[t1], XT1[e1])
                        if Kt1t2[t1][e2] == None:
                            Kt1t2[t1][e2] = self.kernel(XT1[t1], XT2[e2])
                        kt1e1 = Kt1t1[t1][e1]
                        kt1e2 = Kt1t2[t1][e2]
                    else:
                        kt1e1 = self.kernel(XT1[t1], XT1[e1])
                        kt1e2 = self.kernel(XT1[t1], XT2[e2])
                    s += e_a[e] * (kt1e1 - kt1e2)
                s2max = -sys.maxint
                s2idx = None
                for t2 in XTidx[t1]:
                    s = bias * XT2isd[t2] # 0
                    for e in range(e_num):
                        e1 = e_t1_idx[e]
                        e2 = e_t2_idx[e]
                        if gm:
                            if Kt2t1[t2][e1] == None:
                                Kt2t1[t2][e1] = self.kernel(XT2[t2], XT1[e1])
                            if Kt2t2[t2][e2] == None:
                                Kt2t2[t2][e2] = self.kernel(XT2[t2], XT2[e2])
                            kt2e1 = Kt2t1[t2][e1]
                            kt2e2 = Kt2t2[t2][e2]
                        else:
                            kt2e1 = self.kernel(XT2[t2], XT1[e1])
                            kt2e2 = self.kernel(XT2[t2], XT2[e2])
                        s += e_a[e] * (kt2e1 - kt2e2)
                    if s > s2max:
                        s2max = s
                        s2idx = t2
                tau = 0
#                if pa:
#                    # PA-I passive aggressive update
#                    if s1 - s2max < 1:
#                        loss = 1.0 - (s1 - s2max)
#                        #tau=min(pa,loss / self.normdiff(XT1[t1], XT2[s2idx]))
#                        tau = loss / self.normdiff(XT1[t1], XT2[s2idx])
                if s1 <= s2max:
                    tau = 1.0
                # Make update
                if tau > 0:
                    e_s2 = e_t2_idx.index(s2idx) if s2idx in e_t2_idx else None
                    if e_s2 == None:
                        if self.B and e_num >= self.B:
                            e_t1_idx.pop(0)
                            e_t2_idx.pop(0)
                            e_a.pop(0)
                            bias -= tau * (XT1isd[t1] - XT2isd[s2idx])
                            e_num -= 1
                        e_t1_idx.append(t1)
                        e_t2_idx.append(s2idx)
                        e_a.append(tau)
                        e_num += 1
                    else:
                        e_a[e_s2] += tau
                    bias += tau * (XT1isd[t1] - XT2isd[s2idx])
                print >> sys.stderr, "\r",
            print >> sys.stderr, "\r"
            # Dev scoring.
            print >> sys.stderr, "Dev Acc: ",
            dev_acc = 0
            for d1 in range(n_d1_samples):
                s1 = bias * XD1isd[d1] # 0
                for e in range(e_num):
                    e1 = e_t1_idx[e]
                    e2 = e_t2_idx[e]
                    if gm:
                        if Kd1t1[d1][e1] == None:
                            Kd1t1[d1][e1] = self.kernel(XD1[d1], XT1[e1])
                        if Kd1t2[d1][e2] == None:
                            Kd1t2[d1][e2] = self.kernel(XD1[d1], XT2[e2])
                        kd1e1 = Kd1t1[d1][e1]
                        kd1e2 = Kd1t2[d1][e2]
                    else:
                        kd1e1 = self.kernel(XD1[d1], XT1[e1])
                        kd1e2 = self.kernel(XD1[d1], XT2[e2])
                    s1 += e_a[e] * (kd1e1 - kd1e2)
                s2max = -sys.maxint
                for d2 in XDidx[d1]:
                    s = bias * XD2isd[d2] # 0
                    for e in range(e_num):
                        e1 = e_t1_idx[e]
                        e2 = e_t2_idx[e]
                        if gm:
                            if Kd2t1[d2][e1] == None:
                                Kd2t1[d2][e1] = self.kernel(XD2[d2], XT1[e1])
                            if Kd2t2[d2][e2] == None:
                                Kd2t2[d2][e2] = self.kernel(XD2[d2], XT2[e2])
                            kd2e1 = Kd2t1[d2][e1]
                            kd2e2 = Kd2t2[d2][e2]
                        else:
                            kd2e1 = self.kernel(XD2[d2], XT1[e1])
                            kd2e2 = self.kernel(XD2[d2], XT2[e2])
                        s += e_a[e] * (kd2e1 - kd2e2)
                    if s > s2max:
                        s2max = s
                if s1 > s2max:
                    dev_acc += 1

            print "%d out of %d predictions correct!" %(dev_acc,n_d1_samples)
#            print >> sys.stderr, "%.4f" % (float(dev_acc)/n_d1_samples)
            if dev_acc <= prev_dev_acc:
                e_a = prev_a
                e_t1_idx = prev_t1_idx
                e_t2_idx = prev_t2_idx
                bias = prev_bias
                break
            else:
                prev_dev_acc = dev_acc
        print >> sys.stderr, "done!"

        # Support vectors
        self.sv_1 = [XT1[e1] for e1 in e_t1_idx]
        self.sv_2 = [XT2[e2] for e2 in e_t2_idx]
        self.sv_a = e_a
        self.bias = bias
        print "%d support vectors out of %d constraints" % (len(self.sv_a),
                                                            n_t2_samples)

    def project(self, X, Xisd):
        n_samples = len(X)
        scores = [None]*n_samples
        for i in range(n_samples):
            s = self.bias * Xisd[i] # 0
            for sv_a, sv_1, sv_2 in zip(self.sv_a, self.sv_1, self.sv_2):
                s += sv_a * (self.kernel(X[i], sv_1) - self.kernel(X[i], sv_2))
            scores[i] = s
        return scores


#
#
# NON-MAINTAINED SECTION
#
#


class KernelLBPerceptron(object):

    def __init__(self, kernel=polynomial_kernel, T=1, B=100):
        self.kernel = kernel
        self.normdiff = polynomial_normdiff
        self.T = T
        self.B = B

    def fit(self, X, y):
        n_samples = len(X)
        alpha = [0.0]*n_samples

        # Gram matrix
        K = [[0]*n_samples for _ in range(n_samples)]

        print >> sys.stderr, "Training..."
        errqueue = deque()
        errset = set([])
        errnum = 0
        for t in range(self.T):
            print >> sys.stderr, t+1,
            for i in range(n_samples):
                s = 0
                for e in errqueue:
                    # build Gram matrix lazily
                    if K[i,e] == 0:
                        K[i,e] = self.kernel(X[i], X[e])
                    s += alpha[e] * K[i,e] * y[e]
                if s * y[i] <= 0:
                    if i not in errset:
                        if self.B and errnum >= self.B:
                            e = errqueue.popleft()
                            errset.remove(e)
                            errnum -= 1
                            alpha[e] = 0.0
                        errqueue.append(i)
                        errset.add(i)
                        errnum += 1
                    alpha[i] += 1.0
        print >> sys.stderr, "done!"

        # Support vectors
        self.sv_x = []
        self.sv_y = []
        self.sv_a = []
        for i in range(n_samples):
            if alpha[i] > 0:
                self.sv_x.append(X[i])
                self.sv_y.append(y[i])
                self.sv_a.append(alpha[i])

        print "%d support vectors out of %d points" % (len(self.sv_a),
                                                       n_samples)

    def project(self, X):
        n_samples = len(X)
        y_predict = [0.0]*n_samples
        for i in range(n_samples):
            s = 0
            for sv_a, sv_y, sv_x in zip(self.sv_a, self.sv_y, self.sv_x):
                s += sv_a * sv_y * self.kernel(X[i], sv_x)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        return [cmp(y,0) for y in self.project(X)]

if __name__ == "__main__":
#    import pylab as pl

    def gen_lin_separable_data(n=1000):
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, n)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, n)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data(n=1000):
        mean1 = [-1, 1]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, n/2)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, n/2)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, n/2)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, n/2)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2, n=900):
        X1_train = X1[:n]
        y1_train = y1[:n]
        X2_train = X2[:n]
        y2_train = y2[:n]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2, n=900):
        X1_test = X1[n:]
        y1_test = y1[n:]
        X2_test = X2[n:]
        y2_test = y2[n:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = Perceptron(T=20)
        print X_train, y_train
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

    def test_kernel():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = KernelLBPerceptron(gaussian_kernel, T=20, B=50)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

    def test_rank_kernel(fname):

        if not fname:
            n = 5000
            n1 = 8*n/10
            n2 = 9*n/10
            X1, _, X2, _ = gen_non_lin_separable_data(n)
            
            # Modify to sparse format
            X1sparse = [None]*n
            X2sparse = [None]*n
            for i in range(n):
                X1sparse[i] = list(enumerate(X1[i],1))
                X2sparse[i] = list(enumerate(X2[i],1))
            X1 = X1sparse
            X2 = X2sparse

            X1_train = X1[:n1]
            X2_train = X2[:n1]
            Xidx_train = [(i,) for i in range(len(X1_train))]
            X1_dev = X1[n1:n2]
            X2_dev = X2[n1:n2]
            Xidx_dev = [(i,) for i in range(len(X1_dev))]
            X1_test = X1[n2:]
            X2_test = X2[n2:]
        else:
            f = open(fname, 'rb')
            X1_train,X2_train,Xidx_train,X1_dev,X2_dev,Xidx_dev=cPickle.load(f)
            f.close()

        clf = KernelLBRankPerceptron(kernel=polynomial_kernel, T=20, B=100)
        clf.fit(X1_train, X2_train, Xidx_train, X1_dev, X2_dev, Xidx_dev)

        predict_X1 = clf.project(X1_dev)
        predict_X2 = clf.project(X2_dev)
        correct = 0
        for i in range(len(predict_X1)):
            if predict_X1[i] > max([predict_X2[j] for j in Xidx_dev[i]]):
                correct += 1
        print "%d out of %d predictions correct" % (correct, len(predict_X1))

    test_rank_kernel('CC.cmcn.dev.debug')
