#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Enrique Henestroza Anguiano
#

"""
Classifier for liblinear, svmlin. Agnostic about features or labels. Uses
'ranking', or dynamic classes.
"""

import sys
import os
import codecs
import svmlight
from libsvm import svmutil, svm
# Optional LIBLinear
try:
    from liblinear import liblinearutil, liblinear
    liblinear_found = True
except ImportError:
    liblinear_found = False
from ctypes import *
from dtbutils import *
from perceptron import KernelLBRankPerceptron, polynomial_kernel
import numpy as np
import cPickle


class Classifier(object):

    def __init__(self, param={'model':'', 'pref':'def', 'verbose':False,\
                              'classtype':'classifier'}):
        
        self._param = param

        # Model
        self._classtype = param['classtype']
        self._modelname = param['model']
        self._pref = param['pref']
        self._model = None
        self._numex_train = 0
        self._numex_dev = 0
        self._max_feat = 0
        self._labs = {}
        self._svmclassp = "-s 0 -t 1 -d 3 -g 0.1 -r 0.0 -e 1.0 -c 1.0 -q"
        self._svmstrucp = "-z p -t 1 -d 3 -s 0.1 -r 0.0 -e 1.0 -c 0.05 -b 0"


    #
    # Make a decision using a model on an example, A feature vector is:
    #     ((feat1,val1), (feat3,val3), ...)
    # Where each index corresponds to a feature in the model alphabet. Output
    # a list of tuples (class/idx, score) sorted by decending score. 
    #
    def score(self, feats):
        m = self._model

        if self._classtype == "classifier":
            x,_ = svm.gen_svm_nodearray(dict(feats))
            return int(svm.libsvm.svm_predict(m, x))

        elif self._classtype == "structured":
            maxscore = -sys.maxint
            maxidx = None
            for idx in range(len(feats)):
                dec_val = svmlight.classify(m, [(0, feats[idx])])
                if dec_val > maxscore:
                    maxscore = dec_val
                    maxidx = idx
            return maxidx

        elif self._classtype == "percrank":
            X = [None]*len(feats)
            Xisd = [0]*len(feats)
            Xisd[0] = 1
            for idx in range(len(feats)):
                X[idx] = set([f for f,v in feats[idx]])
            dec_vals = m.project(X, Xisd)
            return dec_vals.index(max(dec_vals))

    #
    # Reads a ranking problem.
    #
    def read_rank_problem(self, ef):
        efile = codecs.open(ef, 'r', 'ascii')
        qid = None
        allex = []
        rex = []

        print >> sys.stderr, "Reading ranking problem..."
        for line in efile:
            fields = line.rstrip().split(' ')
            glab = int(fields.pop(0))
            cqid = int(fields.pop(0).split(":")[1])
            feats = []
            for field in fields:
                f,v = field.split(":")
                #feats.append((int(f),float(v)))
                feats.append(int(f))
            feats = set(feats)
            if qid == None:
                qid = cqid
                rex = [(glab, feats)]
            elif qid == cqid:
                rex.append((glab, feats))
            else:
                allex.append(rex)
                qid = cqid
                rex = [(glab, feats)]
        allex.append(rex)
        efile.close()

        # Only supports a one-vs-all ranking (highest glab over rest)
        print >> sys.stderr, "Generating ranking constraints...",
        X1 = []
        X2 = []
        X2cnt = 0
        Xidx = []
        X1isdef = []
        X2isdef = []
        bline = 0
        for rex in allex:
            glabs = [glab for glab,_ in rex]
            gidx = glabs.index(max(glabs))
            cidx = []
            for i in range(len(rex)):
                glab,feats = rex[i]
                if i == 0 and glab == 1:
                    bline += 1
                if i == gidx:
                    X1.append(feats)
                    if i == 0:
                        X1isdef.append(1)
                    else:
                        X1isdef.append(0)
                else:
                    cidx.append(X2cnt)
                    X2.append(feats)
                    if i == 0:
                        X2isdef.append(1)
                    else:
                        X2isdef.append(0)
                    X2cnt += 1
            Xidx.append(tuple(cidx))
        print >> sys.stderr, X2cnt
        return X1, X1isdef, X2, X2isdef, Xidx, bline


    #
    # Append stream of examples to file. Feature vectors are as follows:
    #     [(feat1, val1), (feat3, val3), ..., (featn, valn)]
    #
    def write_examples(self, examples, mode="train"):
        exstream = codecs.open(self._modelname+"/"+self._pref+"."+mode,\
                               'a', 'ascii')

        # Classification examples over a single line. Label and feature vector:
        # 2 0:1 2:1 5:1
        # 5 1:1 2:1 4:1
        if self._classtype == "classifier":
            for glab,feats in examples:
                if mode == 'train':
                    self._numex_train += 1
                    self._max_feat = max(self._max_feat, feats[-1][0])
                    self._labs[glab] = True
                else:
                    self._numex_dev += 1
                print >> exstream, glab, \
                      " ".join([str(f)+":"+str(v) for f,v in feats])
        # Structured binary examples.
        # 1 qid:1 1:1 2:-1 5:-1
        # 0 qid:1 1:-1 2:1 4:-1
        elif self._classtype in ["structured", "percrank"]:
            for idxg,ex in examples:
                if mode == 'train':
                    self._numex_train += 1
                    qid = self._numex_train
                else:
                    self._numex_dev += 1
                    qid = self._numex_dev
                for idx in range(len(ex)):
                    feats = ex[idx]
                    if mode == 'train':
                        self._max_feat = max(self._max_feat, feats[-1][0])
                    if idxg == idx:
                        glab = 1
                    else:
                        glab = 0
                    print >> exstream, glab, 'qid:'+str(qid),\
                          " ".join([str(f)+":"+str(v) \
                                    for f,v in feats])
        exstream.close()


    #
    # Train model.
    #
    def train_model(self):
        if self._classtype in ["structured", "percrank"]:
            self._labs = {1:True}
        print >> sys.stderr, "Training model with",\
              self._numex_train,"examples,", self._max_feat+1, "features and",\
              len(self._labs), "labels."
        if self._numex_dev:
            print >> sys.stderr, "Also with", self._numex_dev,"dev examples."
            
        ef = self._modelname+"/"+self._pref+".train"
        df = self._modelname+"/"+self._pref+".dev"
        mf = self._modelname+"/"+self._pref+".model"
        if self._classtype == "classifier":
            os.system("$LIBSVM/svm-train "+self._svmclassp+" "+ef+" "+mf)
        elif self._classtype == "structured":
            os.system("$SVMLIGHT/svm_learn "+self._svmstrucp+" "+ef+" "+mf)
        elif self._classtype == "percrank":
            X1,X1isdef,X2,X2isdef,Xidx,bline = self.read_rank_problem(ef)
            X1dev,X1devisdef,X2dev,X2devisdef,Xdevidx,devbline = \
                self.read_rank_problem(df)
            m = KernelLBRankPerceptron(kernel=polynomial_kernel, T=10, B=0)
            m.fit(X1, X1isdef, X2, X2isdef, Xidx, X1dev, X1devisdef, X2dev,\
                  X2devisdef, Xdevidx, gm=False, bl=devbline)
            mfile = open(mf, 'wb')
            cPickle.dump([m.sv_a,m.sv_1,m.sv_2,m.bias], mfile, -1)
            mfile.close()


    #
    # Load model.
    #
    def load_model(self):
        if not os.path.isfile(self._modelname+"/"+self._pref+".model"):
            return False

        if self._classtype == "classifier":
            self._model = svmutil.svm_load_model(self._modelname+\
                                                 "/"+self._pref+".model")
        elif self._classtype == "structured":
            self._model = svmlight.read_model(self._modelname+\
                                              "/"+self._pref+".model")
        elif self._classtype == "percrank":
            m = KernelLBRankPerceptron(kernel=polynomial_kernel)
            mfile = open(self._modelname+"/"+self._pref+".model", 'rb')
            m.sv_a,m.sv_1,m.sv_2,m.bias = cPickle.load(mfile)
            mfile.close()
            self._model = m
        
        return True
        
