#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Enrique Henestroza Anguiano
#

"""
Association Score module. Agnostic about the two types of items associated..
"""

import sys
import os
import math
import codecs
import copy
import classifier
import cPickle
from dtbutils import *
from numpy import array
from itertools import chain, combinations, product, repeat

class AssociationScores(object):

    def __init__(self):
        self._x_alpha = None
        self._y_alpha = None
        self._scores = None # scores[xid][yid] = float
        self._examples = None
        self._defscore = -1.0
        self._x_red = None
        self._red = None


    #
    # Obtain a score given a x and a y.
    #
    def get_score(self, x, y, unkx=False, unky=False):
        xid = self._x_alpha.get(x, -1)
        yid = self._y_alpha.get(y, -1)

        if unkx and xid < 0:
            xid = self._x_alpha.get(x[:-1]+("<UNK>",), -1)
            if xid < 0:
                xid = self._x_alpha.get((x[0],"<UNK>",x[2],"<UNK>"), -1)
        if xid >= 0:
            if unky and yid < 0 or self._scores[xid][yid] == None:
                yid = self._y_alpha.get(y[:-1]+("<UNK>",), -1)
            if yid >= 0:
                score = self._scores[xid][yid]
                if score != None:
                    return True, score
        return False, self._defscore


    #
    # Choose a candidate from an example (first must be the default).
    # Returns an index in the example.
    #
    def choose_cand(self, ex, beta=0, unkx=False, unky=False):
        red = self._red
        x_red = self._x_red
        _, x, def_y = ex[0]
        def_exist, def_score = self.get_score(x,def_y,unkx=unkx,unky=unky)
        if not def_exist:
            return False, 0
        if x_red and x[:red] not in x_red:
            return True, 0

        max_score = def_score
        max_i = 0
        for i in range(1, len(ex)):
            _, _, alt_y = ex[i]
            alt_score = self.get_score(x,alt_y,unkx=unkx,unky=unky)[1]
            curbeta = x_red[x[:red]] if x_red else beta
            if self._defscore == -1.0:
                alt_score = (1 + alt_score) * curbeta - 1
            elif self._defscore == 0.0:
                alt_score *= curbeta
            if alt_score > max_score:
                max_score = alt_score
                max_i = i
        return True, max_i


    #
    # Read association scores from a pickle file.
    # 
    def load_scores(self, infile):
        pkstream = open(infile, "rb")
        self._x_alpha = cPickle.load(pkstream)
        self._y_alpha = cPickle.load(pkstream)
        self._scores = []
        for _ in range(len(self._x_alpha)):
            self._scores.append(cPickle.load(pkstream))
        pkstream.close()


    #
    # Tune x_red beta values, with a "red" size on the x tuple.
    #
    def tune_beta(self, beta, red=3):
        print >> sys.stderr, "Tuning beta parameters."
        x_alpha = self._x_alpha
        self._x_red = None
        self._red = None

        x_red = {}
        for x in x_alpha:
            x_red[x[:red]] = True
        for x in x_red.keys():
            cacc = [0]*len(beta)
            dacc = 0
            tot = 0
            exes = []
            for ex in self._examples:
                cur_x = ex[0][1]
                if not cur_x or cur_x[:red] != x:
                    continue
                if ex[0][0]:
                    dacc += 1
                for b in range(len(beta)):
                    _,max_cand = self.choose_cand(ex, beta=beta[b], \
                                                  unkx=False, unky=False)
                    if ex[max_cand][0]:
                        cacc[b] += 1
                tot += 1
            if tot > 0:
                cacc_b = [(cacc[b], len(beta)-b) for b in range(len(beta))]
                top_cacc, top_b = sorted(cacc_b, reverse=True)[0]
                if top_cacc > dacc:
                    x_red[x] = beta[len(beta)-top_b]
                    continue
            del x_red[x]
        self._x_red = x_red
        self._red = red


    #
    # Read evaluation examples from a pickle file. Each example is a tuple of
    # tuples: ((class1, x1, y1), (class2, x2, y2), ... )
    # 
    def load_examples(self, infile):
        pkstream = open(infile, "rb")
        self._examples = cPickle.load(pkstream)
        pkstream.close()

