#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#
# Enrique Henestroza Anguiano :
#
# Calculates and describes the lexical coverage of a resource/conll source
# with respect to a conll (test set) target, open-class POS words only.
#

import sys
import codecs
from dtbutils import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-s", "--source", default="")
parser.add_option("-t", "--target", default="")
parser.add_option("-m", "--mode", default='plts') # 'conll'
parser.add_option("-p", "--pretreat", default='') # 'source', 'target', 'both'
parser.add_option("-w", "--wordonly", default=False, action="store_true")
parser.add_option("-i", "--ignorenpp", default=False, action="store_true")
(opts, args) = parser.parse_args()

# Load source vocabulary.
vocab = {} # cpos -> vocabset
f = codecs.open(opts.source, 'r', ENCODING)
if opts.mode == 'plts':
    for line in f:
        pterm = line.rstrip().split("\t")[0]
        cpos,lem = pterm.split("|")
        if cpos not in OPENCPOS:
            continue
        if opts.pretreat in ['source', 'both']:
            lem = pretreat_lem(lem, cpos)
        if cpos not in vocab:
            vocab[cpos] = {}
        vocab[cpos][lem] = True
elif opts.mode == 'conll':
    for line in f:
        if not line.rstrip():
            continue
        tok = line.rstrip().split('\t')
        cpos = tok[CPOS]
        if cpos not in OPENCPOS:
            continue
        if opts.wordonly:
            lex = tok[TOK]
        else:
            lem = tok[LEM]
            if opts.pretreat in ['source', 'both']:
                lem = pretreat_lem(lem, cpos)
            lex = lem
        if cpos not in vocab:
            vocab[cpos] = {}
        vocab[cpos][lex] = True
f.close()

# Compare against target vocabulary.
cov = {}
tot = {}
for cpos in vocab:
    cov[cpos] = 0
    tot[cpos] = 0
f = codecs.open(opts.target, 'r', ENCODING)
for line in f:
    if not line.rstrip():
        continue
    tok = line.rstrip().split('\t')
    cpos = tok[CPOS]
    lem = tok[LEM]
    if opts.ignorenpp and tok[FPOS] == "NPP":
        continue
    if cpos not in OPENCPOS or cpos == "V" and lem in STOP:
        continue

    if opts.wordonly:
        lex = tok[TOK]
    else:
        if opts.pretreat in ['target', 'both']:
            newlem = pretreat_lem(lem, cpos)
            if newlem != lem:
                continue
            lem = newlem
        lex = lem
        
    if cpos in vocab:
        if lex in vocab[cpos]:
            cov[cpos] += 1
        tot[cpos] += 1
f.close()

# Print results for coverage
for cpos in vocab:
    print "%(p)s\t%(c)02.02f" % {'p':cpos, 'c':100.0*cov[cpos]/tot[cpos]}
