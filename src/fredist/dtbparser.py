#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Enrique Henestroza Anguiano
#

"""
DTB Parser (discriminative transition-based parsing).
"""

import sys
import features
import classifier
import re
import os
import codecs
import cPickle
from math import log
from itertools import chain, combinations, product
from operator import itemgetter
from dtbutils import *
from numpy import zeros, array

class DTBParser(object):

    def __init__(self, param={'model':'', 'template':'', 'type':'parser',\
                              'arc':'eager', 'diag':'', 'minfc':1,\
                              'selpref':'', 'fposgroups':(),\
                              'subcat':'', 'neigh':'dist-3',\
                              'featset':'default', 'lexgen':'', 'lexgenk':''}):
        
        self._param = param
        self._features = features.Features(self._param)

        # Open diagnostics file
        self._diag = None
        self._diagcnt = None
        if param['diag']:
            self._diag = codecs.open(param['diag'], 'w', 'utf-8')
            self._diagcnt = 1

        # Arc-Eager-MC statistics
        if param['arc'] in ['eagermc', 'eagerhy']:
            self._arceagermcstat = [0, 0, 0] # s_0_cnt, s_0_dom_cnt, other_cnt
        else:
            self._arceagermcstat = None

        # Models
        self._models = {}
        self._alphas = {}
        self._alphas_cnt = {}
        self._alphas_nextid = {}
        # shared objects between fpos in a group
        for fposgroup in param['fposgroups']:
            if param['type'] == 'parser':
                if param['arc'] in ['eager', 'standard']:
                    # Transition model (Shift, ArcL, ArcR, etc.)
                    param_trans = param.copy()
                    param_trans['pref'] = "-".join(fposgroup)+".trns"
                    param_trans['classtype'] = "classifier"
                    trans_model = classifier.Classifier(param_trans)
                    # ArcL labeling model
                    param_llab = param.copy()
                    param_llab['pref'] = "-".join(fposgroup)+".llab"
                    param_llab['classtype'] = "classifier"
                    llab_model = classifier.Classifier(param_llab)
                    # ArcR labeling model
                    param_rlab = param.copy()
                    param_rlab['pref'] = "-".join(fposgroup)+".rlab"
                    param_rlab['classtype'] = "classifier"
                    rlab_model = classifier.Classifier(param_rlab)
                    models = [trans_model, llab_model, rlab_model]
                    alphas = [{}, {}, {}]
                    alphas_cnt = [{}, {}, {}]
                    alphas_nextid = [[1], [1], [1]]
                    for fpos in fposgroup:
                        self._models[fpos] = models
                        self._alphas[fpos] = alphas
                        self._alphas_cnt[fpos] = alphas_cnt
                        self._alphas_nextid[fpos] = alphas_nextid
                elif param['arc'] in ['eagermc', 'eagerhy']:
                    # Transition model (Shift, ArcL, ArcR, etc.)
                    param_trans = param.copy()
                    param_trans['pref'] = "-".join(fposgroup)+".trns"
                    param_trans['classtype'] = "classifier"
                    trans_model = classifier.Classifier(param_trans)
                    # ArcL labeling model
                    param_llab = param.copy()
                    param_llab['pref'] = "-".join(fposgroup)+".llab"
                    param_llab['classtype'] = "classifier"
                    llab_model = classifier.Classifier(param_llab)
                    # ArcR labeling model
                    param_rlab = param.copy()
                    param_rlab['pref'] = "-".join(fposgroup)+".rlab"
                    param_rlab['classtype'] = "classifier"
                    rlab_model = classifier.Classifier(param_rlab)
                    # ArcR multiple candidate model
                    param_rmcn = param.copy()
                    param_rmcn['pref'] = "-".join(fposgroup)+".rmcn"
                    param_rmcn['classtype'] = "structured"
                    rmcn_model = classifier.Classifier(param_rmcn)
                    models = [trans_model, llab_model, rlab_model, rmcn_model]
                    alphas = [{}, {}, {}, {}]
                    alphas_cnt = [{}, {}, {}, {}]
                    alphas_nextid = [[1], [1], [1], [1]]
                    for fpos in fposgroup:
                        self._models[fpos] = models
                        self._alphas[fpos] = alphas
                        self._alphas_cnt[fpos] = alphas_cnt
                        self._alphas_nextid[fpos] = alphas_nextid
            elif param['type'] == 'corrector':
                # Corr Multiple candidate model
                param_cmcn = param.copy()
                param_cmcn['pref'] = "-".join(fposgroup)+".cmcn"
                param_cmcn['classtype'] = "structured" # "percrank"
                cmcn_model = classifier.Classifier(param_cmcn)
                # Corr Labeling model
                param_clab = param.copy()
                param_clab['pref'] = "-".join(fposgroup)+".clab"
                param_clab['classtype'] = "classifier"
                clab_model = classifier.Classifier(param_clab)
                models = [cmcn_model, clab_model]
                alphas = [{}, {}]
                alphas_cnt = [{}, {}]
                alphas_nextid = [[1], [1]]
                for fpos in fposgroup:
                    self._models[fpos] = models
                    self._alphas[fpos] = alphas
                    self._alphas_cnt[fpos] = alphas_cnt
                    self._alphas_nextid[fpos] = alphas_nextid


    # 
    # Parsing of a CONLL file, into parsed output or training examples.
    #
    def parse_conll(self, instream, mode="training"):
        cnt = 0
        for osent, tsent in read_conll(instream, mode="parse"):
            if cnt % 100 == 0:
                print >> sys.stderr, ".",
            exes = self.parse_sentence(tsent, mode=mode)
            if mode in ["parsing"]:
                for line in sentence_to_conll(osent, tsent):
                    yield line
            elif mode in ["training"]:
                for ex in exes:
                    yield ex
            cnt += 1
        print >> sys.stderr, cnt, "sentences."


    #
    # Given a CONLL sentence, yield a stream of examples for this sentence, or
    # yield a stream of CONLL-style token lines with predicted heads.
    #
    def parse_sentence(self, sent, mode="training"):
        param = self._param
        arc = param["arc"]
        neigh = param["neigh"]
        models = self._models
        diag = self._diag
        get_features = self._features.get_features

        # Wipe out governor and label information (no funny business!)
        ggovs = [None]
        glabs = [None]
        for id in range(1, len(sent)):
            ggovs.append(sent[id][GOV])
            glabs.append(sent[id][LAB])
            sent[id][GOV] = None
            sent[id][LAB] = None

        num_root = 0
        for ggov in ggovs:
            if ggov == 0:
                num_root += 1
                
        stack = [] # Last element is top of the stack.
        buffer = range(1, len(sent)) # First element is head of the buffer.
        untreated = dict([(x,True) for x in range(1, len(sent))])
        ex_parse = []

        if diag:
            diagstr = str(self._diagcnt)

        while len(buffer) > 0:

            # If stack is empty, can only SHFT
            if len(stack) == 0:
                if diag:
                    diagstr += " * SH"
                stack.append(buffer.pop(0))
                continue

            s_0 = stack[-1]
            n_0 = buffer[0]
            fpos = sent[n_0][FPS] # For model selection
            if fpos not in models:
                print >> sys.stderr, "POS tag not accounted for!"
                sys.exit(0)
                continue

            # Hard-coded to treat CC, CS, P, P+D, P+PRO, as eagermc.
            if param["arc"] == "eagerhy":
                if fpos in ["P", "P+D", "P+PRO"]: # "CC", "CS"]:
                    arc = "eagermc"
                else:
                    arc = "eager"

            # Check that all dependents are fulfilled for oracle training
            if mode == "training":
                n_0_fulfill = True
                if arc == "standard":
                    for id in untreated:
                        if ggovs[id] == n_0:
                            n_0_fulfill = False

                # Check that n_0 doesn't require access to a token on the stack
                n_0_access = False
                if arc == "eager":
                    if ggovs[n_0] in stack:
                        n_0_access = True
                    else:
                        for s_i in stack:
                            if ggovs[s_i] == n_0:
                                n_0_access = True
                                break

            # Check whether n_0 has gov among multiple-candidates of s_0
            if arc == "eagermc":
                cur_gov = sent[n_0][GOV]
                sent[n_0][GOV] = s_0 # provisional
                n_0_cands = neighborhood(sent, n_0, neigh="dist-0",ipos=[])
                sent[n_0][GOV] = cur_gov
                for idx in range(len(n_0_cands)):
                    if n_0_cands[idx][0] == ggovs[n_0]:
                        n_0_gmcidx = idx
                        break
                else:
                    n_0_gmcidx = None

            #
            # ACT decisions: shift, reduce, arc-left, arc-right(-mc)
            #
            best_trans = None
            best_lab_id = None
            feats = get_features(sent, stack, buffer, None, arc)
            
            # Find oracle action
            if mode == "training":
                if arc == "standard":
                    # LEFT-ARC
                    if ggovs[s_0] == n_0:
                        best_trans = ARCL
                    # RIGHT-ARC
                    elif ggovs[n_0] == s_0 and n_0_fulfill:
                        best_trans = ARCR
                    # SHFT default
                    else:
                        best_trans = SHFT
                elif arc == "eager":
                    # LEFT-ARC
                    if ggovs[s_0] == n_0:
                        best_trans = ARCL
                    # RIGHT-ARC
                    elif ggovs[n_0] == s_0:
                        best_trans = ARCR
                    # SHFT due to unfulfilled s_0
                    elif s_0 in untreated:
                        best_trans = SHFT
                    # REDC
                    elif n_0_access:
                        best_trans = REDC
                    # SHFT default
                    else:
                        best_trans = SHFT
                elif arc == "eagermc":
                    # LEFT-ARC
                    if ggovs[n_0_cands[-1][0]] == n_0:
                        best_trans = ARCL
                    # RIGHT-ARC-MC
                    elif n_0_gmcidx != None:
                        best_trans = ARCR
                    # SHFT default
                    else:
                        best_trans = SHFT
                ex_parse.append([fpos,MTRNS,best_trans,feats])

            # Find the highest-scoring legal action
            elif mode == "parsing":
                feats_trans = self.convert_features(fpos, MTRNS, feats,\
                                                    lock_alpha=True)
                m = models[fpos][MTRNS]
                best_trans = m.score(feats_trans) if m else SHFT

                # Check agorithmic action legality, default to SHFT if needed.
                if (arc == "standard" and\
                    ((best_trans == ARCL and s_0 not in untreated) or\
                     (best_trans == ARCR and n_0 not in untreated))) or\
                   (arc == "eager" and\
                    ((best_trans == ARCL and s_0 not in untreated) or\
                     (best_trans == REDC and s_0 in untreated))):
                    best_trans = SHFT

            # Perform SHFT action.
            if best_trans == SHFT:
                stack.append(buffer.pop(0))
                if diag:
                    diagstr += " SH"

            # Perform REDC action.
            elif best_trans == REDC:
                stack.pop()
                if diag:
                    diagstr += " RE"

            # Perform LEFT-ARC action.
            elif best_trans == ARCL:
                n_0_dep = s_0
                if arc == "eagermc":
                    n_0_dep = n_0_cands[-1][0]

                if mode == "training":
                    best_llab = glabs[n_0_dep]
                    ex_parse.append([fpos,MLLAB,IDLABEL[best_llab],feats])
                elif mode == "parsing":
                    feats_llab = self.convert_features(fpos, MLLAB, feats,\
                                                       lock_alpha=True)

                    m = models[fpos][MLLAB]
                    dec_llab = m.score(feats_llab) if m else 0
                    best_llab = LABELID[dec_llab]

                sent[n_0_dep][GOV] = n_0
                sent[n_0_dep][LAB] = best_llab
                if diag:
                    if arc == "eagermc":
                        for _ in range(len(n_0_cands)-1):
                            diagstr += " RE"
                    diagstr += " LA+"+best_llab
                    
                # Increment n_0 left children, labset
                sent[n_0][LCH] = sorted(sent[n_0][LCH] + [n_0_dep])
                sent[n_0][LABS][best_llab] = True
                
                # Update algorithm structures
                del untreated[n_0_dep]
                if arc == "eagermc": # Implicit 'RE' transitions
                    for _ in range(len(n_0_cands)):
                        stack.pop()
                else:
                    stack.pop()

            # Perform RIGHT-ARC(-MC) action.
            elif best_trans == ARCR:
                n_0_gov = s_0
                # Multiple candidate decision.
                if arc == "eagermc":
                    feats = get_features(sent, None, buffer, n_0_cands, arc)
                    if mode == "training":
                        best_mcidx = n_0_gmcidx
                        ex_parse.append([fpos,MRMCN,best_mcidx,feats])
                    elif mode == "parsing":
                        featsm = [None]*len(n_0_cands)
                        for i in range(len(n_0_cands)):
                            featsm[i] = self.convert_features(fpos, MRMCN,
                                                              feats[i],\
                                                              lock_alpha=True)
                        m = models[fpos][MRMCN]
                        best_mcidx = m.score(featsm) if m else 0
                    feats = feats[best_mcidx] # For labeling purposes
                    n_0_gov = n_0_cands[best_mcidx][0]

                # Label decision.
                if mode == "training":
                    best_rlab = glabs[n_0]
                    ex_parse.append([fpos,MRLAB,IDLABEL[best_rlab],feats])
                elif mode == "parsing":
                    feats_rlab = self.convert_features(fpos, MRLAB, feats,\
                                                       lock_alpha=True)
                    m = models[fpos][MRLAB]
                    dec_rlab = m.score(feats_rlab) if m else 0
                    best_rlab = LABELID[dec_rlab]

                sent[n_0][GOV] = n_0_gov
                sent[n_0][LAB] = best_rlab
                if diag:
                    if arc == "eagermc":
                        if len(n_0_cands) > 1:
                            if best_mcidx == 0:
                                self._arceagermcstat[0] += 1
                            elif best_mcidx == len(n_0_cands) - 1:
                                self._arceagermcstat[1] += 1
                            else:
                                self._arceagermcstat[2] += 1
                        for _ in range(best_mcidx):
                            diagstr += " RE"
                    diagstr += " RA+"+best_rlab
                    
                # Increment n_0_gov right children, labset
                sent[n_0_gov][RCH] = sorted(sent[n_0_gov][RCH] + [n_0])
                sent[n_0_gov][LABS][best_rlab] = True
                
                # Update algorithm structures
                del untreated[n_0]
                if arc == "standard":
                    buffer[0] = stack.pop()
                elif arc == "eager":
                    stack.append(buffer.pop(0))
                elif arc == "eagermc": # Implicit 'RE' transitions
                    for _ in range(best_mcidx):
                        stack.pop()
                    stack.append(buffer.pop(0))

        if diag:
            print >> diag, diagstr
            self._diagcnt += 1

        # Attach all untreated tokens to the root.
        if mode == "training" and len(untreated) != num_root:
            print >> sys.stderr, "Non-projective dep!", #, untreated
            # sys.exit(0)
        for i in untreated:
            sent[i][GOV] = 0
            sent[i][LAB] = 'root'

        # Output CONLL parsed tokens
        return ex_parse


    #
    # Correction of a parsed CONLL file, into output or training examples.
    #
    def correct_conll(self, instream, goldstream=None, mode="training"):
        cnt = 0
        if goldstream:
            gold_read_conll = read_conll(goldstream)
        for osent, tsent in read_conll(instream, mode="correct"):
            if cnt % 100 == 0:
                print >> sys.stderr, ".",
            if goldstream:
                _, tgsent = gold_read_conll.next() 
                for id in range(1, len(tsent)):
                    tsent[id][GOLD] = tgsent[id][GOV],tgsent[id][LAB]
            exes = self.correct_sentence(tsent, mode=mode)
            if mode in ["correcting", "oracle"]:
                for line in sentence_to_conll(osent, tsent):
                    yield line
            elif mode in ["training"]:
                for ex in exes:
                    yield ex
            cnt += 1
        print >> sys.stderr, "\n"+str(cnt)+" sentences."


    #
    # Given a CONLL sentence, yield a stream of correction examples for this
    # sentence, or yield a stream of CONLL-style lines with corrected heads.
    #
    def correct_sentence(self, sent, mode="training"):
        param = self._param
        arc = param["arc"]
        neigh = param["neigh"]
        models = self._models
        diag = self._diag
        get_features = self._features.get_features

        ex_correct = []

        if diag:
            diagstr = str(self._diagcnt)

        for did in range(1, len(sent)):
            d = sent[did]
            fpos = d[FPS] # For model selection
            if fpos not in models:
                continue

            # Do not correct root or "ponct" attachments.
            if d[GOV] == 0 or d[LAB] == "ponct":
                continue

            # "Undo" attachment.
            buffer = [did]
            if did in sent[d[GOV]][LCH]:
                sent[d[GOV]][LCH].remove(did)
            else:
                sent[d[GOV]][RCH].remove(did)

            # Get projective candidates for d, ignore "ponct" dependencies.
            cands = neighborhood(sent, did, neigh=neigh)

            gold_mcidx = None
            if mode in ["training", "oracle"]: # Check if gold governor exists
                for idx in range(len(cands)):
                    if cands[idx][0] == d[GOLD][0]:
                        gold_mcidx = idx
                        break
                else:
                    continue

            feats = get_features(sent, None, buffer, cands, arc)

            # Finding the correct candidate: MCMCN
            if len(cands) == 1:
                best_mcidx = 0
            elif mode == "training":
                best_mcidx = gold_mcidx
                ex_correct.append([fpos,MCMCN,best_mcidx,feats])
            elif mode == "oracle":
                best_mcidx = gold_mcidx
            elif mode == "correcting":
                featsm = [None]*len(feats)
                for i in range(len(feats)):
                    featsm[i] = self.convert_features(fpos, MCMCN, feats[i],\
                                                      lock_alpha=True)
                m = models[fpos][MCMCN]
                best_mcidx = m.score(featsm) if m else 0
#                print >> sys.stderr, d[TOK]+":",\
#                      [(sent[cands[idx][0]][TOK],score)\
#                       for idx,score in decs_rmcn]

            best_cand = cands[best_mcidx][0]

            # Finding the correct label for the best candidate: MCLAB
            feats = feats[best_mcidx]
            if mode == "training":
                best_clab = d[GOLD][1]
                ex_correct.append([fpos,MCLAB,IDLABEL[best_clab],feats])
            elif mode == "oracle":
                best_clab = d[GOLD][1]
            elif mode == "correcting" and best_mcidx != 0:
                feats_clab = self.convert_features(fpos, MCLAB, feats,\
                                                   lock_alpha=True)
                m = models[fpos][MCLAB]
                dec_clab = m.score(feats_clab) if m else 0
                best_clab = LABELID[dec_clab]
            else:
                best_clab = sent[did][LAB]

            if diag and d[GOV] != best_cand:
                candtoks = [sent[idx][TOK] for idx,_,_ in cands]
                diagstr += " CORR: ["+u",".join(candtoks)+"]"+" "+\
                           sent[best_cand][TOK]+" -> "+d[TOK]+","

            # Build training set non-sequentially (don't carry out corrections)
            if mode in ["training"]:
                best_cand = sent[did][GOV]
                best_clab = sent[did][LAB]
                
            sent[did][GOV] = best_cand
            sent[did][LAB] = best_clab
            if best_cand > did:
                sent[best_cand][LCH] = sorted(sent[best_cand][LCH]+[did])
            else:
                sent[best_cand][RCH] = sorted(sent[best_cand][RCH]+[did])

        if diag:
            print >> diag, diagstr
            self._diagcnt += 1

#        if mode != "training":
#            sys.exit(0)

        return ex_correct


    #
    # Converts features to format where feature names are integers, drop fpos.
    #
    def convert_features(self, fpos, midx, feats, lock_alpha=False):
        alpha = self._alphas[fpos][midx]
        if not lock_alpha:
            alpha_cnt = self._alphas_cnt[fpos][midx]
            alpha_nextid = self._alphas_nextid[fpos][midx]
        sfeats = []
        for f,v in feats:
            try:
                if not lock_alpha:
                    if f not in alpha:
                        alpha[f] = alpha_nextid[0]
                        alpha_cnt[f] = 1
                        alpha_nextid[0] += 1
                    else:
                        alpha_cnt[f] += 1
                sfeats.append((alpha[f],v))
            except KeyError:
                continue
        return sorted(sfeats)


    #
    # Train models.
    #
    def train_models(self, train, gold=False, dev=False, devg=False):
        param = self._param
        model = param['model']
        ptype = param['type']
        fposgroups = param['fposgroups']
        minfc = param['minfc']
        ms = self._models

        # Ensure that 'model' directory is a freshly created directory.
        os.system("rm -rf "+model)
        os.system("mkdir "+model)

        # Open CONLL stream(s)
        trainstream = codecs.open(train, 'r', ENCODING)
        goldstream = codecs.open(gold, 'r', ENCODING) if gold else None
        devstream = codecs.open(dev, 'r', ENCODING) if dev else None
        devgstream = codecs.open(devg, 'r', ENCODING) if devg else None

        # Parser models
        if ptype in ['parser']:
            # Do oracle parsing/reparsing
            for mode in ['train', 'dev']:
                print >> sys.stderr, "Oracle parsing for "+mode+" examples",
                if mode == 'train':
                    tmpstream = open(model+'/tmp.pkl', 'wb')
                    exes = self.parse_conll(trainstream, mode="training")
                    excnt = 0
                    for fpos,midx,best_dec,feats in exes:
                        if midx == MRMCN:
                            for i in range(len(feats)):
                                self.convert_features(fpos, midx, feats[i],\
                                                      lock_alpha=False)
                        else:
                            self.convert_features(fpos, midx, feats,\
                                                  lock_alpha=False)
                        excnt += 1
                        cPickle.dump([fpos,midx,best_dec,feats], tmpstream, -1)
                    tmpstream.close()

                    if self._arceagermcstat:
                        print >> sys.stderr, "Arc-Eager-MC statistics:"
                        print >> sys.stderr, self._arceagermcstat
                    if minfc > 1:
                        # Do hapax elimination, reduce alphabet.
                        print >> sys.stderr, "Removing hapax from alphabet..."
                        for fposgroup in fposgroups:
                            fpos = fposgroup[0]
                            for midx in range(len(ms[fpos])):
                                alpha_cnt = self._alphas_cnt[fpos][midx]
                                alpha = {}
                                alpha_nextid = 1
                                for f,cnt in alpha_cnt.items():
                                    if cnt >= minfc:
                                        alpha[f] = alpha_nextid
                                        alpha_nextid += 1
                                self._alphas[fpos][midx] = alpha
                                self._alphas_nextid[fpos][midx] =[alpha_nextid]
                    tmpstream = open(model+'/tmp.pkl', 'rb')
                    for _ in xrange(excnt):
                        fpos,midx,best_dec,feats = cPickle.load(tmpstream)
                        if midx == MRMCN:
                            ex = [best_dec, \
                                  [self.convert_features(fpos,midx,feats[i],\
                                                         lock_alpha=True)\
                                   for i in range(len(feats))]]
                        else:
                            ex = [best_dec, \
                                  self.convert_features(fpos, midx, feats,\
                                                        lock_alpha=True)]
                        ms[fpos][midx].write_examples([ex], mode=mode)
                    tmpstream.close()
                    os.system('rm '+model+'/tmp.pkl')
                elif mode == 'dev' and devstream:
                    exes = self.parse_conll(devstream, mode="training")
                    for fpos,midx,best_dec,feats in exes:
                        if midx == MRMCN:
                            ex = [best_dec, \
                                  [self.convert_features(fpos,midx,feats[i],\
                                                         lock_alpha=True)\
                                   for i in range(len(feats))]]
                        else:
                            ex = [best_dec, \
                                  self.convert_features(fpos, midx, feats,\
                                                        lock_alpha=True)]
                        ms[fpos][midx].write_examples([ex], mode=mode)
                    tmpstream.close()

        # Corrector models
        elif ptype == 'corrector':
            # Do oracle correcting
            for mode in ['train', 'dev']:
                print >> sys.stderr, "Oracle correcting for "+mode+" examples",
                if mode == 'train':
                    tmpstream = open(model+'/tmp.pkl', 'wb')
                    exes = self.correct_conll(trainstream,\
                                              goldstream=goldstream,\
                                              mode="training")
                    excnt = 0
                    for fpos,midx,best_dec,feats in exes:
                        if midx == MCMCN:
                            for i in range(len(feats)):
                                self.convert_features(fpos, midx, feats[i],\
                                                      lock_alpha=False)
                        else:
                            self.convert_features(fpos, midx, feats,\
                                                  lock_alpha=False)
                        excnt += 1
                        cPickle.dump([fpos,midx,best_dec,feats], tmpstream, -1)
                    tmpstream.close()
                    if minfc > 1:
                        # Do hapax elimination, reduce alphabet.
                        print >> sys.stderr, "Removing hapax from alphabet..."
                        for fposgroup in fposgroups:
                            fpos = fposgroup[0]
                            for midx in range(len(ms[fpos])):
                                alpha_cnt = self._alphas_cnt[fpos][midx]
                                alpha = {}
                                alpha_nextid = 1
                                for f,cnt in alpha_cnt.items():
                                    if cnt >= minfc:
                                        alpha[f] = alpha_nextid
                                        alpha_nextid += 1
                                self._alphas[fpos][midx] = alpha
                                self._alphas_nextid[fpos][midx] =[alpha_nextid]
                    tmpstream = open(model+'/tmp.pkl', 'rb')
                    for _ in xrange(excnt):
                        fpos,midx,best_dec,feats = cPickle.load(tmpstream)
                        if midx == MCMCN:
                            ex = [best_dec, \
                                  [self.convert_features(fpos,midx,feats[i],\
                                                         lock_alpha=True)\
                                   for i in range(len(feats))]]
                        else:
                            ex = [best_dec, \
                                  self.convert_features(fpos, midx, feats,\
                                                        lock_alpha=True)]
                        ms[fpos][midx].write_examples([ex], mode=mode)
                    tmpstream.close()
                    os.system('rm '+model+'/tmp.pkl')
                elif mode == "dev" and devstream and devgstream:
                    exes = self.correct_conll(devstream,\
                                              goldstream=devgstream,\
                                              mode="training")
                    for fpos,midx,best_dec,feats in exes:
                        if midx == MCMCN:
                            ex = [best_dec, \
                                  [self.convert_features(fpos,midx,feats[i],\
                                                         lock_alpha=True)\
                                   for i in range(len(feats))]]
                        else:
                            ex = [best_dec, \
                                  self.convert_features(fpos, midx, feats,\
                                                        lock_alpha=True)]
                        ms[fpos][midx].write_examples([ex], mode=mode)
                    tmpstream.close()

        for fposgroup in fposgroups:
            fpos = fposgroup[0]
            print >> sys.stderr, "----------\n", \
                  "Training models for fpos group:", "-".join(fposgroup)
            for midx in range(len(ms[fpos])):
                # Model training
                m = ms[fpos][midx]
                # Check if any features exist
                numfeats = self._alphas_nextid[fpos][midx][0]-1
                if numfeats > 0:
                    m.train_model()

        # Save alphabets and templates
        pkstream = open(model+"/alphas.pkl", 'wb')
        cPickle.dump(self._alphas, pkstream, -1)
        os.system('cp '+param['template']+' '+model+'/templates.py')
        pkstream.close()
        print >> sys.stderr, ". done!"

        trainstream.close()
        if devstream:
            devstream.close()
        if goldstream:
            goldstream.close()


    #
    # Load models.
    #
    def load_models(self):
        param = self._param
        fposgroups = param['fposgroups']
        model = param['model']
        ms = self._models

        print >> sys.stderr, "Loading models",

        # Load alphabets and models
        pkstream = open(model+"/alphas.pkl", 'rb')
        self._alphas = cPickle.load(pkstream)
        pkstream.close()
        for fposgroup in fposgroups:
            fpos = fposgroup[0]
            for midx in range(len(ms[fpos])):
                if not ms[fpos][midx].load_model():
                    ms[fpos][midx] = None
                print >> sys.stderr, ".",

        print >> sys.stderr, "done!"


if __name__ == "__main__":

    import sys
    import optparse

    parser = optparse.OptionParser()
    parser.add_option("-r", "--arc", \
                      default='eager', \
                      help="Transition-based parsing style (default: eager).")
    parser.add_option("-y", "--parsertype", \
                      choices=['parser', 'corrector'], \
                      default='parser', \
                      help="Parser or corrector.")
    parser.add_option("-f", "--divfpos", \
                      default='V-VINF_CC_P-P+D', \
                      help="':'-/_ separated lists of fpos groups.")
    parser.add_option("-t", "--train", \
                      default='', \
                      help="Read training parse data from CONLL file.")
    parser.add_option("-d", "--dev", \
                      default='', \
                      help="Read dev parse data from CONLL file.")
    parser.add_option("-e", "--devg", \
                      default='', \
                      help="Read dev gold data from CONLL file.")
    parser.add_option("-g", "--gold", \
                      default='', \
                      help="Read gold parse data from CONLL file.")
    parser.add_option("-p", "--predict", \
                      default='', \
                      help="Read predicted parse data from CONLL file.")
    parser.add_option("-m", "--model", \
                      default='', \
                      help="File prefix for dumping/loading models.")
    parser.add_option("-c", "--minfc", \
                      type="int", \
                      default=1, \
                      help="The minimum frequency for a feature to be used.")
    parser.add_option("-k", "--jackknife", \
                      type="int", \
                      default=0, \
                      help="Perform jack-knifed training/parsing.")
    parser.add_option("-i", "--featset", \
                      default='default', \
                      help="Feature set to use.")
    parser.add_option("-x", "--lexgen", \
                      default='', \
                      help="File for lexical generalization on features.")
    parser.add_option("-a", "--lexgenk", \
                      type="int", \
                      default=0, \
                      help="Limit on number of classes to use per word form.")
    parser.add_option("-n", "--neighbor", \
                      default='dist-3', \
                      help="Neighborhood type to use around dependents.")
    parser.add_option("-u", "--subcat", \
                      default='', \
                      help="Use subcat scores from file during parsing.")
    parser.add_option("-s", "--selpref", \
                      default='', \
                      help="Use selpref scores from file during parsing.")
    parser.add_option("--diagnostics", \
                      default='', \
                      help="File for outputting sentence action-sequences.")
    (opts, args) = parser.parse_args()

    fposgroups = tuple([tuple(x.split('-')) for x in opts.divfpos.split('_')])

    param  = {'type':opts.parsertype, 'selpref':opts.selpref,\
              'subcat':opts.subcat, 'neigh':opts.neighbor, 'minfc':opts.minfc,\
              'arc':opts.arc, 'lexgen':opts.lexgen, 'lexgenk':opts.lexgenk,\
              'model':opts.model, 'featset':opts.featset,\
              'fposgroups':fposgroups, 'diag':opts.diagnostics}

    # Identify feature template to use.
    if not opts.train and opts.model:
        param['template'] = param['model']+"/templates.py"
    elif param['type'] in ['parser']:
        param['template'] = TEMPLATEDIR+"/"+param['type']+"_"+\
                            param['arc']+"_"+param['featset']+".py"
    elif param['type'] in ['corrector']:
        param['template']  = TEMPLATEDIR+"/"+param['type']+"_"+\
                             param['featset']+".py"

    dtbp = DTBParser(param)

    if not opts.model and not (not opts.train and opts.gold and \
                               opts.parsertype == "corrector"):
        print >> sys.stderr, "Please provide location of model to train/load."
        sys.exit(0)

    if opts.jackknife and opts.train and opts.predict:
        fsize = int(opts.jackknife)
        ostream = codecs.open(opts.predict,'w',ENCODING)
        print >> sys.stderr, "***"+str(fsize)+"-FOLD JACKKNIFING STARTED***"

        for i in range(fsize):
            # Count number of sentences in instream.
            instream = codecs.open(opts.train,'r',ENCODING)
            cnt = 0
            for line in instream:
                if line == "\n":
                    cnt += 1
            instream.close()
            cnt = cnt / fsize

            # Create temporary fold i and not i CONLL files
            fstream = codecs.open(opts.predict+".f"+str(i), "w", \
                                  ENCODING)
            nstream = codecs.open(opts.predict+".n"+str(i), "w", \
                                  ENCODING)
            j = 0
            curcnt = 0
            instream = codecs.open(opts.train,'r',ENCODING)
            for line in instream:
                if i == j:
                    fstream.write(line)
                else:
                    nstream.write(line)
                if line == "\n":
                    curcnt += 1
                    if curcnt == cnt and j+1 < fsize:
                        curcnt = 0
                        j += 1
            instream.close()
            fstream.close()
            nstream.close()

            # Train "not" model for fold i
            print >> sys.stderr, "***TRAINING NOT-FOLD "+str(i)+"***"
            fnoti = opts.predict+".n"+str(i)
            dtbp = DTBParser(param)
            dtbp.train_models(fnoti)
            
            # Test "not" model on fold i, print output
            print >> sys.stderr, "***PARSING FOLD "+str(i)+"***"
            dtbp = DTBParser(param)
            dtbp.load_models()
            fi = opts.predict+".f"+str(i)
            instream = codecs.open(fi,'r',ENCODING)
            for outs in dtbp.parse_conll(instream, mode="parsing"):
                ostream.write(outs+"\n")
            instream.close()
            os.system('rm '+opts.predict+'.*')

        print >> sys.stderr, "***JACKKNIFING FINISHED***"

    # Train DTB model from input CONLL file, and dump model to file.
    elif opts.train:
        print >> sys.stderr, "***TRAINING STARTED***"
        dtbp.train_models(opts.train, gold=opts.gold, dev=opts.dev,\
                          devg=opts.devg)
        print >> sys.stderr, "***TRAINING FINISHED***"
        
    # Load DTB model and parse input CONLL to standard output.
    else:
        instream = codecs.open(opts.predict,'r',ENCODING) if opts.predict else\
                   codecs.getreader(ENCODING)(sys.stdin)
        if opts.parsertype == "parser":
            dtbp.load_models()
            print >> sys.stderr, "***PARSING STARTED***"
            for outs in dtbp.parse_conll(instream, mode="parsing"):
                print outs.encode(ENCODING)
            print >> sys.stderr, "***PARSING FINISHED***"
        elif opts.parsertype == "corrector":
            if opts.gold and not opts.model:
                    print >> sys.stderr, "***CORRECTING (ORACLE) STARTED***"
                    gs = codecs.open(opts.gold, 'r', ENCODING)
                    for outs in dtbp.correct_conll(instream, goldstream=gs,\
                                                   mode="oracle"):
                        print outs.encode(ENCODING)
                    print >> sys.stderr, "***CORRECTING FINISHED***"
            else:
                dtbp.load_models()
                print >> sys.stderr, "***CORRECTING STARTED***"
                for outs in dtbp.correct_conll(instream, mode="correcting"):
                    print outs.encode(ENCODING)
                print >> sys.stderr, "***CORRECTING FINISHED***"
                
        if opts.predict:
            instream.close()

    del dtbp
