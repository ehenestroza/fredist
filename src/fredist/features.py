#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Enrique Henestroza Anguiano
#

"""
DTB Parser (discriminative transition-based parsing). Feature module.
"""

import sys
import re
import copy
import codecs
from dtbutils import *
import ppattach
import assoc
from math import log
from moduleloader import load_module

class Features(object):

    def __init__(self, param={'arc':'eager', 'type':'parser', 'selpref':'',\
                              'subcat':'', 'labels':['unlab'],\
                              'verbose':False, 'featset':'default',\
                              'template':'', 'lexgen':'', 'lexgenk':0}):

        templateModule = load_module(param['template'])
        self._template = templateModule.Template()
        self._fv = self._template.feature_vector
        self._arc = param["arc"]
        self._ptype = param["type"]

        # Generalized lexical classes. dictionary: lem -> k-nearest
        self._lexgen = {}
        if param['lexgen']:
            lexgenf = codecs.open(param['lexgen'], 'r', 'utf-8')
            for wline in lexgenf:
                pterm,lexgens = wline.rstrip().split("\t")
                cpos,lem = pterm.split("|")
                # Only care about lexical class, not weight.
                lexgens = [x.split(":")[0].split('|')[1] \
                           for x in lexgens.split(" ")]
                if param['lexgenk'] > 0:
                    lexgens = lexgens[:param['lexgenk']]
                self._lexgen[(cpos,lem)] = lexgens
            lexgenf.close()

        # Subcategorization scores.
        self._ppattach = {}
        self._assoc = {}
        for assoct in ['subcat', 'selpref']:
            if param[assoct]:
                self._ppattach[assoct] = \
                    ppattach.PPAttachment(param={'verbose':False,\
                                                 'neigh':'dist-3',\
                                                 'type':assoct})
                self._assoc[assoct] = assoc.AssociationScores()
                self._assoc[assoct].load_scores(param[assoct])
        self._subcat = True if 'subcat' in self._assoc else False
        self._selpref = True if 'selpref' in self._assoc else False


    #
    # Obtain features given a sentence, stack, and buffer.
    #
    def get_features(self, sent, stack, buffer, cands, arc):
        f = [[[] for _ in range(TOKENTOTFEATNUM)] for _ in range(FEATITEMNUM)]
        t = [None]*FEATITEMNUM

        fpos = sent[buffer[0]][FPS]

        # Parser tokens: n_0, n_1, n_2, n_3, s_0, s_1, s_2
        if self._ptype == "parser":
            t[N_0] = sent[buffer[0]]
            t[N_1] = sent[buffer[1]] if len(buffer) > 1 else None
            t[N_2] = sent[buffer[2]] if len(buffer) > 2 else None
            t[N_3] = sent[buffer[3]] if len(buffer) > 3 else None

            # Transition decision
            if stack:
                t[S_0] = sent[stack[-1]]
                t[S_1] = sent[stack[-2]] if len(stack) > 1 else None
                t[S_2] = sent[stack[-3]] if len(stack) > 2 else None
        if self._ptype == "corrector" or arc in ["eagermc"]:
            t[D_0] = sent[buffer[0]]

        # Build up token features.
        for i in range(FEATITEMNUM):
            if t[i]:
                self.build_token_feats(sent, i, t[i], f[i], arc)

        # Build features for each candidate, return multiple feature vectors.
        if cands:
            candfeats = []
            for i in range(0, len(cands)):
                # Build candidate token features
                t[C_0] = sent[cands[i][0]]
                f[C_0] = [[] for _ in range(TOKENTOTFEATNUM)]
                self.build_token_feats(sent, C_0, t[C_0], f[C_0], arc)

                # Build candidate+dependent features
                f[C_D] = [[] for _ in range(CANDFEATNUM)]
                self.build_cand_feats(sent, C_D, t[D_0], cands[i], f[C_D])

                feats = self._fv(f[S_0],f[S_1],f[S_2],\
                                 f[N_0],f[N_1],f[N_2],f[N_3],\
                                 f[D_0],f[C_0],f[C_D],fpos)
                newfeats = []

                # Add selectional preference feature
                if self._selpref:
                    assocscore = self.get_pp_assoc(sent,t[D_0][ID],\
                                                   sent[cands[i][0]][ID],\
                                                   'selpref')
                    if assocscore != None:
                        newfeats.append((((C_D,SLPF),1),assocscore))
                if self._subcat:
                    assocscore = self.get_pp_assoc(sent,t[D_0][ID],\
                                                   sent[cands[i][0]][ID],\
                                                   'subcat')
                    if assocscore != None:
                        newfeats.append((((C_D,SBCT),1),assocscore))

                for ft in feats:
                    if self._lexgen:
                        (fttokid,ftvalid),ftval = ft
                        if ftvalid == LEM:
                            cpos = sent[t[fttokid][ID]][CPS]
                            lem = pretreat_lem(ftval,cpos)
                            if not (cpos.startswith("V") and lem in STOP \
                                    or lem in ['<NUM>','<NAN>']):
                                lexgens = self._lexgen.get((cpos,lem),None)
                                # Back-off to UNK lemma for a particular cpos.
                                if not lexgens:
                                    lexgens = self._lexgen.get((cpos,'<UNK>'),\
                                                               None)
                                if lexgens:
                                    for lexgen in lexgens:
                                        newfeats.append((((fttokid,ftvalid),\
                                                          lexgen),1.0))
                                    continue
                    # If anything fails, do no generalization.
                    newfeats.append((ft,1.0))
                candfeats.append(newfeats)
            return candfeats

        # Return single feature vector.
        else:
            f[C_0] = [[] for _ in range(TOKENTOTFEATNUM)]
            f[C_D] = [[] for _ in range(CANDFEATNUM)]
            feats = self._fv(f[S_0],f[S_1],f[S_2],f[N_0],f[N_1],f[N_2],f[N_3],\
                             f[D_0],f[C_0],f[C_D],fpos)
            newfeats = []
            for ft in feats:
                if self._lexgen:
                    (fttokid,ftvalid),ftval = ft
                    if ftvalid == LEM:
                        cpos = sent[t[fttokid][ID]][CPS]
                        lem = pretreat_lem(ftval,cpos)
                        if not (cpos.startswith("V") and lem in STOP \
                                or lem in ['<NUM>','<NAN>']):
                            lexgens = self._lexgen.get((cpos,lem),None)
                            # Back-off to UNK lemma for a particular cpos.
                            if not lexgens:
                                lexgens = self._lexgen.get((cpos,'<UNK>'),\
                                                           None)
                            if lexgens:
                                for lexgen in lexgens:
                                    newfeats.append((((fttokid,ftvalid),\
                                                      lexgen),1.0))
                                continue
                # If anything fails, do no generalization.
                newfeats.append((ft,1.0))
            return newfeats


    #
    # Build up a feature list corresponding to a token. 'i' is the FEATITEM id,
    # and 't' is the token item from a CONLL sentence.
    #
    def build_token_feats(self, sent, i, t, f, arc):
        # Primary features
        f[TOK].append(((i,TOK),t[TOK]))
        f[LEM].append(((i,LEM),t[LEM]))
        f[CPS].append(((i,CPS),t[CPS]))
        f[FPS].append(((i,FPS),t[FPS]))

        if t[OBJ]:
            toko = sent[t[OBJ]]
            f[OTOK].append(((i,OTOK),toko[TOK]))
            f[OLEM].append(((i,OLEM),toko[LEM]))
            f[OFPS].append(((i,OFPS),toko[FPS]))
            f[OCPS].append(((i,OCPS),toko[CPS]))
            f[OLAB].append(((i,OLAB),toko[LAB]))
        if t[LCH]:
            tokl = sent[t[LCH][0]]
            f[LTOK].append(((i,LTOK),tokl[TOK]))
            f[LLEM].append(((i,LLEM),tokl[LEM]))
            f[LFPS].append(((i,LFPS),tokl[FPS]))
            f[LCPS].append(((i,LCPS),tokl[CPS]))
            f[LLAB].append(((i,LLAB),tokl[LAB]))
        if t[RCH]:
            tokr = sent[t[RCH][-1]]
            f[RTOK].append(((i,RTOK),tokr[TOK]))
            f[RLEM].append(((i,RLEM),tokr[LEM]))
            f[RFPS].append(((i,RFPS),tokr[FPS]))
            f[RCPS].append(((i,RCPS),tokr[CPS]))
            f[RLAB].append(((i,RLAB),tokr[LAB]))
        if t[GOV]:
            tokh = sent[t[GOV]]
            f[HTOK].append(((i,HTOK),tokh[TOK]))
            f[HLEM].append(((i,HLEM),tokh[LEM]))
            f[HFPS].append(((i,HFPS),tokh[FPS]))
            f[HCPS].append(((i,HCPS),tokh[CPS]))
            f[LAB].append(((i,LAB),t[LAB]))

        # Existential features for arc-eager-mc 'frontier' of s_0
        if arc == 'eagermc' and i == S_0:
            t_cgov = t[GOV]
            while t_cgov != None:
                tokh = sent[t_cgov]
                f[TOK].append(((i,TOK),tokh[TOK]))
                f[LEM].append(((i,LEM),tokh[LEM]))
                f[FPS].append(((i,FPS),tokh[FPS]))
                f[CPS].append(((i,CPS),tokh[CPS]))
                t_cgov = tokh[GOV]
            f[TOK] = list(set(f[TOK]))
            f[LEM] = list(set(f[LEM]))
            f[FPS] = list(set(f[FPS]))
            f[CPS] = list(set(f[CPS]))


    #
    # Build up a feature list corresponding to a candidate pair.
    #
    def build_cand_feats(self, sent, i, d, cand, f):
        c,cdist,cpath = cand
        c = sent[c]

        # ISPR,GDIS,GPTH,LDIS,LDIR,NDEP,PUNC,MFPS,MLAB,PFPS,PLAB
        if d[GOV] == c[ID]:
            f[ISPR].append(((i,ISPR),1))
        else:
            f[ISPR].append(((i,ISPR),0))
        if cdist != None:
            f[GDIS].append(((i,GDIS),cdist))
        if cpath != None:
            f[GPTH].append(((i,GPTH),cpath))
        f[LDIS].append(((i,LDIS),min(4,int(0.5+log(1+abs(c[ID]-d[ID]),2)))))
        if c[ID] > d[ID]:
            f[LDIR].append(((i,LDIR),1))
        else:
            f[LDIR].append(((i,LDIR),0))
        f[NDEP].append(((i,NDEP),\
                        min(4,int(0.5+log(1+len(c[LCH])+len(c[RCH]),2)))))
        inter_brack = 0
        inter_quote = 0
        inter_perco = 0
        for j in range(min(c[ID],d[ID])+1, max(c[ID],d[ID])):
            if sent[j][TOK] in ['(',')','[',']']:
                inter_brack += 1
            elif sent[j][TOK] in ["'",'"']:
                inter_quote += 1
            elif sent[j][TOK] in ['.',':',';']:
                inter_perco += 1
        if inter_brack == 1 or inter_quote == 1 or inter_perco > 0:
            f[PUNC].append(((i,PUNC),1))
        else:
            f[PUNC].append(((i,PUNC),0))
        dm = None
        dp = None
        allch = c[LCH] if d[ID] < c[ID] else c[RCH]
        if allch:
            for j in range(len(allch)):
                if allch[j] > d[ID]:
                    if j > 0:
                        dm = allch[j-1]
                    dp = allch[j]
                    break
            else:
                dm = allch[-1]
        if dm:
            f[MFPS].append(((i,MFPS),sent[dm][FPS]))
            f[MCPS].append(((i,MCPS),sent[dm][CPS]))
            f[MLAB].append(((i,MLAB),sent[dm][LAB]))
        if dp:
            f[PFPS].append(((i,PFPS),sent[dp][FPS]))
            f[PCPS].append(((i,PCPS),sent[dp][CPS]))
            f[PLAB].append(((i,PLAB),sent[dp][LAB]))

        
    #
    # Obtain association preference features for pp-attachment.
    #
    def get_pp_assoc(self, sent, did, cid, assoct):
        pp, gov = self._ppattach[assoct].format_pp_gov(sent, did=did, cid=cid)

        if pp and gov:
            _, score = self._assoc[assoct].get_score(pp, gov)
            return score
        return None


#
#        if lexgen == "lexgen":
#
#            # Make a probabilistically smoothed feature vector.
#            new_feats = feats
#            feats = new_feats
#



