#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Enrique Henestroza Anguiano
#

"""
DTBParser utility constants, dictionaries, etc.
"""

import re
import unicodedata
import copy
import sys

# Enumerate CONLL fields (with some added ones)
CONLLNUM = 15
ID,TOK,LEM,CPS,FPS,FEAT,GOV,LAB,MAP,GOLD,LCH,RCH,LABS,OBJ,REFL =range(CONLLNUM)

# Enumerate additional structural feature classes for a token.
TOKENFEATNUM = 22
TOKENTOTFEATNUM = CONLLNUM+TOKENFEATNUM
LVAL,LTOK,LLEM,LFPS,LCPS,LLAB,RVAL,RTOK,RLEM,RFPS,RCPS,RLAB,HTOK,HLEM,HFPS,\
    HCPS,HLAB,OTOK,OLEM,OFPS,OCPS,OLAB = range(CONLLNUM,TOKENTOTFEATNUM)

# Enumerate additional Parse Correction feature classes for a cand-dep pair.
CANDFEATNUM = 15
ISPR,GDIS,GPTH,LDIS,LDIR,NDEP,PUNC,PFPS,PCPS,PLAB,MFPS,MCPS,MLAB,SLPF,SBCT =\
    range(CANDFEATNUM)

# Basic feature items.
FEATITEMNUM = 10
S_2,S_1,S_0,N_0,N_1,N_2,N_3,D_0,C_0,C_D = range(FEATITEMNUM)

# Open part-of-speech categories.
OPENCPOS = set(["A", "ADV", "N", "V"])

# Enumerate weight functions for easy access
RELFREQ, CHISQ, TTEST, BINOM, PMI, LRATIO = range(6)
WGTIDS = {"relfreq":0, "chisq":1, "ttest":2, "binom":3, "pmi":4, "lratio":5}
WGTMIN = {"relfreq":0.0, "chisq":-1.0, "ttest":-1.0, "binom":-1.0, \
          "pmi":-1.0, "lratio":0.0}

# Encoding - make common to all modules
ENCODING = "utf-8"

# Stop-list for LEM clustering for verbes auxiliaires.
STOP = set([u"avoir", u"être"])

# Labels
LABELID = ('aff','a_obj','arg','ato','ats','aux_caus','aux_pass','aux_tps',\
           'comp','coord','de_obj','dep','dep_coord','det','mod','mod_rel',\
           'obj','p_obj','ponct','root','suj')
LABELENUM = tuple(enumerate(LABELID))
IDLABEL = dict([(y,x) for x,y in enumerate(LABELID)])
NUMLABEL = len(LABELID)

# Transitions (only arc-eager uses REDC)
SHFT = 0
ARCL = 1
ARCR = 2
REDC = 3

# Parser Models (only arc-eager-mc uses MRMCN)
MTRNS = 0
MLLAB = 1
MRLAB = 2
MRMCN = 3

# Corrector Models
MCMCN = 0
MCLAB = 1

# ** USER MODIFIED ** Location of preposition list and feature templates.
DATADIR = 'path/to/fredist/data/'
TEMPLATEDIR = DATADIR
PREPLIST = DATADIR+'/preps_autopos_autolem.txt'

# Intermediate POS between fine and coarse.
INTERPOS = {'V':'V', 'VINF':'VINF', 'VIMP':'V', 'VPP':'VPP', 'VPR':'VPR',\
            'NC':'N', 'NPP':'N', 'CS':'CS', 'CC':'CC', 'CLS':'CL', 'CLO':'CL',\
            'CLR':'CLR', 'CL':'CL', 'ADJ':'A', 'ADJWH':'A', 'ADV':'ADV',\
            'ADVWH':'ADV', 'PRO':'PRO', 'PROREL':'PROREL', 'PROWH':'PRO',\
            'DET':'D', 'DETWH':'D', 'P':'P', 'P+D':'P', 'ET':'ET', 'I':'I',\
            'PONCT':'PONCT', 'PREF':'PREF', 'VS':'V', 'P+PRO':'P'}

#
# FUNCTIONS
#

#
# Check for punctuation as-per eval07.pl
#
def is_punct(tok):
    for ch in tok:
        if unicodedata.category(ch)[0] != 'P':
            return False
    return True

#
# Grouping of some lemmas, for distributional methods.
#
def pretreat_lem(lem, cpos):
    tlem = lem
    # Group lemmas containing any numbers
    if re.search('\d', tlem, re.UNICODE):
        tlem = u'<NUM>'
    else:
        # Group open-pos lemmas if not alpha-numeric (except meta-characters)
        relem = re.sub(r'[\_\-\']', r'', tlem)
        if cpos in OPENCPOS and re.search('\W', relem, re.UNICODE):
            tlem = u'<NAN>'
    return tlem

#
# Read a CONLL sentence from a filestream.
#
def read_conll(instream, mode="parse", refl=True):
    tsent = [()] # Sentence tokens start at id=1.
    osent = []
    for line in instream:
        if line.rstrip() == "":
            # Add gold OBJ and REFL information.
            if mode in ["extract", "correct"]:
                for did in range(1, len(tsent)):
                    dep = tsent[did]
                    gid = dep[GOV]
                    gov = tsent[gid]
                    lab = dep[LAB]

                    # Add right- and left- children.
                    if gid != 0: # and dep[LAB] != "ponct":
                        if did < gid:
                            gov[LCH].append(did)
                        else:
                            gov[RCH].append(did)

                    # Add objects for pp-attachment and coordination.
                    if gid != 0 and\
                       ((gov[FPS] in ["P", "P+D"] and lab == "obj") or \
                       (gov[FPS] in ["CC"] and lab == "dep_coord")):
                        # Favor obj closest on the right
                        if not gov[OBJ] or \
                           (gid < did and (gov[OBJ] < gid or \
                                           did < gov[OBJ])) or \
                           (did < gid and gov[OBJ] < gid and \
                            did > gov[OBJ]):
                            gov[OBJ] = did

                    # Add reflexive marker to lemmas.
                    if dep[FPS] == "CLR" and gid != 0:
                        # Favor reflexive closest on the left
                        if not gov[REFL] or \
                           (gid > did and (gov[REFL] > gid or \
                                           did > gov[OBJ])) or \
                           (did > gid and gov[OBJ] > gid and \
                            did < gov[OBJ]):
                            gov[REFL] = did
                            if refl and gov[CPS] == "V":
                                # Check for 'faire' dep in between.
                                found_faire = False
                                if did < gid:
                                    for fid in range(did+1, gid):
                                        if tsent[fid][LEM] == u"faire":
                                            found_faire = True
                                            break
                                if not found_faire:
                                    gov[LEM] = u"se_"+gov[LEM]

#                            if tsent[gid][FPS] == "V":
#                                tsent[gid][LEM] = u"se_"+tsent[gid][LEM]
#                                # For reparsing, change lemma (for scores) but
#                                # leave map as-is (for other features).
#                                if mode == "extract":
#                                    fields_map = []
#                                    for lem,wgt in tsent[gid][MAP]:
#                                        fields_map.append((u"se_"+lem,wgt))
#                                    tsent[gid][MAP] = fields_map
            yield osent, tsent
            tsent = [()]
            osent = []
        else:
            fields = line.rstrip().split('\t')
            osent.append(copy.deepcopy(fields))
            # Modify fields required for treating the sentence.
            fields[ID] = int(fields[ID])
#            if mode in ["correct"]:
#                fields[FPS] = INTERPOS[fields[FPS]]
            if mode in ["extract"]:
                fields[LEM] = pretreat_lem(fields[LEM], fields[CPS])
            fields[GOLD] = None
            if fields[MAP] == "_" or fields[LEM] in STOP:
                fields[MAP] = {fields[LEM]:1.0}
            else:
                # Combine possible grouped lemmas
                fields_map = {}
                for x in fields[MAP].split('|'):
                    lem, wgt = x.rsplit('=', 1)
                    # lem = pretreat_lem(lem, fields[CPS]) #Assume pretreated!
                    wgt = float(wgt)
                    fields_map[lem] = fields_map.get(lem, 0.0) + wgt
                fields[MAP] = fields_map
            fields[MAP] = fields[MAP].items()
            fields_feat = {}
            if fields[FEAT] != "_":
                for feat in fields[FEAT].split("|"):
                    f,v = feat.split("=")
                    fields_feat[f] = v
            fields[FEAT] = fields_feat
            fields[GOV] = -1 if fields[GOV] == "_" else int(fields[GOV])
            tsent.append(fields + [[], [], {}, None, None])

#
# Convert sentence from original+treated lists to CONLL string.
#
def sentence_to_conll(osent, tsent):
    for tok in osent:
        if tsent:
            tok[GOV] = str(tsent[int(tok[ID])][GOV])
            tok[LAB] = str(tsent[int(tok[ID])][LAB])
        yield "\t".join(tok)
    yield ""

#
# Obtain a neighborhood surrounding a dependent's predicted governor.
# Optionally restrict the CPOS of candidate governors, and ignore certain
# CPOS for the purpose of projectivity constraints. Predicted governor
# ALWAYS returned (in 0th index).
#
def neighborhood(sent, did, neigh="dist-3", ipos=['PONCT']):
    d = sent[did]
    gid = d[GOV]
    g = sent[gid]

#    if neigh == "binary":
#        cands = [gid]
#        # If g is a N, it must be the object of a V
#        if g[CPOS] == "N" and g[LAB] == "obj" and \
#               sent[g[GOV]][CPOS] == "V" and \
#               sent[g[GOV]][ID] < gid and gid < did:
#            if not cpos or "V" in cpos:
#                cands.append(g[GOV])
#        # If g is a V, it must have an intervening N object
#        elif g[CPOS] == "V" and g[OBJ] and \
#             sent[g[OBJ]][CPOS] == "N" and \
#             gid < g[OBJ] and g[OBJ] < did:
#            if not cpos or "N" in cpos:
#                cands.append(g[OBJ])
#        return cands

    if neigh.startswith("dist"):
        type_dist = neigh.split("-")
        dist = None
        if len(type_dist) == 2:
            dist = int(type_dist[1])
        if dist and dist < 2:
            return [(gid, "1-0", sent[gid][FPS])]

        # Work way out
        cands = [] # list of (cid, cdist, cpath)
        seenc = set()
        hid = gid # First "up" node is the predicted governor
        hdist = 0
        hpath = []
        while hid and (not dist or hdist < dist):
            hpath.append(sent[hid][FPS])
            cands.append((hid, str(hdist+1)+"-0", "-".join(hpath)))
            seenc.add(hid)
            hdist += 1

            # Children of h nearest and to the left/right of d
            h_lid = None
            h_rid = None
            ch = sent[hid][LCH] if hid > did else sent[hid][RCH]
            for cid in ch:
                if sent[cid][FPS] not in ipos:
                    if cid < did:
                        h_lid = cid
                    elif cid > did:
                        h_rid = cid
                        break
            h_lid = None if h_lid in seenc else h_lid
            h_rid = None if h_rid in seenc else h_rid

            # Work way down right-most children of h_lid
            cid = h_lid
            hcpath = []
            cdist = 0
            while cid and (not dist or hdist + cdist < dist):
                hcpath.append(sent[cid][FPS])
                cands.append((cid, str(hdist)+"-"+str(cdist+1),\
                              "-".join(hpath+hcpath)))
                seenc.add(cid)
                c_rid = None
                for rid in sent[cid][RCH]:
                    if sent[rid][FPS] not in ipos:
                        c_rid = rid
                cid = c_rid
                cdist += 1

            # Work way down left-most children of h_rid
            cid = h_rid
            hcpath = []
            cdist = 0
            while cid and (not dist or hdist + cdist < dist):
                hcpath.append(sent[cid][FPS])
                cands.append((cid, str(hdist)+"-"+str(cdist+1),\
                              "-".join(hpath+hcpath)))
                seenc.add(cid)
                c_lid = None
                for lid in sent[cid][LCH]:
                    if sent[lid][FPS] not in ipos:
                        c_lid = lid
                        break
                cid = c_lid
                cdist += 1

            # Continue up according to certain conditions
            h_hid = sent[hid][GOV]
            if h_hid == 0 or (hid < did and h_rid or hid > did and h_lid):
                hid = None
            else:
                hid = h_hid

        return cands
