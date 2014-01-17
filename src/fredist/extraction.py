#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#
# Enrique Henestroza Anguiano :
#
# Conversion from CONLL format files to context relation count files, or 
# alternately bigram context relation count files. Then weight and similarity 
# calculation, followed by distributional thesaurus creation.
#
# CONLL file example line:
# 2       dogs    dog     N       NC      n=p     3       suj     _       _
#
# Context relation file example line:
# N|dog   *u*suj  V|bark  1
#
# Similarity half-matrix example lines:
# -       N|dog   N|cat   N|rat
# N|rat   0.1     0.2
# N|cat   0.5
# N|dog
#
# Distributional thesaurus example line:
# N|dog   N|cat:0.27      N|rat:0.22
# 

import sys
import os
import time
import cPickle
import operator
from math import sqrt, log, e
import re
import codecs
from scipy import stats, special
from dtbutils import *

class DistributionalSimilarityExtractor:

    def __init__(self):

        return

    
    #
    # Extract contexts from a CONLL file.
    #
    def conll_to_contexts(self, conll_file, ctxt_out_file, \
                          ctxt_type="syntactic", ctxt_dir="up,down", \
                          pterm_min_freq=1000, ctxt_min_freq=1000, \
                          pterm_pos="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", \
                          pterm_cpos="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", \
                          pterm_use_lem="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", \
                          sterm_pos="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", \
                          sterm_cpos="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", \
                          sterm_use_lem="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", \
                          skip_pos="P,P+D,CC,CS", skip_cpos="", \
                          skip_use_lem="P,P+D,CC,CS", skip_only=False, \
                          use_deplabel=True, weight_fun="pmi"):

        # Read in data.
        print >> sys.stderr, "Extracting context relations from CONLL..."
        t0 = time.time()

        pterm_min_freq = int(pterm_min_freq)
        ctxt_min_freq = int(ctxt_min_freq)
        ctxt_type_set = set(ctxt_type.split(","))
        ctxt_dir_set = set(ctxt_dir.split(","))
        pterm_pos_set = set(pterm_pos.split(","))
        pterm_cpos_set = set(pterm_cpos.split(","))
        pterm_use_lem_set = set(pterm_use_lem.split(","))
        sterm_pos_set = set(sterm_pos.split(","))
        sterm_cpos_set = set(sterm_cpos.split(","))
        sterm_use_lem_set = set(sterm_use_lem.split(","))
        skip_pos_set = set(skip_pos.split(","))
        skip_cpos_set = set(skip_cpos.split(","))
        skip_use_lem_set = set(skip_use_lem.split(","))

        sent = [()] # (LEMMA, CPS, FPS, HEAD, LABEL)
        pterm_cnt = {} # PTERM -> COUNT
        prel_cnt = {} # (PTERM, REL) -> COUNT
        crel_cnt = {} # PTERM -> (REL, STERM) -> COUNT
        ctxt_cnt = {} # (REL, STERM) -> COUNT
        rel_cnt = {} # REL -> COUNT
        sterm_cnt = {} # STERM -> COUNT
        tot_cnt = 0

        conll_f = codecs.open(conll_file, 'r', ENCODING)

        for _,sent in read_conll(conll_f, mode="extract"):

            # Extract context relations from a full sent.
            for i in range(1, len(sent)):
                crels = []
                dep = sent[i]
                deppos = dep[CPS] if dep[FPS] in pterm_cpos_set \
                         else dep[FPS]

                # Linear dependency context relations.
                if "linear" in ctxt_type_set:
                    prv = sent[i-1] if i > 1 else None
                    nxt = sent[i+1] if i < len(sent)-1 else None

                    # Store previous token relation.
                    if prv and \
                           "prev" in ctxt_dir_set and \
                           dep[FPS] in pterm_pos_set and \
                           prv[FPS] in sterm_pos_set:
                        prvpos = prv[CPS] if prv[FPS] in sterm_cpos_set \
                                 else prv[FPS]
                        rel = tuple(["*p*"])
                        pterm = deppos,"<"+deppos+">"
                        if dep[FPS] in pterm_use_lem_set:
                            pterm = deppos,dep[LEM]
                        # Store up to two relations, depending on sterm lex
                        sterm = prvpos,"<"+prvpos+">"
                        crels.append((pterm,sterm,rel))
                        if prv[FPS] in sterm_use_lem_set:
                            sterm = prvpos,prv[LEM]
                            crels.append((pterm,sterm,rel))

                    # Store next token relation.
                    if nxt and \
                           "next" in ctxt_dir_set and \
                           dep[FPS] in pterm_pos_set and \
                           nxt[FPS] in sterm_pos_set:
                        nxtpos = nxt[CPS] if nxt[FPS] in sterm_cpos_set \
                                 else nxt[FPS]
                        rel = tuple(["*n*"])
                        pterm = deppos,"<"+deppos+">"
                        if dep[FPS] in pterm_use_lem_set:
                            pterm = deppos,dep[LEM]
                        # Store up to two relations, depending on sterm lex
                        sterm = nxtpos,"<"+nxtpos+">"
                        crels.append((pterm,sterm,rel))
                        if nxt[FPS] in sterm_use_lem_set:
                            sterm = nxtpos,nxt[LEM]
                            crels.append((pterm,sterm,rel))

                # Syntactic dependency context relations.
                if "syntactic" in ctxt_type_set:
                    gov = sent[dep[GOV]]
                    path = []
                    if use_deplabel:
                        path.append(dep[LAB])
                        
                    # Skip at most one time to next governor up.
                    skipped = False
                    if len(gov) > 0 and gov[FPS] in skip_pos_set:
                        govpos = gov[CPS] if gov[FPS] in skip_cpos else \
                                 gov[FPS]
                        if gov[FPS] in skip_use_lem_set:
                            path.append(govpos+"|"+gov[LEM])
                        else:
                            path.append(govpos)
                        if use_deplabel:
                            path.append(gov[LAB])
                        gov = sent[gov[GOV]]
                        if len(gov) > 0:
                            if gov[FPS] in skip_pos_set: # Can't skip twice
                                gov = []
                            else:
                                skipped = True
                            
                    if len(gov) > 0 and (skipped or not skip_only):
                        # Store upward relation.
                        if "up" in ctxt_dir_set and \
                               dep[FPS] in pterm_pos_set and \
                               gov[FPS] in sterm_pos_set:
                            govpos = gov[CPS] if gov[FPS] in sterm_cpos_set \
                                     else gov[FPS]
                            rel = tuple(["*u*"] + path)
                            pterm = deppos,"<"+deppos+">"
                            if dep[FPS] in pterm_use_lem_set:
                                pterm = deppos,dep[LEM]
                            # Store up to two relations, depending on sterm lex
                            sterm = govpos,"<"+govpos+">"
                            crels.append((pterm,sterm,rel))
                            if gov[FPS] in sterm_use_lem_set:
                                sterm = govpos,gov[LEM]
                                crels.append((pterm,sterm,rel))

                        # Store downward relation.
                        if "down" in ctxt_dir_set and \
                               gov[FPS] in pterm_pos_set and \
                               dep[FPS] in sterm_pos_set:
                            govpos = gov[CPS] if gov[FPS] in pterm_cpos_set \
                                     else gov[FPS]
                            path.reverse()
                            rel = tuple(["*d*"] + path)
                            pterm = govpos,"<"+govpos+">"
                            if gov[FPS] in pterm_use_lem_set:
                                pterm = govpos,gov[LEM]
                            # Store up to two relations, depending on sterm lex
                            sterm = deppos,"<"+deppos+">"
                            crels.append((pterm,sterm,rel))
                            if dep[FPS] in sterm_use_lem_set:
                                sterm = deppos,dep[LEM]
                                crels.append((pterm,sterm,rel))

                # Store relevant pterm, context, context relation counts.
                for pterm,sterm,rel in crels:
                    ctxt = rel, sterm
                    prel = pterm, rel
                    crel = pterm, rel, sterm
                    pterm_cnt[pterm] = pterm_cnt.get(pterm, 0) + 1
                    ctxt_cnt[ctxt] = ctxt_cnt.get(ctxt, 0) + 1
                    if pterm not in crel_cnt:
                        crel_cnt[pterm] = {}
                    crel_cnt[pterm][ctxt] = crel_cnt[pterm].get(ctxt, 0) + 1
                    tot_cnt += 1

            sent = [()]
        conll_f.close()

        # print >> sys.stderr, "# crel occurrences:", tot_cnt

        # Retain and weight frequent pterm and ctxt only 
        fctxt = open(ctxt_out_file, "wb")

        # Store vocabularies
        id_to_pterm = []
        unk_pos = {}
        cnt = 0
        for pterm in sorted(pterm_cnt.keys()):
            pterm_pos, pterm_lem = pterm
            if pterm_pos.startswith("V") and pterm_lem in STOP:
                del pterm_cnt[pterm]
                del crel_cnt[pterm]
                continue
            if pterm_cnt[pterm] >= pterm_min_freq:
                id_to_pterm.append(pterm)
                cnt += 1
            else:
                unk = (pterm_pos, "<UNK>")
                if pterm_pos not in unk_pos:
                    unk_pos[pterm_pos] = True
                    pterm_cnt[unk] = 0
                    crel_cnt[unk] = {}
                    id_to_pterm.append(unk)
                    cnt += 1
                pterm_cnt[unk] += pterm_cnt[pterm]
                for ctxt in crel_cnt[pterm].keys():
                    crel_cnt[unk][ctxt] = crel_cnt[unk].get(ctxt, 0) + \
                                          crel_cnt[pterm][ctxt]
                del pterm_cnt[pterm]
                del crel_cnt[pterm]
                
        id_to_pterm = tuple(id_to_pterm)
        print >> sys.stderr, "# pterm found:", cnt
        id_to_ctxt = []
        cnt = 0
        for ctxt in sorted(ctxt_cnt.keys()):
            if ctxt_cnt[ctxt] >= ctxt_min_freq:
                id_to_ctxt.append(ctxt)
                cnt += 1
            else:
                del ctxt_cnt[ctxt]
        id_to_ctxt = tuple(id_to_ctxt)
        print >> sys.stderr, "# ctxt found:", cnt
        cPickle.dump(id_to_pterm, fctxt, -1)
        cPickle.dump(id_to_ctxt, fctxt, -1)

        # Reusable extremum weights, given the total count n
        n = tot_cnt
        n2 = n**2

        # PMI min: c1=c2=n/2, c12=1
        pmi_min = log(4.0/n, 2)

        # PMI max: c1=pterm_min_freq, c2=ctxt_min_freq, c12=min(c1,c2)
        maxmin_cut = float(max(pterm_min_freq, ctxt_min_freq))
        pmi_max = log(n/maxmin_cut, 2)

        # LRATIO min: useful only if p1 < p2, so min is 0
        lratio_min = 0.0

        # LRATIO max: c1=c2=c12=n/x, where x is optimal divisor
        x = 3.9215536345675
        x2 = x**2
        lratio_max  = -2*((n/x)*log(1/x2) + (n-n/x)*log(1-1/x2) - \
                          (n/x)*log(1/x) - (n-n/x)*log(1-1/x))

        for ptermid in xrange(len(id_to_pterm)):
            curvector = [None]*len(id_to_ctxt)
            pterm = id_to_pterm[ptermid]

            for ctxtid in xrange(len(id_to_ctxt)):
                ctxt = id_to_ctxt[ctxtid]

                # Quick values for math
                c12 = crel_cnt[pterm].get(ctxt, 0)
                c1 = pterm_cnt[pterm]
                c2 = ctxt_cnt[ctxt]
                p1 = float(c1 * c2) / n2 # Null hypothesis
                p2 = float(c12) / n # Alternative hypothesis
                cont = [[c12,c2-c12],[c1-c12,n+c12-c1-c2]]
                
                if weight_fun == "relfreq": # [0,1] proportion
                    wgt = float(c12) / c1

                elif weight_fun == "chisq": # [-1,1] p-value
                    _,pval,_,_ = stats.chi2_contingency(cont)
                    wgt = 1 - pval if p1 < p2 else pval - 1

                elif weight_fun == "ttest": # [-1,1] p-value
                    wgt = -1
                    if c12 > 0:
                        tval = (c12 - float(c1*c2)/n) /\
                               sqrt(c12 * (1 - float(c12/n)))
                        if tval < 0:
                            wgt = -1 + 2*special.stdtr(n, tval)
                        else:
                            wgt = 1 - 2*special.stdtr(n, -tval)

                elif weight_fun == "binom": # [-1,1] Exact one-sided test
                    wgt = 0.0
                    if p1 < p2: # implied that c12 > 0
                        wgt = stats.binom.cdf(c12-1,n,p1)
                    else:
                        wgt = stats.binom.cdf(c12,n,p1) - 1

                elif weight_fun == "pmi": # [-1,1] information
                    # Transform from [pmi_min,pmi_max]
                    wgt = pmi_min
                    #wgt = -1
                    if c12 > 0:
                        wgt = log(float(c12 * n) / (c1 * c2), 2)
                        #if pmi < 0:
                        #    wgt = -pmi / pmi_min
                        #else:
                        #    wgt = pmi / pmi_max

                elif weight_fun == "lratio": # [0, 1]
                    # Transform from [0, lratio_max]
                    wgt = lratio_min
                    if p1 < p2:
                        wgt = -2*(c12*log(p1) + (n-c12)*log(1-p1) - \
                                  c12*log(p2) - (n-c12)*log(1-p2))
                        wgt /= lratio_max

                curvector[ctxtid] = wgt
            cPickle.dump(curvector, fctxt, -1)
        fctxt.close()
        print >> sys.stderr, "Done in %s sec." %(time.time()-t0)


    #
    # Constructs a similarity matrix from a collection of contexts. Uses a
    # Cluto-compatible plain-text format.
    #
    def contexts_to_simmatrix(self, ctxt_file, simmatrix_file, \
                              measure_fun="cosine", \
                              pterm_pos="N", positive_only=True):

        print >> sys.stderr, "Calculating similarity matrix..."
        t0 = time.time()

        pterm_pos_set = pterm_pos.split(",")

        # Read in context relation table, with specific weight
        fctxt = open(ctxt_file, "rb")
        id_to_pterm = cPickle.load(fctxt)
        id_to_ctxt = cPickle.load(fctxt)
        newid_to_pterm = []
        newid_to_pterm_wgt2 = []
        creltab = []
        for x in range(len(id_to_pterm)):
            creltab_x = cPickle.load(fctxt)

            # Use only selection pterm pos
            if id_to_pterm[x][0] not in pterm_pos_set:
                continue
            newid_to_pterm.append(id_to_pterm[x])

            sumwgt = 0.0
            for y in range(len(id_to_ctxt)):
                if positive_only and creltab_x[y] < 0.0:
                    creltab_x[y] = 0.0
                sumwgt += creltab_x[y] ** 2
            newid_to_pterm_wgt2.append(sumwgt)
            creltab.append(creltab_x)
        fctxt.close()

        # Calculate similarity matrix, line-by-line.
        n = len(newid_to_pterm)
        fsim = codecs.open(simmatrix_file, "w", "ascii")
        fsim.write(str(n)+"\n")
        for x1 in xrange(n):
            outline = []
            for x2 in xrange(n):
                if x1 == x2:
                    outline.append(1.0)
                else:
                    top = 0.0
                    bot = sqrt(newid_to_pterm_wgt2[x1] * \
                               newid_to_pterm_wgt2[x2])
                    if bot == 0:
                        outline.append(0.0)
                    else:
                        for y in xrange(len(id_to_ctxt)):
                            top += creltab[x1][y] * creltab[x2][y]
                        outline.append(top/bot)
            fsim.write(" ".join(map(str, outline))+"\n")
        fsim.close()
        print >> sys.stderr, "\nDone in %s sec." %(time.time()-t0)


    #
    # Convert similarity matrix to clusters.
    #
    def simmatrix_to_clusters(self, ctxt_file, simmatrix_file, clust_file, \
                              vocab_cutoffs="0.1,0.2", pterm_pos="N", \
                              compress=False, reflex=False):
        
        print >> sys.stderr, "Creating clusters from similarity matrix..."
        t0 = time.time()

        pterm_pos_set = pterm_pos.split(",")

        # Read in vocabularies from context file
        fctxt = open(ctxt_file, "rb")
        id_to_pterm = cPickle.load(fctxt)
        id_to_ctxt = cPickle.load(fctxt)
        fctxt.close()

        # Use only selection pterm pos
        newid_to_pterm = []
        for x in range(len(id_to_pterm)):
            if id_to_pterm[x][0] not in pterm_pos_set:
                continue
            newid_to_pterm.append(id_to_pterm[x])

        # Perform clustering using CLUTO
        for x in vocab_cutoffs.split(","):
            vocab_cutoff = float(x)
            num_clust = str(int(vocab_cutoff*len(newid_to_pterm)))
            os.system("$CLUTO/scluster -clmethod=agglo -crfun=upgma "+\
                      simmatrix_file+" "+num_clust+" &> out.log")
            f = codecs.open(simmatrix_file+".clustering."+num_clust,\
                            'r', 'ascii')
            fclust = codecs.open(clust_file+"."+x, "w", ENCODING)
            for i in range(len(newid_to_pterm)):
                clustid = f.readline().rstrip()
                if not reflex and newid_to_pterm[i][1].startswith("se_"):
                    continue
                fclust.write("|".join(newid_to_pterm[i])+\
                             "\t"+newid_to_pterm[i][0]+"|"+clustid+":1.0\n")
            f.close()
            fclust.close()

        # Remove intermediary files
        os.system("rm "+simmatrix_file+".clustering.*")
        os.system("rm "+simmatrix_file+".tree")

        print >> sys.stderr, "\nDone in %s sec." %(time.time()-t0)


    #
    # Convert similarity matrix to (normalized) distributional thesaurus.
    #
    def simmatrix_to_disthes(self, ctxt_file, simmatrix_file, disth_file, \
                             neigh_cutoffs="10,50", pterm_pos="N", \
                             pterm_wgts="0.0,0.5", reflex=False):

        print >> sys.stderr, "Creating thesaurus from similarity matrix..."
        t0 = time.time()

        pterm_pos_set = pterm_pos.split(",")

        # Read in vocabularies from context file
        fctxt = open(ctxt_file, "rb")
        id_to_pterm = cPickle.load(fctxt)
        fctxt.close()

        # Use only selection pterm pos
        newid_to_pterm = []
        for x in range(len(id_to_pterm)):
            if id_to_pterm[x][0] not in pterm_pos_set:
                continue
            newid_to_pterm.append(id_to_pterm[x])

        # Write output(s)
        for pw in pterm_wgts.split(","):
            pw = float(pw)
            for nc in neigh_cutoffs.split(","):
                nc = int(nc)
                fdisth = codecs.open(disth_file+"."+str(pw)+"_"+str(nc),\
                                     "w", ENCODING)
                
                # Read in a similarity matrix from plain-text file
                fsim = codecs.open(simmatrix_file, "r", "ascii")
                n = int(fsim.readline().rstrip())

                # Sanity check
                if n != len(newid_to_pterm):
                    print >> sys.stderr, "Inconsistent input files!"
                    sys.exit(1)
                
                for x1 in xrange(n):
                    if not reflex and newid_to_pterm[x1][1].startswith("se_"):
                        continue
                    sim_x1 = map(float, fsim.readline().rstrip().split(" "))
                    sim_id_x1 = []
                    for x2 in xrange(n):
                        if x1 == x2:
                            continue
                        if not reflex and \
                               newid_to_pterm[x2][1].startswith("se_"):
                            continue
                        sim_id_x1.append((sim_x1[x2], x2))
                    sim_id_x1.sort(reverse=True)
                    if pw > 0:
                        sim_id_x1_out = ["|".join(newid_to_pterm[x1])+":"+\
                                         str(pw)]
                    else:
                        sim_id_x1_out = []
                    if nc > 0:
                        sim_id_x1 = sim_id_x1[:min(len(sim_id_x1),nc)]
                    density_x1 = sum([x for x,_ in sim_id_x1]) / (1-pw)
                    for sim,x2 in sim_id_x1:
                        if sim <= 0:
                            continue
                        sim_id_x1_out.append("|".join(newid_to_pterm[x2])+":"+\
                                             str(sim/density_x1))
                    fdisth.write("|".join(newid_to_pterm[x1])+"\t")
                    fdisth.write(" ".join(sim_id_x1_out)+"\n")

                fsim.close()
                fdisth.close()

        print >> sys.stderr, "Done in %s sec." %(time.time()-t0)


if __name__ == "__main__":

    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option("--in-corpus", dest="incorpus", default="input.conll", help='Give the filename of the input CONLL corpus to use. Default=\'input.conll\'.')
    parser.add_option("--out-name", dest="outname", default="output", help='Assign an output name prefix for generated files. Default=\'output\'.')
    parser.add_option("--ctxt-type", dest="ctxttype", default="syntactic", help='Comma separated list of context types, syntactic and/or linear. Default=\'syntactic\'.')
    parser.add_option("--ctxt-dir", dest="ctxtdir", default="up,down", help='Used only for syntactic ctxt-type. Define the direction within a tree of contexts with respect to the primary term. Default=\'up,down\'.')
    parser.add_option("--pterm-minfreq", dest="ptermfreq", default=100, help='The minimum required frequency of a primary term within desired context relations. Default=100.')
    parser.add_option("--ctxt-minfreq", dest="ctxtfreq", default=100, help='The minimum required frequency of a context within desired context relations. Default=100.')
    parser.add_option("--pterm-pos", dest="ptermpos", default="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", help='Comma separated list of fine POS categories to extract in the primary term position. Default=\'ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS\'.')
    parser.add_option("--pterm-cpos", dest="ptermcpos", default="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", help='Comma separated list of fine POS categories in the primary term position for which we want to use the coarse POS category instead. Default=\'ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS\'.')
    parser.add_option("--pterm-uselem", dest="ptermuselem", default="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", help='Comma separated list of fine POS categories in the primary term position for which we want to include the lemma. Default=\'ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS\'.')
    parser.add_option("--sterm-pos", dest="stermpos", default="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", help='Comma separated list of fine POS categories to extract in the secondary term position. Default=\'ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS\'.')
    parser.add_option("--sterm-cpos", dest="stermcpos", default="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", help='Comma separated list of fine POS categories in the secondary term position for which we want to use the coarse POS category instead. Default=\'ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS\'.')
    parser.add_option("--sterm-uselem", dest="stermuselem", default="ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS", help='Comma separated list of fine POS categories in the secondary term position for which we want to include the lemma. Default=\'ADJ,ADV,NC,V,VIMP,VINF,VPP,VPR,VS\'.')
    parser.add_option("--skip-pos", dest="skippos", default="P,P+D,CC,CS", help='Used only for syntactic ctxt-type. Comma separated list of fine POS categories in the secondary term position that we want to skip over and fold into the relation. Default=\'P,P+D,CC,CS\'.')
    parser.add_option("--skip-cpos", dest="skipcpos", default="P,P+D,CC,CS", help='Used only for syntactic ctxt-type. Comma separated list of fine POS categories skipped in the secondary term position for which we want to use the coarse POS category instead. Default=\'P,P+D,CC,CS\'.')
    parser.add_option("--skip-uselem", dest="skipuselem", default="P,P+D,CC,CS", help='Used only for syntactic ctxt-type. Comma separated list of fine POS categories skipped in the secondary term position for which we want to include the lemma in the relation folding. Default=\'P,P+D,CC,CS\'.')
    parser.add_option("--skip-only", dest="skiponly", action="store_true", default=False, help='Used only for syntactic ctxt-type. Determines whether to extract only those contexts in which skipping has occurred. Default=False.')
    parser.add_option("--use-deplabel", dest="usedeplabel", action="store_true", default=False, help='Used only for syntactic ctxt-type. Determines whether to include dependency labels in the relation. Default=False.')
    parser.add_option("--weight-fun", dest="weightfun", default="pmi", help='Used during context weighting. Specifies the weight function to use: pmi, ttest, lin, and relfreq. Default=\'pmi\'.')
    parser.add_option("--measure-fun", dest="measurefun", default="cosine", help='Used during similarity calculation. Specifies the measure function use: jaccard, lin, and cosine. Default=\'cosine\'.')
    parser.add_option("--disth-pos", dest="disthpos", default="A,ADV,N,V", help='Comma separated list of POS categories (fine or coarse depending on pterm-cpos) for which to construct distributional thesauri. Default=\'A,ADV,N,V\'.')
    parser.add_option("--vocab-cutoffs", dest="vocabcutoffs", default="0.10,0.20", help='Used during clustering calculation. Specifies the various vocab cutoffs to use in determining clustering stopping points.')
    parser.add_option("--neigh-cutoffs", dest="neighcutoffs", default="10,20", help='Used during thesaurus construction. Specifies the various neighbor cutoffs to use for each primary term.')
    parser.add_option("--pterm-wgts", dest="ptermwgts", default="0.5", help='Used during thesaurus construction. Specifies the various probability mass / weight to assign to each primary term.')
    parser.add_option("--from-ctxt", dest="fromctxt", action="store_true", default=False, help='Whether to assume weighted contexts have already been created.')
    parser.add_option("--from-sim", dest="fromsim", action="store_true", default=False, help='Whether to assume the similarity matrix has already been created.')
    (opts, args) = parser.parse_args()
    
    dse = DistributionalSimilarityExtractor()
    
    print >> sys.stderr, opts

    if not (opts.fromctxt or opts.fromsim):
        dse.conll_to_contexts(opts.incorpus, opts.outname+".ctxt.pkl", ctxt_type=opts.ctxttype, ctxt_dir=opts.ctxtdir, pterm_min_freq=opts.ptermfreq, ctxt_min_freq=opts.ctxtfreq, pterm_pos=opts.ptermpos, pterm_cpos=opts.ptermcpos, pterm_use_lem=opts.ptermuselem, sterm_pos=opts.stermpos, sterm_cpos=opts.stermcpos, sterm_use_lem=opts.stermuselem, skip_pos=opts.skippos, skip_cpos=opts.skipcpos, skip_use_lem=opts.skipuselem, skip_only=opts.skiponly, use_deplabel=opts.usedeplabel, weight_fun=opts.weightfun)

    if opts.disthpos:
        for pos in opts.disthpos.split(","):
            if not opts.fromsim:
                dse.contexts_to_simmatrix(opts.outname+".ctxt.pkl", opts.outname+"."+pos+".sim", measure_fun=opts.measurefun, pterm_pos=pos)
            dse.simmatrix_to_clusters(opts.outname+".ctxt.pkl", opts.outname+"."+pos+".sim", opts.outname+"."+pos+".clust", pterm_pos=pos, vocab_cutoffs=opts.vocabcutoffs)
            dse.simmatrix_to_disthes(opts.outname+".ctxt.pkl", opts.outname+"."+pos+".sim", opts.outname+"."+pos+".disth", pterm_pos=pos, neigh_cutoffs=opts.neighcutoffs, pterm_wgts=opts.ptermwgts)
