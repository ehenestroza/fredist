#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Enrique Henestroza Anguiano
#

"""
PP attachment module. Supports calculation of pp subcat and preference scores,
and correction of a predicted parse using a simple score-based model.
"""

import sys
import codecs
import assoc
import cPickle
from dtbutils import *
from math import log

class PPAttachment(object):

    def __init__(self, param={'verbose':False, 'neigh':'dist-3',\
                              'type':'selpref'}):
        self._param = param

        # Make an ipos list (ignored gov FPS), and fpos list (accepted gov FPS)
        self._fpos = set(['NC','NPP','V','VINF','VIMP','VPP','VPR','VS'])
        self._ipos = set(INTERPOS.keys()) - self._fpos

        # Load list of accepted prepositions
        f = codecs.open(PREPLIST, "r", ENCODING)
        preps = []
        for line in f:
            preps.append(pretreat_lem(line.rstrip(), "P"))
        f.close()
        self._preps = set(preps)


    #
    # Given a CONLL stream, yield pp-attachment scores. Neighborhood
    # is either "binary" (v,p,n,o), "dist-X".
    # Score(gov, pp) = # gov->pp / # gov in neigh(pp)
    #
    def pp_scores(self, infile, pkfile, gov_cutoff=50, pp_cutoff=1000,\
                  score_type="nrf"):
        assoctype = self._param['type']

        instream = codecs.open(infile, "r", ENCODING)
        pkstream = open(pkfile, "wb")

        pp_cnt = {} # pp -> cnt
        if score_type == "pmi":
            gov_cnt = {} # gov -> cnt
        govofpp_cnt = {} # pp -> gov -> (attach_count, neigh_count)
        tot_cnt = 0

        # Count data from input file.
        for _, sent in read_conll(instream, mode="extract"):
            for did in range(1, len(sent)):
                d = sent[did]
                pp, govs = self.pp_neighbors(sent, did)
                if pp and govs:
                    pp_cnt[pp] = pp_cnt.get(pp, 0) + 1
                    cur_govofpp_cnt = govofpp_cnt.get(pp, {})
                    for gid, gov in govs:
                        att_cnt, neigh_cnt = cur_govofpp_cnt.get(gov, (0,0))
                        if d[GOV] == gid:
                            if score_type == "pmi":
                                gov_cnt[gov] = gov_cnt.get(gov, 0) + 1
                            tot_cnt += 1
                            att_cnt += 1
                        neigh_cnt += 1
                        cur_govofpp_cnt[gov] = att_cnt, neigh_cnt
                    govofpp_cnt[pp] = cur_govofpp_cnt
        instream.close()

        # Do count cutoffs, generalise low-frequency lemmas with <UNK>
        for pp in pp_cnt.keys():
            # Group together low-freq pps by object lemma.
            pcnt = pp_cnt[pp]
            if pcnt < pp_cutoff and assoctype == "selpref":
                unkpp = pp[0:3]+("<UNK>",)
                pp_cnt[unkpp] = pp_cnt.get(unkpp, 0) + pcnt
                del pp_cnt[pp]
                govofpp_cnt[unkpp] = govofpp_cnt.get(unkpp, {})
                for gov,(acnt,ncnt) in govofpp_cnt[pp].items():
                    unkacnt,unkncnt = govofpp_cnt[unkpp].get(gov, (0,0))
                    govofpp_cnt[unkpp][gov] = acnt+unkacnt,ncnt+unkncnt
                del govofpp_cnt[pp]
        for pp in pp_cnt.keys():
            # Group together low-freq pps by prep lemma.
            pcnt = pp_cnt[pp]
            if pcnt < pp_cutoff:
                unkpp = pp[0],"<UNK>",pp[2],"<UNK>"
                pp_cnt[unkpp] = pp_cnt.get(unkpp, 0) + pcnt
                del pp_cnt[pp]
                govofpp_cnt[unkpp] = govofpp_cnt.get(unkpp, {})
                for gov,(acnt,ncnt) in govofpp_cnt[pp].items():
                    unkacnt,unkncnt = govofpp_cnt[unkpp].get(gov, (0,0))
                    govofpp_cnt[unkpp][gov] = acnt+unkacnt,ncnt+unkncnt
                del govofpp_cnt[pp]
        for pp in pp_cnt.keys():
            cur_govofpp_cnt = govofpp_cnt[pp]
            # Group together low-freq govs.
            for gov,(acnt,ncnt) in cur_govofpp_cnt.items():
                gcnt = gov_cnt.get(gov, 0) if score_type == "pmi" else ncnt
                if gcnt < gov_cutoff:
                    unkgov = gov[:-1]+("<UNK>",)
                    unkacnt,unkncnt = cur_govofpp_cnt.get(unkgov, (0,0))
                    cur_govofpp_cnt[unkgov] = acnt+unkacnt,ncnt+unkncnt
                    del cur_govofpp_cnt[gov]
        if score_type == "pmi":
            for gov in gov_cnt.keys():
                # Group together low-freq govs.
                gcnt = gov_cnt[gov]
                if gcnt < gov_cutoff:
                    unkgov = gov[:-1]+("<UNK>",)
                    gov_cnt[unkgov] = gov_cnt.get(unkgov, 0) + gcnt
                    del gov_cnt[gov]

        # Create alphabet, calculate scores.
        pp_alpha = {}
        id_to_pp = []
        gov_alpha = {}
        id_to_gov = []
        pp_id_cnt = 0
        gov_id_cnt = 0

        if score_type == "pmi":
            min_pmi = sys.maxint
            max_pmi = -sys.maxint
            for pp in pp_cnt.keys():
                pcnt = pp_cnt[pp]
                pp_alpha[pp] = pp_id_cnt
                id_to_pp.append(pp)
                pp_id_cnt += 1
                cur_govofpp_cnt = govofpp_cnt[pp]
                for gov in cur_govofpp_cnt.keys():
                    gcnt = gov_cnt.get(gov, 0)
                    acnt,_ = cur_govofpp_cnt[gov]
                    if acnt > 0:
                        pmi = log((1.0*tot_cnt*acnt)/(1.0*pcnt*gcnt),2)
                        if pmi > max_pmi:
                            max_pmi = pmi
                        if pmi < min_pmi:
                            min_pmi = pmi
                        cur_govofpp_cnt[gov] = pmi
                    else:
                        del cur_govofpp_cnt[gov]
                del pp_cnt[pp]
            for gov in gov_cnt.keys():
                if gov not in gov_alpha:
                    gov_alpha[gov] = gov_id_cnt
                    id_to_gov.append(gov)
                    gov_id_cnt += 1

            print >> sys.stderr, "# PPs: ", len(id_to_pp)
            print >> sys.stderr, "# GOVs: ", len(id_to_gov)
            cPickle.dump(pp_alpha, pkstream, -1)
            cPickle.dump(gov_alpha, pkstream, -1)

            for ppid in xrange(len(id_to_pp)):
                cur_govofpp_cnt = govofpp_cnt[id_to_pp[ppid]]
                curvector = [None]*len(id_to_gov)
                for gid in xrange(len(id_to_gov)):
                    gov = id_to_gov[gid]
                    pmi = cur_govofpp_cnt.get(gov, min_pmi)
                    if pmi < 0:
                        pmi = -pmi / min_pmi
                    elif pmi > 0:
                        pmi = pmi / max_pmi
                    curvector[gid] = pmi
                cPickle.dump(curvector, pkstream, -1)
            pkstream.close()

        elif score_type == "nrf":
            for pp in pp_cnt.keys():
                pp_alpha[pp] = pp_id_cnt
                id_to_pp.append(pp)
                pp_id_cnt += 1
                cur_govofpp_cnt = govofpp_cnt[pp]
                for gov,(acnt,ncnt) in cur_govofpp_cnt.items():
                    if gov not in gov_alpha:
                        gov_alpha[gov] = gov_id_cnt
                        id_to_gov.append(gov)
                        gov_id_cnt += 1
                    cur_govofpp_cnt[gov] = float(acnt)/ncnt
                del pp_cnt[pp]
        
            print >> sys.stderr, "# PPs: ", len(id_to_pp)
            print >> sys.stderr, "# GOVs: ", len(id_to_gov)
            cPickle.dump(pp_alpha, pkstream, -1)
            cPickle.dump(gov_alpha, pkstream, -1)
            
            for ppid in xrange(len(id_to_pp)):
                cur_govofpp_cnt = govofpp_cnt[id_to_pp[ppid]]
                curvector = [None]*len(id_to_gov)
                for gov in cur_govofpp_cnt:
                    curvector[gov_alpha[gov]] = cur_govofpp_cnt[gov]
                cPickle.dump(curvector, pkstream, -1)
            pkstream.close()


    #
    # Given a subcat file, yield pp-attachment scores. File format line:
    # verblemma       preplemma       objpos
    def pp_scores_from_file(self, infile, pkfile):
        
        if self._param['type'] != 'subcat':
            print >> sys.stderr, \
            'Function pp_scores_from_file() only supports subcat association.'

        instream = codecs.open(infile, "r", ENCODING)
        pkstream = open(pkfile, "wb")

        pp_alpha = {}
        id_to_pp = []
        pp_id_cnt = 0
        gov_alpha = {}
        id_to_gov = []
        gov_id_cnt = 0
        gov_of_pp = {}
        for line in instream:
            vlem,plem,opos = line.rstrip().split('\t')
            pp = 'P', plem, opos, '<'+opos+'>'
            gov = 'V', vlem
            if pp not in pp_alpha:
                pp_alpha[pp] = pp_id_cnt
                id_to_pp.append(pp)
                gov_of_pp[pp] = {}
                pp_id_cnt += 1
            if gov not in gov_alpha:
                gov_alpha[gov] = gov_id_cnt
                id_to_gov.append(gov)
                gov_id_cnt += 1
            gov_of_pp[pp][gov] = True
        instream.close()

        print >> sys.stderr, "# PPs: ", len(id_to_pp)
        print >> sys.stderr, "# GOVs: ", len(id_to_gov)
        cPickle.dump(pp_alpha, pkstream, -1)
        cPickle.dump(gov_alpha, pkstream, -1)

        for ppid in xrange(len(id_to_pp)):
            cur_govofpp = gov_of_pp[id_to_pp[ppid]]
            curvector = [None]*len(id_to_gov)
            for gid in xrange(len(id_to_gov)):
                gov = id_to_gov[gid]
                score = 1 if gov in cur_govofpp else 0
                curvector[gid] = score
            cPickle.dump(curvector, pkstream, -1)
        pkstream.close()

    #
    # Function for outside calls to obtain a properly formed pp and candidate
    # governors (or None if criteria are not met).
    #
    def format_pp_gov(self, sent, did=None, cid=None):
        preps = self._preps
        atype = self._param['type']
        fpos = self._fpos

        pp = None
        if did != None:
            # Check for correct type of pos/lem of prep.
            if sent[did][CPS] not in ["P", "P+D"] or \
                   sent[did][LEM] not in preps:
                return None,None
            d = sent[did]
            d_o = sent[d[OBJ]] if d[OBJ] else False
            if not d_o or d_o[FPS] not in ["NC", "NPP", "VINF"]:
                return None,None
            if d_o[FPS] in ["VINF"]: #, "NPP"]:
                d_o_lem = "<"+d_o[FPS]+">"
            else:
                d_o_lem = d_o[LEM]
            if atype == "selpref":
                pp = "P", d[LEM], d_o[CPS], d_o_lem
            elif atype in ["subcat", "pospref"]:
                pp = "P", d[LEM], d_o[CPS], "<"+d_o[CPS]+">"

        gov = None
        if cid != None:
            if cid == 0:
                return None,None
            # Check for correct type of candidate.
            if did != None and cid > did or \
                   fpos and sent[cid][FPS] not in fpos:
                return None,None
            cand = sent[cid]
            cand_lem = cand[LEM]
#            if cand[FPS] in ["NPP"]:
#                cand_lem = "<"+cand[FPS]+">"
#            else:
#                cand_lem = cand[LEM]
            if atype in ["selpref", "subcat"]:
                gov = cand[CPS], cand_lem
            elif atype == "pospref":
                gov = cand[CPS], "<"+cand[CPS]+">"

        return pp, gov


    #
    # Subroutine that, given a dependent id and a sentence, returns a tuple
    # (pp, ((gid1, gov1), (gid2, gov2), ...)) corresponding to a neighborhood
    # around the dependent and conforming to the assoctype. None if failed.
    # 
    def pp_neighbors(self, sent, did):
        preps = self._preps
        neigh = self._param['neigh']
        atype = self._param['type']
        ipos = self._ipos
        
        pp,_ = self.format_pp_gov(sent, did=did, cid=sent[did][GOV])

        if not pp:
            return None,None

        # Get list of cand governors (first item is the predicted one).
        cands = neighborhood(sent, did, neigh=neigh, ipos=ipos)

        if not cands:
            return None,None

        govs = []
        for cid,_,_ in cands:
            if cid == 0 or cid > did:
                continue
            _,gov = self.format_pp_gov(sent, did=None, cid=cid)
            if gov:
                govs.append((cid, gov))
        return pp, tuple(govs)


    #
    # Given a CONLL file, yield pp-attachment examples. Neighborhood
    # is either "binary" (v,p,n,o), "dist2", or "dist3". Account for ALL
    # gold-pos prepositions, so when evaluating we have preposition UAS.
    #
    def pp_examples(self, predfile, goldfile):
        
        predstream = codecs.open(predfile, "r", ENCODING)
        goldstream = codecs.open(goldfile, "r", ENCODING)
        prediter = read_conll(predstream, mode="extract")
        golditer = read_conll(goldstream, mode="extract")
        puas = 0
        ouas = 0
        tot = 0
        while True:
            try:
                osent, sent = prediter.next()
                _, goldsent = golditer.next()
            except StopIteration:
                break

            # Create pp-attachment examples. Account for ALL gold-pos preps!
            for did in range(1, len(sent)):
                
                if goldsent[did][CPS] not in ["P", "P+D"] or \
                       is_punct(goldsent[did][TOK]):
                    #print "\t".join(osent[did-1]).encode(ENCODING)
                    continue
            
                ex = ((True,None,None),) \
                     if goldsent[did][GOV] == sent[did][GOV] else \
                     ((False,None,None),)

                pp, govs = self.pp_neighbors(sent, did)
                if pp and govs:
                    ex = []
                    osent[did-1][GOV] = []
                    for gid, gov in govs:
                        c_class = True if gid == goldsent[did][GOV] else False
                        ex.append((c_class, pp, gov))
                        osent[did-1][GOV].append(str(gid))
                    osent[did-1][GOV] = ",".join(osent[did-1][GOV])
                    ex = tuple(ex)

                # Quick counts.
                if ex[0][0]:
                    puas += 1
                for cand in ex:
                    if cand[0]:
                        ouas += 1
                tot += 1

                #print "\t".join(osent[did-1]).encode(ENCODING)
                yield ex

            #print ""

        # Quick eval.
        print >> sys.stderr, "PUAS: %(puas)02.02f%%, OUAS: %(ouas)02.02f%%" \
              %{'ouas':100.0*ouas/tot, 'puas':100.0*puas/tot}
        
        predstream.close()
        goldstream.close()


if __name__ == "__main__":

    import sys
    import optparse

    parser = optparse.OptionParser()
    parser.add_option("-o", "--gold", \
                      action="store", \
                      default='', \
                      help="read gold parse data from CONLL file")
    parser.add_option("-t", "--train", \
                      action="store", \
                      default='', \
                      help="read parse data from CONLL file to create scores.")
    parser.add_option("-p", "--predict", \
                      action="store", \
                      default='', \
                      help="read predicted parse data from CONLL file")
    parser.add_option("-f", "--fromfile", \
                      action="store_true", \
                      default=False, \
                      help="Read in a preexisting subcat file.")
    parser.add_option("-v", "--verbose", \
                      action="store_true", \
                      default=False, \
                      help="Output transition sequences during parsing")
    parser.add_option("-s", "--ppscores", \
                      action="store", \
                      default='', \
                      help="Association score file (either create or use).")
    parser.add_option("-e", "--ppeval", \
                      default='', \
                      help="Generate/load PP-evaluation set file.")
    parser.add_option("-n", "--neighbor", \
                      default='binary', \
                      help="Neighborhood type to use around dependents.")
    parser.add_option("-a", "--assoctype", \
                      choices=['subcat', 'selpref', 'pospref'], \
                      default='subcat', \
                      help="Either subcat, selpref or pospref association.")
    parser.add_option("-c", "--scoretype", \
                      choices=['pmi', 'nrf'], \
                      default='nrf', \
                      help="Either pmi or nrf calculation.")
    parser.add_option("-b", "--beta", \
                      action="store", \
                      default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45, \
                      0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0", \
                      help="Multiplicative biases for predicted governor.")
    parser.add_option("-d", "--cutdep", \
                      action="store", \
                      type="int", \
                      default=1000, \
                      help="Frequency cutoff for deps (pps) in corpus.")
    parser.add_option("-g", "--cutgov", \
                      action="store", \
                      type="int", \
                      default=50, \
                      help="Frequency cutoff for gov in corpus.")
    (opts, args) = parser.parse_args()
    
    param  = {'verbose':opts.verbose, 'neigh':opts.neighbor, \
              'type':opts.assoctype}
    
    ppattach = PPAttachment(param)

    if opts.train and opts.ppscores:
        print >> sys.stderr, "***GENERATING PP-ATTACHMENT SCORES***"
        if opts.fromfile:
            ppattach.pp_scores_from_file(opts.train, opts.ppscores)
        else:
            ppattach.pp_scores(opts.train, opts.ppscores, \
                               gov_cutoff=opts.cutgov, pp_cutoff=opts.cutdep, \
                               score_type=opts.scoretype)
        print >> sys.stderr, "***SCORES CREATED***"

    elif opts.ppscores and opts.ppeval and opts.predict:
        print >> sys.stderr, "***CORRECTING PP-ATTACHMENT IN CONLL***"
        a = assoc.AssociationScores()
        a._defscore = -1.0 if opts.scoretype == 'pmi' else 0.0
        a.load_scores(opts.ppscores)
        a.load_examples(opts.ppeval)
        a.tune_beta(map(float, opts.beta.split(",")), 3)
        print >> sys.stderr, a._x_red
        predstream = codecs.open(opts.predict, "r", ENCODING)
        for osent,sent in read_conll(predstream, mode="extract"):
            for did in range(1, len(sent)):
                pp, govs = ppattach.pp_neighbors(sent, did)
                if pp and govs:
                    ex = [] # lab,pp,gov
                    for gid, gov in govs:
                        ex.append((gid, pp, gov))
                    cid = ex[a.choose_cand(ex)[1]][0]
                    osent[did-1][GOV] = str(cid)
                print "\t".join(osent[did-1]).encode(ENCODING)
            print ""
        print >> sys.stderr, "***FINISHED***"

    elif opts.ppeval and opts.predict and opts.gold:
        print >> sys.stderr, "***GENERATING PP-ATTACHMENT EVAL EXAMPLES***"
        pkstream = open(opts.ppeval, "wb")
        exes = tuple(ppattach.pp_examples(opts.predict, opts.gold))
        cPickle.dump(exes, pkstream, -1)
        pkstream.close()
        print >> sys.stderr, "***EVAL EXAMPLES CREATED***"

