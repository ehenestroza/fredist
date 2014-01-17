#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#
# Enrique Henestroza Anguiano :
#
# Transform from wordnet 3.0 xml to simple format of the style (per line):
#
# <POS>|<WORD> <SYNSET-ID-1>:<PREVALENCE> <SYNSET-ID-2>:<PREVALENCE> ...
#
# Synset prevalence scores must add up to 1 for each entry.

import sys
import nltk
from nltk.corpus import wordnet, wordnet_ic
import re
import codecs
from dtbutils import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--wnfile", default="", help='WordNet file.')
parser.add_option("--outfile", default="", help='Output file.')
parser.add_option("--encoding", default=ENCODING, help='Encoding.')
parser.add_option("--cpos", default="N", help='cpos to use.')
parser.add_option("--disth", default="", help='thesaurus to use for ASR.')
(opts, args) = parser.parse_args()
enc = opts.encoding
pos_wordnet_to_coarse = {'a': 'A', 'b': 'ADV', 'n': 'N', 'v': 'V'}
pterm_to_synsets = {}

# Wordnet information content information.
ic = wordnet_ic.ic('ic-bnc.dat') 

# Read in wordnet file, creating a dictionary of word -> synset ID list.
wnf = codecs.open(opts.wnfile, "r", enc)
for line in wnf:
    line = line.strip()
    idobj = re.search('<ID>(.*?)</ID>', line)
    posobj = re.search('<POS>(.*?)</POS>', line)
    lits = re.findall('<LITERAL>(.*?)<SENSE>', line)

    if idobj and posobj:
        cpos = pos_wordnet_to_coarse[posobj.group(1)]
        if cpos != opts.cpos:
            continue
        _,offset,idpos = idobj.group(1).split('-')
        ss = wordnet._synset_from_pos_and_offset(idpos, int(offset))

        for lit in lits:
            newlit = lit.strip()
            if not re.search(' ', newlit) and newlit == newlit.lower():
                if newlit.startswith("s'"): # Merge reflexive to base lemma
                    newlit = newlit[2:]
                while newlit.startswith("_"):
                    newlit = newlit[1:]
                # pretreat the literal
                lem = pretreat_lem(newlit, cpos)
                if lem != newlit:
                    continue
                # don't add auxiliary verbs (etre, avoir, etc.)
                if opts.cpos == "V" and lem in STOP:
                    continue

                if (cpos,lem) not in pterm_to_synsets:
                    pterm_to_synsets[(cpos,lem)] = [ss]
                elif ss not in pterm_to_synsets[(cpos,lem)]:
                    pterm_to_synsets[(cpos,lem)].append(ss)
wnf.close()

# Compute the prevalence of each sense for a lemma, using thesaurus.
outf = codecs.open(opts.outfile, "w", enc)
seen_pterms = {}
if opts.disth:
    disthf = codecs.open(opts.disth, "r", ENCODING)

    # Load thesaurus incrementally, to avoid unnecessary memory consumption.
    for line in disthf:
        pterm,neighs = line.rstrip().split("\t")
        pterm = tuple(pterm.split("|"))
        neighs = neighs.split(" ")[1:] # Assumes that pterm is duplicated here.
        sims = []
        for neigh in neighs:
            term,sim = neigh.split(":")
            term = tuple(term.split("|"))
            sim = float(sim)
            sims.append((term,sim))

        sslist = pterm_to_synsets.get(pterm, [])
        if not sslist:
            continue
        if len(sslist) == 1:
            outf.write("|".join(pterm)+"\t"+sslist[0].name+":1.0"+"\n")
            seen_pterms[pterm] = True
            continue

        # Calculate prevalence of each sense.
        prevs = [] # sid -> prev
        tot_prev = 0.0
        for sid in range(len(sslist)):
            prevs.append(0.0)

        for p2term,sim in sims:
            if pterm == p2term:
                continue
            sim_p_p2 = sim
            ss2list = pterm_to_synsets.get(p2term, [])

            # Calculate normalizing wnss
            norm_wnss = 0.0
            for sid in range(len(sslist)):
                max_wnss = 0.0
                for s2id in range(len(ss2list)):
                    s_s2_wnss = wordnet.jcn_similarity(sslist[sid],\
                                                       ss2list[s2id],\
                                                       ic)
                    if s_s2_wnss > max_wnss:
                        max_wnss = s_s2_wnss
                norm_wnss += max_wnss
            if norm_wnss <= 0.0:
                continue
            
            # Increment prevalence score for each sense
            for sid in range(len(sslist)):
                max_wnss = 0.0
                for s2id in range(len(ss2list)):
                    s_s2_wnss = wordnet.jcn_similarity(sslist[sid],\
                                                       ss2list[s2id],\
                                                       ic)
                    if s_s2_wnss > max_wnss:
                        max_wnss = s_s2_wnss
                prev = max_wnss / norm_wnss
                prevs[sid] += prev
                tot_prev += prev

        if tot_prev > 0:
            prevs = sorted([(prev/tot_prev,sid) \
                            for sid,prev in enumerate(prevs)], reverse=True)
            outf.write("|".join(pterm)+"\t")
            for prev,sid in prevs:
                if prev <= 0.0:
                    break
                outf.write(sslist[sid].name+":"+str(prev)+" ")
            outf.write("\n")
            seen_pterms[pterm] = True

    disthf.close()

for pterm,sslist in sorted(pterm_to_synsets.items()):
    if pterm not in seen_pterms:
        outf.write("|".join(pterm)+"\t"+\
                   " ".join([ss.name+":"+str(1.0/len(sslist)) \
                             for ss in sorted(sslist)])+"\n")
outf.close()

