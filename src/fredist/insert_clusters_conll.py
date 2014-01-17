#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#
# Enrique Henestroza Anguiano :
#
# Fill the of the 9th column with lemma soft-clusters.
#

import sys
import codecs
from dtbutils import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--conllfile", default="", help='')
parser.add_option("--clustfile", default="", help='')
parser.add_option("--pretreat", action="store_true", default=False, help='')
parser.add_option("--cpos", default="", help='')
parser.add_option("--firstonly", action="store_true", default=False, help='')
parser.add_option("--unk", action="store_true", default=False, help='')
parser.add_option("--disth", action="store_true", default=False, help='')
parser.add_option("--clustnpp", action="store_true", default=False, help='')
parser.add_option("--wordformfile", default="", help='')
(opts, args) = parser.parse_args()
cposset = opts.cpos.split("_") if opts.cpos else []

clustf = codecs.open(opts.clustfile, 'r', 'utf-8')
lem_to_clusts = {}
for wline in clustf:
    pterm,clusts = wline.rstrip().split("\t")
    cpos,lem = pterm.split("|")
    if cposset and cpos not in cposset:
        continue
    clusts = [tuple(x.split(":")) for x in clusts.split(" ")]
    if opts.disth:
        lem_to_clusts[(cpos,lem)] = [(x.split('|')[1],y) for x,y in clusts]
    else:
        if opts.firstonly:
            clusts = [(clusts[0][0], "1.0")]
        lem_to_clusts[(cpos,lem)] = [('CL_'+cpos+'_'+x,y) \
                                     for x,y in clusts]

clustf.close()

conllf = codecs.open(opts.conllfile, 'r', 'utf-8')
wff = codecs.open(opts.wordformfile, 'w', 'utf-8') if opts.wordformfile else 0
wfdict = {}
for cline in conllf:
    if cline.rstrip() == "":
        if not wff:
            print ""
        continue
    cfields = cline.rstrip().split('\t')[0:10]
    if not cposset or cfields[CPS] in cposset:
        if opts.pretreat:
            c_lem = pretreat_lem(cfields[LEM], cfields[CPS])
        else:
            c_lem = cfields[LEM]

        clusts = []
        if cfields[FPS] == "NPP" and opts.clustnpp:
            clusts = [('CL_NPP', "1.0")]
        elif not (c_lem in STOP and cfields[CPS] == "V"):
            clusts = lem_to_clusts.get((cfields[CPS], c_lem), [])
            if opts.unk and not clusts:
                if opts.disth:
                    clusts = lem_to_clusts.get((cfields[CPS], "<UNK>"), [])
                else:
                    clusts = [('CL_UNK_'+cfields[CPS], "1.0")]
        if wff:
            if clusts:
                wfdict[cfields[TOK]] = clusts[0][0]
            continue
        if not clusts:
            clusts = [('CL_ORIG_'+cfields[CPS]+'_'+c_lem, "1.0")]
        cfields[8] = "|".join(["=".join(x) for x in clusts])
    if not wff:
        print "\t".join(cfields).encode('utf-8')
conllf.close()

if wff:
    for k,v in sorted(wfdict.items()):
        wff.write(k+"\t"+v+"\n")
    wff.close()
