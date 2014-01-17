#!/bin/bash

#
# Creates a set of commands for FreDist to perform training and testing of 
# transition-based parsing and neighborhood parse correction, using the CONLL
# format for input and output files. Includes options for the integration of a
# "lexical generalization" resource, which defines word classes with which to
# replace lemmas, or a "lexical preference" resource, which provides scores
# between possible governor/dependent pairs (specifically for PP-attachment)
# that are added as supplemental features in the models.
#
# - Requires that the $FREDIST variable be the location of source .py files.
#

PYTHON='python' # change if using a different version of python.
mode=$1 # xval, train, oracle, test, parse.
ptype=$2 # parser, corrector.
r=$3 # standard, eager, eagermc, none (for correction).
divfpos=$4 # ADJ-ADJWH_ADV-ADVWH_CC-CS_CLO-CLR-CLS_, etc.
i=$5 # default.
lexgen=$6 # filename for lexgen, 'none' for none.
lexgenk=$7 # 0 (no limit), 1, 2, etc.
if [ $lexgen = 'none' ]
    then
    x=""
    suff=i-$i.g-nolexgen.r-$r
else
    x="-x $lexgen -a $lexgenk"
    suff=i-$i.g-lexgen_${lexgenk}.r-$r
fi
dtbparser="$FREDIST/dtbparser.py"

#lab="'aff:a_obj:arg:ato:ats:aux_caus:aux_pass:aux_tps:comp:coord:de_obj:dep:dep_coord:det:mod:mod_rel:obj:p_obj:ponct:root:suj'"

if [ $mode = 'xval' ]
    then
    if [ $ptype = 'parser' ]
	then
	c=${8}
	k=${9}
	CORPUSTRAIN=${10}
	CORPUSGOLD=${11}

        # Xval/Jack-knifing Parser
	echo -n " $PYTHON $dtbparser $x -c $c -f '"$divfpos"' -y $ptype -r $r -k $k -i $i -g $CORPUSTRAIN.conll -t $CORPUSTRAIN.conll -p $CORPUSTRAIN.pred-${ptype}.$suff.conll -m $CORPUSTRAIN.model-${ptype}-xval.$suff ; "

        # Evaluation
	echo -n "./eval07p.pl -p -g $CORPUSGOLD.conll -s $CORPUSTRAIN.pred-${ptype}.$suff.conll > $CORPUSTRAIN.pred-${ptype}.$suff.eval ; "
	echo -n "./eval07p.pl -q -b -p -g $CORPUSGOLD.conll -s $CORPUSTRAIN.pred-${ptype}.$suff.conll > $CORPUSTRAIN.pred-${ptype}.$suff.evalb ; "
	echo ""
    fi

elif [ $mode = 'train' ]
    then
    c=${8}
    CORPUSTRAIN=${9}
    gold=''
    dev=''
    devg=''
    subcat=''
    selpref=''
    if [ $ptype = 'parser' ]
	then
	dev="-d ${10}.conll"
    fi
    if [ $ptype = 'corrector' ]
	then
	gold="-g ${10}.conll"
	dev="-d ${11}.conll"
	devg="-e ${12}.conll"
	if [ ${13} != 'none' ] || [ ${14} != 'none' ]
	    then
	    assocsuff=''
	    if [ ${13} = 'none' ]
		then
		subcat=''
	    else
		subcat="-u ${13}.pkl"
		assocsuff="${assocsuff}_subcat"
	    fi
	    if [ ${14} = 'none' ]
		then
		selpref=''
	    else
		selpref="-s ${14}.pkl"
		assocsuff="${assocsuff}_selpref"
	    fi
	    suff="$suff.a-assoc$assocsuff"
	fi
    fi

    # Training Parser or Corrector
    echo -n " $PYTHON $dtbparser $subcat $selpref $x -c $c -f '"$divfpos"' $gold $dev $devg -y $ptype -r $r -i $i -t $CORPUSTRAIN.conll -m $CORPUSTRAIN.model-${ptype}.$suff --diagnostics $CORPUSTRAIN.model-${ptype}.$suff.diag ; "
    echo ""
    
elif [ $mode = 'oracle' ]
    then
    if [ $ptype = 'corrector' ]
	then
	CORPUSTEST=${8}
	CORPUSTESTG=${9}
	suff=oracle

        # Correcting
	echo -n " $PYTHON $dtbparser $x -f '"$divfpos"' -y $ptype -r $r -i $i -p $CORPUSTEST.conll -g $CORPUSTESTG.conll --diagnostics $CORPUSTEST.pred-${ptype}.$suff.diag > $CORPUSTEST.pred-${ptype}.$suff.conll ; "

        # Evaluation
	echo -n "./eval07p.pl -p -g $CORPUSTESTG.conll -s $CORPUSTEST.pred-${ptype}.$suff.conll > $CORPUSTEST.pred-${ptype}.$suff.eval ; "
	echo -n "./eval07p.pl -q -b -p -g $CORPUSTESTG.conll -s $CORPUSTEST.pred-${ptype}.$suff.conll > $CORPUSTEST.pred-${ptype}.$suff.evalb ; "

	echo ""
    fi

elif [ $mode = 'test' ]
    then
    CORPUSTRAIN=${8}
    CORPUSTEST=${9}
    CORPUSTESTG=${10}
    IDX=${11}
    subcat=''
    selpref=''
    if [ $ptype = 'corrector' ]
	then
	if [ ${12} != 'none' ] || [ ${13} != 'none' ]
	    then
	    assocsuff=''
	    if [ ${12} = 'none' ]
		then
		subcat=''
	    else
		subcat="-u ${12}.pkl"
		assocsuff="${assocsuff}_subcat"
	    fi
	    if [ ${13} = 'none' ]
		then
		selpref=''
	    else
		selpref="-s ${13}.pkl"
		assocsuff="${assocsuff}_selpref"
	    fi
	    suff="$suff.a-assoc$assocsuff"
	fi
    fi

    # Parsing or Correcting
    echo -n " $PYTHON $dtbparser $subcat $selpref $x -f '"$divfpos"' -y $ptype -r $r -i $i -p $CORPUSTEST.conll -m $CORPUSTRAIN.model-${ptype}.$suff --diagnostics $CORPUSTEST.pred-${ptype}.$suff.diag > $CORPUSTEST.pred-${ptype}.$suff.$IDX.conll ; "

    # Evaluation
    echo -n "./eval07p.pl -p -g $CORPUSTESTG.conll -s $CORPUSTEST.pred-${ptype}.$suff.$IDX.conll > $CORPUSTEST.pred-${ptype}.$suff.$IDX.eval ; "
    echo -n "./eval07p.pl -q -b -p -g $CORPUSTESTG.conll -s $CORPUSTEST.pred-${ptype}.$suff.$IDX.conll > $CORPUSTEST.pred-${ptype}.$suff.$IDX.evalb ; "

    echo ""

elif [ $mode = 'parse' ]
    then
    CORPUSTRAIN=${8}
    CORPUSTEST=${9}
    IDX=${10}
    subcat=''
    selpref=''
    if [ $ptype = 'corrector' ]
	then
	if [ ${11} != 'none' ] || [ ${12} != 'none' ]
	    then
	    assocsuff=''
	    if [ ${11} = 'none' ]
		then
		subcat=''
	    else
		subcat="-u ${11}.pkl"
		assocsuff="${assocsuff}_subcat"
	    fi
	    if [ ${12} = 'none' ]
		then
		selpref=''
	    else
		selpref="-s ${12}.pkl"
		assocsuff="${assocsuff}_selpref"
	    fi
	    suff="$suff.a-assoc$assocsuff"
	fi
    fi

    # Parsing or Correcting
    echo -n " $PYTHON $dtbparser $subcat $selpref $x -f '"$divfpos"' -y $ptype -r $r -i $i -p $CORPUSTEST.conll -m $CORPUSTRAIN.model-${ptype}.$suff --diagnostics $CORPUSTEST.pred-${ptype}.$suff.diag > $CORPUSTEST.pred-${ptype}.$suff.$IDX.conll ; "

    echo ""

fi
