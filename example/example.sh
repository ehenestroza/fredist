#
# An example script showing how to train and test an example CONLL treebank.
#

# For training a baseline arc-eager parser on FTB training set:
./create_tasklist_dtb.sh 'train' 'parser' 'eager' 'ADJ-ADJWH_ADV-ADVWH_CC-CS_CLO-CLR-CLS_DET-DETWH_ET_I_NC-NPP_P-P+D-P+PRO_PONCT_PREF_PRO-PROREL-PROWH_V-VIMP-VINF-VPP-VPR-VS' 'default' 'none' '1' '1' example_train example_dev | sh


# For testing that baseline arc-eager parser on FTB test set:
./create_tasklist_dtb.sh 'test' 'parser' 'eager' 'ADJ-ADJWH_ADV-ADVWH_CC-CS_CLO-CLR-CLS_DET-DETWH_ET_I_NC-NPP_P-P+D-P+PRO_PONCT_PREF_PRO-PROREL-PROWH_V-VIMP-VINF-VPP-VPR-VS' 'default' 'none' '1' example_train example_test example_test 'base' | sh

# The resulting evaluation file contains the accuracy of the baseline parser:
cat example_test.pred-parser.i-default.g-nolexgen.r-eager.base.eval
