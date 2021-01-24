#!/bin/bash
A=$1 #this is the language pair
LANG=`echo ${A} | cut -d '-' -f 2`
echo $LANG

mkdir results_shannon -p

# shannon_pairwise.py -f train-${A}-ORGNL.tok -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_train-${A}_ORGNL.out
#shannon_pairwise.py -f train-${A}-RBMT.unk.tok.onl -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_train-${A}_RBMT_UNKN.out
# shannon_pairwise.py -f train-${A}-SMT-UNKN.out.tok -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_train-${A}_SMT_UNKN.out
# shannon_pairwise.py -f train-${A}-LSTM-BPE.out.tok.nobpe -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_train-${A}_LSTM_NOBPE.out
# shannon_pairwise.py -f train-${A}-TRANS-BPE.out.tok.nobpe -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_train-${A}_TRANS_NOBPE.out

# shannon_pairwise.py -f train-${A}-SMT-NODUP-UNKN.back.out.tok -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_back_train-${A}_SMT_UNKN.out
# shannon_pairwise.py -f train-${A}-LSTM-BPE.back.out.tok.nobpe -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_back_train-${A}_LSTM_NOBPE.out
# shannon_pairwise.py -f train-${A}-TRANS-BPE.back.out.tok.nobpe -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_back_train-${A}_TRANS_NOBPE.out

shannon_pairwise.py -f ${A}-test.tok.ref.tok -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_test-${A}_REF.out
shannon_pairwise.py -f test-${A}-RBMT.unk.tok -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_test-${A}_RBMT.out
shannon_pairwise.py -f test-${A}-SMT-NOUNK.out.final.tok -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_test-${A}_SMT.out
shannon_pairwise.py -f test-${A}-LSTM-BPE.out.final.tok -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_test-${A}_LSTM.out
shannon_pairwise.py -f test-${A}-TRANS-BPE.out.final.tok -l $LANG -v train-${A}-ORGNL.freq_voc -t 0 > results_shannon/all_mostfrequent_test-${A}_TRANS.out
