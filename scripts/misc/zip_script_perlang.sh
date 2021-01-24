#!/bin/bash
A=$1 #this is the language pair
LANG=`echo ${A} | cut -d '-' -f 2`
echo $LANG

mkdir results_zip -p

zip results_zip/${A}_data.zip train-${A}-ORGNL.tok \
    train-${A}-RBMT.unk.tok.onl \
    train-${A}-RBMT.unk.tok.noerr \
    train-${A}-SMT-UNKN.out.tok \
    train-${A}-LSTM-BPE.out.tok.nobpe \
    train-${A}-TRANS-BPE.out.tok.nobpe \
    train-${A}-SMT-NODUP-UNKN.back.out.tok \
    train-${A}-LSTM-BPE.back.out.tok.nobpe \
    train-${A}-TRANS-BPE.back.out.tok.nobpe
