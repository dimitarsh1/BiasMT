#!/home/dimitar/anaconda3/envs/lex_div/bin/python
# -*- coding: utf-8 -*-

import codecs
import statistics
import argparse
import os
import numpy as np
from scipy.stats import ttest_ind
from mosestokenizer import *
from biasmt_metrics import *
import sys
import time

def main():
    ''' main function '''
    # read argument - file with data
    parser = argparse.ArgumentParser(description='Computes the lexical profile.')
    parser.add_argument('-f', '--files', required=True, help='the files to read (min 1).', nargs='+')

    args = parser.parse_args()

    sentences = {}

    settings = {'1000-2000-rest': (1000, 2000), '100-to-2000-rest': (100, 2000), '500-to-2000-rest': (500, 2000)}
    metrics_bs = {}

    # 1. read all the file
    for textfile in args.files:
        system = os.path.splitext(os.path.basename(textfile))[0]
        sentences[system] = []

        with codecs.open(textfile, 'r', 'utf8') as ifh:
            sentences[system] = [s.strip() for s in ifh.readlines() if s.strip()] # ! Spacy UDPIPE crashes if we keep also empty lines

    # 2. Compute overall metrics
    for syst in sentences:
        for sett in settings:
            (step, last) = settings[sett]
            a = time.time()
            print(syst, end=": ")
            score = textToLFP(sentences[syst], step, last)
            print("Step: " + str(step) + " Last: " + str(last))
            print(" & ".join([str(s) for s in score]))

    sys.exit("Done")

if __name__=="__main__":
    main()
