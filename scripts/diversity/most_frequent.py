#!/home/dimitar/anaconda3/envs/lex_div/bin/python
# -*- coding: utf-8 -*-

import codecs
import argparse
import os
import sys
from nltk.probability import FreqDist

def main():
    ''' main function '''
    # read argument - file with data
    parser = argparse.ArgumentParser(description='Computes the lexical profile.')
    parser.add_argument('-f', '--files', required=True, help='the files to read (min 1).', nargs='+')

    args = parser.parse_args()

    sentences = {}

    metrics_bs = {}

    # 1. read all the file
    for textfile in args.files:
        system = os.path.splitext(os.path.basename(textfile))[0]
        sentences[system] = []

        with codecs.open(textfile, 'r', 'utf8') as ifh:
            sentences[system] = [s.strip() for s in ifh.readlines() if s.strip()] # ! Spacy UDPIPE crashes if we keep also empty lines

    # 2. Compute overall metrics
    for syst in sentences:
        #create Frequency Dictionary
        print(syst)
        fdist = FreqDist(" ".join(sentences[syst]).split()) # our text is already tokenized. We merge all sentences together
                                                            # and create one huge list of tokens.

        with open(syst + ".freq_voc", "w") as oF:
            oF.write("\n".join([w + "\t" + str(c) for (w, c) in fdist.most_common()]))
        print("Next")
    sys.exit("Done")

if __name__=="__main__":
    main()
