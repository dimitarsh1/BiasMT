#!/usr/bin/python3
# -*- coding: utf-8 -*-

''' reads a text file and exports unique tokens separated by space and their frequencies.
'''
import argparse
import codecs
import os
import itertools
import numpy as np
from lexical_diversity import lex_div as ld
from itertools import combinations
from scipy.stats import ttest_ind
import sacrebleu
from joblib import Parallel, delayed
from mosestokenizer import *
import subprocess

def compute_ter_multeval(sysname, sys, ref, l):
    ''' Getting the ter score of the sample; using external call to multeval)

        :param sysname: the name of the system; to create a directory for sentence-level scores
        :param sys: the sampled sentences from the translation
        :param ref: the reference sentences
        :param l: the langauge for detokenization
        
        :returns: a socre (float)
    '''
    
    rand = str(np.random.randint(100000000))
    tmp_sys_file = 'sys' + rand
    tmp_ref_file = 'ref' + rand
    
    with open(tmp_sys_file, 'w') as of1:
        of1.write('\n'.join(sys))
        
    with open(tmp_ref_file, 'w') as of2:
        of2.write('\n'.join(ref))

    sysname = os.path.join(os.getcwd(), sysname)

    multeval_cmd = ["multeval", "eval", "--refs", os.path.realpath(tmp_ref_file), "--hyps-baseline", os.path.realpath(tmp_sys_file), "--metrics", "ter", "--boot-samples", "1", "--sentLevelDir", sysname, "--threads", "1", "2>", sysname + ".err"]
    stdout =  subprocess.run(' '.join(multeval_cmd), stdout=subprocess.PIPE, shell=True, encoding='utf-8').stdout
    ter_line = stdout.split("\n")[1]
    ter = ter_line.split()[-2]

    os.remove(tmp_sys_file)
    os.remove(tmp_ref_file)    

    try:
        float(ter)
    except:
        print(ter_line)
        print(ter)
        exit(1)

    return float(ter)
    
def compute_bleu_multeval(sysname, sys, ref, l):
    ''' Getting the BLEU score of the sample; using external call to multeval)

        :param sysname: the name of the system; to create a directory for sentence-level scores
        :param sys: the sampled sentences from the translation
        :param ref: the reference sentences
        :param l: the langauge for detokenization
        
        :returns: a socre (float)
    '''
    
    rand = str(np.random.randint(100000000))
    tmp_sys_file = 'sys' + rand
    tmp_ref_file = 'ref' + rand
    
    with open(tmp_sys_file, 'w') as of1:
        of1.write('\n'.join(sys))
        
    with open(tmp_ref_file, 'w') as of2:
        of2.write('\n'.join(ref))

    sysname = os.path.join(os.getcwd(), sysname)

    multeval_cmd = ["multeval", "eval", "--refs", os.path.realpath(tmp_ref_file), "--hyps-baseline", os.path.realpath(tmp_sys_file), "--metrics", "bleu", "--boot-samples", "1", "--sentLevelDir", sysname, "--threads", "1", "2>", sysname + ".err"]
    stdout =  subprocess.run(' '.join(multeval_cmd), stdout=subprocess.PIPE, shell=True, encoding='utf-8').stdout
    bleu_line = stdout.split("\n")[1]
    bleu = bleu_line.split()[-2]

    os.remove(tmp_sys_file)
    os.remove(tmp_ref_file)    

    return float(bleu)

def get_ter(sents, ter, lang):
    ''' Getting the ter score of the sample based on per-sentence ter scores, sentences and overall length)

        :param sents: the sampled sentences (to compute their length)
        :param ter: a ter-score list
        :returns: a socre (float)
    '''
    tot_tokens = 0.0
    tot_errors = 0.0
    for (s, t) in zip(sents, ter):
        ltmp = len(s.split())
        tot_tokens += ltmp
        tot_errors += t*ltmp

    return tot_errors/tot_tokens

def get_bleu(sents, bleu, lang):
    ''' Getting the bleu score of the sample based on per-sentence bleu scores, sentences and overall length)

        :param sents: the sampled sentences (to compute their length)
        :param bleu: a ter-score list
        :returns: a socre (float)
    '''

    bleu_r = 0.0
    for (s, b) in zip(sents, bleu):
        bleu_r += b
    return bleu_r


def get_bleu_sacrebleu(sys, ref, lang):
    ''' Computing BLEU using sacrebleu

        :param sysname: the name of the system
        :param sys: the sampled sentences from the translation
        :param ref: the reference sentences
        :param lang: the langauge for detokenization
        :returns: a socre (float)
    '''
    detokenize = MosesDetokenizer(lang)
    tmp_sys = [detokenize(s.split()) for s in sys]
    tmp_ref = [detokenize(r.split()) for r in ref]
    bleu = sacrebleu.corpus_bleu(tmp_sys, [tmp_ref])
    return bleu.score

def compute_metric(metric_func, sentences, ref, lang, sample_idxs, iters):
    ''' Computing metric

        :param metric_func: get_bleu or get_ter_multeval
        :param sys: the sampled sentences from the translation
        :param ref: the reference sentences
        :param lang: the langauge for detokenization
        :param sample_idxs: indexes for the sample (list)
        :param iters: number of iterations
        :returns: a socre (float)
    '''
    # 5. let's get the measurements for each sample
    scores = {}
    scores = Parallel(n_jobs=8)(delayed(eval(metric_func))([sentences[j] for j in sample_idxs[i]], [ref[j] for j in sample_idxs[i]], lang) for i in range(iters))
             
    return scores
    
    
def compute_significance(metrics, iterations):
    ''' Compute pairwose significance interval

        :param metrics: dictionary with systems and metrics
        :param iterations: the number of iterations
        :returns: a socre (float)
    '''
    # now, we are able to compute statistical significance
    # print('delta(xi) > delta(x):')
    scores = {}
    for system1 in metrics:
        scores[system1] = {}
        for system2 in metrics:
            s = 0.0
            for i in range(iterations):
                if round(metrics[system1][i], 4) > round(metrics[system2][i], 4):
                    s += 1.0
            if system1 == system2:
                scores[system1][system2] = -1.0
            else:
                scores[system1][system2] = s / iterations
    return scores
    

def compute_ttest_scikit(metrics, iterations):
    ''' Compute pairwose significance interval

        :param metrics: dictionary with systems and metrics
        :param iterations: the number of iterations
        :returns: a socre (float)
    '''
    scores = {}
    print('\nScikit ttest:')
    for system1 in metrics:
        scores[system1] = {}
        for system2 in metrics:
            t, p =  ttest_ind(metrics[system1], metrics[system2])
            scores[system1][system2] = p
        
    return scores


def print_latex_table(scores, metric_title):
    ''' Prints a table in latex format; ready to incorporate into a tex file

        :param scores: dictionary with scores and systems
        :param metric_title: identifying the metric (string)
    '''
    print(' '.join([str(s) for s in scores.keys()]))
    for system in scores:
        print(system + ' ' + ' '.join([str(p) for p in scores[system].values()]))

    print('\n')
    print('  & ' + ' & '.join([str(s) for s in scores[system].keys()]) + '\\\\\hline')
    for system in scores:
        print_en = False
        print(system, end='')
        for system2 in scores:
            p = scores[system][system2]
            if (p >= 0.0 and p < 0.05) or p >= 0.95:
                if print_en:
                    print(' & Y', end='')
                else:
                    print(' & ', end='')
            else:
                if print_en:
                    print(' & N', end='')
                else:
                    print(' & ', end='')

            if system == system2:
                print_en = True
        print('\\\\\hline')

def main():
    ''' main function '''
    # read argument - file with data
    parser = argparse.ArgumentParser(description='Extracts words to a dictionary with their frequencies.')
    parser.add_argument('-r', '--reference', required=True, help='the reference file to read.')
    parser.add_argument('-t', '--translations', required=True, help='the translation files to read.', nargs='+')
    parser.add_argument('-l', '--language', required=False, help='the language.', default='en')
    parser.add_argument('-i', '--iterations', required=False, help='the number of iterations for the bootstrap.', default='1000')
    parser.add_argument('-s', '--sample-size', required=False, help='the sample size (in sentences).', default='100')

    args = parser.parse_args()

    sentences = {}

    length = 0
    # 1. read the reference file
    with codecs.open(args.reference, 'r', 'utf8') as ifh:
        ref = [s.strip() for s in ifh.readlines()]
        length = len(ref)
        
    # 2. read all translation files and get the sentences
    for transfile in args.translations:
        system = os.path.splitext(os.path.basename(transfile))[0]
        print(system)
        sentences[system] = []

        with codecs.open(transfile, 'r', 'utf8') as ifh:
            sentences[system] = [s.strip() for s in ifh.readlines()]

        if length != len(sentences[system]):
            raise SystemExit('Inconsistent number of sentences')

    # Compute overall metrics
    for sys in sentences:
        print(" ".join(["BLEU SACREBLEU", sys, str(get_bleu_sacrebleu(sentences[sys], ref, args.language))]))
        print(" ".join(["TER", sys, str(compute_ter_multeval(sys + "_TER_DIR", sentences[sys], ref, args.language))]))
        print(" ".join(["BLEU MULTEVAL", sys, str(compute_bleu_multeval(sys + "_BLEU_DIR", sentences[sys], ref, args.language))]))

    ter_sentences = {}
    for sys in sentences:
        with codecs.open(os.path.join(sys + "_TER_DIR", 'sys1.opt1'), 'r', 'utf8') as ifh:
            ter_sentences[sys] = [float(s.strip().split('|||')[-2].split('=')[-1]) for s in ifh.readlines()]
            
    bleu_sentences = {}
    for sys in sentences:
        with codecs.open(os.path.join(sys + "_BLEU_DIR", 'sys1.opt1'), 'r', 'utf8') as ifh:
            bleu_sentences[sys] = [float(s.strip().split('|||')[-2].split('=')[-1]) for s in ifh.readlines()]

    # 3. read the other variables.
    iters = int(args.iterations)
    sample_size = int(args.sample_size)
    sample_idxs = np.random.randint(0, length, size=(iters, sample_size))

    # 4. Compute Sample metric
    metrics = {}
    for sys in sentences:
        metrics['bleu_sacrebleu'] = {}
        metrics['ter'] = {}
        metrics['bleu_multeval'] = {}
        
    for sys in sentences:
        metrics['bleu_sacrebleu'][sys] = compute_metric('get_bleu_sacrebleu', sentences[sys], ref, args.language, sample_idxs, iters)
        metrics['ter'][sys] = compute_metric('get_ter', sentences[sys], ter_sentences[sys], args.language, sample_idxs, iters)
        metrics['bleu_multeval'][sys] = compute_metric('get_bleu', sentences[sys], bleu_sentences[sys], args.language, sample_idxs, iters)

    for metric in metrics:
        print("-------------------------------------------------")
        print(metric)
        sign_scores = compute_significance(metrics[metric], iters)
        print_latex_table(sign_scores, metric)
        sign_scores = compute_ttest_scikit(metrics[metric], iters)
        print_latex_table(sign_scores, metric)
        
if __name__ == "__main__":
    main()
