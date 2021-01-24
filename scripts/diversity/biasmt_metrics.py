import itertools
from lexical_diversity import lex_div as ld
from lexicalrichness import LexicalRichness as lr
from scipy.stats import ttest_ind
from joblib import Parallel, delayed
import statistics
import spacy_udpipe
import time
import pickle
import os
from nltk.probability import FreqDist
import logging

def plot_freqdist_freq(fd,
                       max_num=None,
                       cumulative=False,
                       title='Frequency plot',
                       linewidth=2):
    """
    As of NLTK version 3.2.1, FreqDist.plot() plots the counts
    and has no kwarg for normalising to frequency.
    Work this around here.

    INPUT:
        - the FreqDist object
        - max_num: if specified, only plot up to this number of items
          (they are already sorted descending by the FreqDist)
        - cumulative: bool (defaults to False)
        - title: the title to give the plot
        - linewidth: the width of line to use (defaults to 2)
    OUTPUT: plot the freq and return None.
    """

    tmp = fd.copy()
    norm = fd.N()
    for key in tmp.keys():
        tmp[key] = float(fd[key]) / norm

    if max_num:
        tmp.plot(max_num, cumulative=cumulative,
                 title=title, linewidth=linewidth)
    else:
        tmp.plot(cumulative=cumulative,
                 title=title,
                 linewidth=linewidth)

    return

def get_lemmas(sentences, nlpD, system_name, freq_voc = None):
    ''' Computes the lemmas and their frequencies for the given sentences

        :param sentences: a list of sentences
        :param nlpd: the data model for the lematizer
        :param freq_voc: a frequency vocabulary
        :returns: a dictionary of lemmas and frequencies
    '''
    a = time.time()

    lemmas = {}

    if os.path.exists(system_name + ".spacy_udpipe.lemmas"):
        logging.debug("Lemmas dict loading from file")
        with open(system_name + ".spacy_udpipe.lemmas", "rb") as SpUpM:
            lemmas = pickle.load(SpUpM)
        logging.debug("Lemmas dict loaded")
    else:
        logging.debug("Lemmas dict building from scratch")
        nlps = list(nlpD.pipe(sentences, n_process=-1))

        for doc in nlps:
            for token in doc:
                lemma=token.lemma_
                tokenLow=str(token).lower()

                if lemma in lemmas: # existing lemma
                    if tokenLow not in lemmas[lemma]:
                        lemmas[lemma][tokenLow]=1
                    else:
                        lemmas[lemma][tokenLow]+=1
                else:                       # unexisting lemma
                    lemmas[lemma]={}        # if this is the first time we have a lemma then there are no tokens
                    lemmas[lemma][tokenLow]=1

        with open(system_name + ".spacy_udpipe.lemmas", "wb") as PoF:
            pickle.dump(lemmas, PoF)

        logging.debug("Lemmas dict built and saved")

    print("Length of all lemmas: " + str(len(lemmas)))
    singleton_lemmas = [lemma + "\t" + str(len(lemmas[lemma])) for lemma in lemmas if len(lemmas[lemma]) < 2]
    print("Length of singleton lemmas: " + str(len(singleton_lemmas)))
    singleton_matching_lemmas = []

    with open(system_name + ".lemmas", "w") as oF:
        oF.write("\n".join([lemma + ": " + "\t".join(str(f) + "|" + str(g) for (f,g) in zip(lemmas[lemma].keys(), lemmas[lemma].values())) for lemma in lemmas]))

    if freq_voc is not None:
        tmp_lemmas = {}
        for lemma in lemmas:
            if len(lemmas[lemma]) > 1:
                for form in lemmas[lemma]:
                    if form in freq_voc:
                        tmp_lemmas[lemma] = lemmas[lemma]
                        break           # we only need one occurance to match
            else:
                singleton_matching_lemmas.append(lemma)
        lemmas = tmp_lemmas

    print("Length of matched lemmas: " + str(len(lemmas)))
    print("Length of singleton maching lemmas: " + str(len(singleton_matching_lemmas)))

    return lemmas

def simpson_diversity(wordFormDict):
    ''' Computes the Simpson Diversity Index

        :param wordFormDict: a dictionary { 'wordform': count }
        :returns: diversity index (number)
    '''

    def p(n, N):
        ''' Relative abundance
        '''
        if n ==  0:
            return 0
        else:
            return float(n)/N

    N = sum(wordFormDict.values())
    return sum(p(n, N)**2 for n in wordFormDict.values() if n != 0)

def inverse_simpson_diversity(wordFormDict):
    ''' Computes the inverse Simpson Diversity Index
    
        :param wordFormDict: a dictionary { 'wordform': count }
        :returns: diversity index (number) 
    '''
    return float(1)/simpson_diversity(wordFormDict)

"""# Shannon Diversity #
The Shannon-Weiner diversity represent the proportion of species abundance in the population. Its being at maximum when all species occur in similar number of individuals and the lowest when the sample contain one species. From my experience there is no limit to compare the diversity value with as for evenness, which resricted between 0-1. For Example, if the sample contain 4 species each represented by 5o individuals the, diversity H equal 1.3863, and if the sample contain 5 species (one more) and each represented by similar number of individuals (50), the diversity equal 1.6094.
"""

def shannon_diversity(wordFormDict):
    '''
    
        :param wordFormDict: a dictionary { 'species': count }
        :returns: Shannon Diversity Index
    '''
    #>>> sdi({'a': 10, 'b': 20, 'c': 30,})
    #1.0114042647073518
    
    from math import log as ln
    
    def p(n, N):
        """ Relative abundance """
        if n ==  0:
            return 0
        else:
            return (float(n)/N) * ln(float(n)/N)
            
    N = sum(wordFormDict.values())
    
    return -sum(p(n, N) for n in wordFormDict.values() if n != 0)

def compute_simpDiv(nestedDict):
    ''' Computes the simpson diversity for every lemma
        example input : {lemma1:{wordf1: count1, wordf2: count2}, lemma2 {wordform1: count1}}
        output {lemma1: simpDiv1, lemma2:simpDiv2}
        
        :param nestedDict: a nested dictionary
        :returns: a dictionary with the simpson diversity for every lemma 
    '''
    simpsonDict = {}
    for l in nestedDict:
        simpsonDict[l]=simpson_diversity(nestedDict[l])
    return statistics.mean(simpsonDict.values())

def compute_invSimpDiv(nestedDict):
    ''' Computes the simpson diversity for every lemma
        example input : {lemma1:{wordf1: count1, wordf2: count2}, lemma2 {wordform1: count1}}
        output {lemma1: simpDiv1, lemma2:simpDiv2}
    
        :param nestedDict: a dictionary of dictionaries
        :returns: a dictionary with the inversed simpson diversity
    '''
    simpsonDict={}
    for l in nestedDict:
        simpsonDict[l]=inverse_simpson_diversity(nestedDict[l])
    return statistics.mean(simpsonDict.values()) 

def compute_shannonDiv(nestedDict):
    ''' Computes the shannon diversity for every lemma
        example input : {lemma1:{wordf1: count1, wordf2: count2}, lemma2 {wordform1: count1}}
        output {lemma1: simpDiv1, lemma2:simpDiv2}
        
        :param nestedDict: a dictionary of dictionaries
        :returns: a dictionary with the shannon diversity
    '''
    shannonDict={}
    for lem in nestedDict:
        shannonDict[lem]=shannon_diversity(nestedDict[lem])
    return statistics.mean(shannonDict.values())

def compute_yules_i(sentences):
    ''' Computing Yules I measure

        :param sentences: dictionary with all words and their frequencies
        :returns: Yules I (the inverse of yule's K measure) (float) - the higher the better
    '''
    _total, vocabulary = get_vocabulary(sentences)
    M1 = float(len(vocabulary))
    M2 = sum([len(list(g))*(freq**2) for freq,g in itertools.groupby(sorted(vocabulary.values()))])

    try:
        return (M1*M1)/(M2-M1)
    except ZeroDivisionError:
        return 0

def compute_ttr(sentences):
    ''' Computes the type token ratio
    
        :param sentences: the sentences
        :returns: The type token ratio (float)
    '''      

    total, vocabulary = get_vocabulary(sentences)    
    return len(vocabulary)/total
    
def compute_mtld(sentences):
    ''' Computes the MTLD
    
        :param sentences: sentences
    
        :returns: The MTLD (float)
    '''      
    
    def my_mtld(lex, threshold, reverse=False):
        """
        Parameters
        ----------
        threshold: float
            Factor threshold for MTLD. Algorithm skips to a new segment when TTR goes below the
            threshold (default=0.72).
        reverse: bool
            If True, compute mtld for the reversed sequence of text (default=False).
        Returns:
            mtld measure (float)
        """
        if reverse:
            word_iterator = iter(reversed(lex.wordlist))
        else:
            word_iterator = iter(lex.wordlist)

        terms = set()
        word_counter = 0
        factor_count = 0

        for word in word_iterator:
            word_counter += 1
            terms.add(word)
            ttr = len(terms)/word_counter

            if ttr <= threshold:
                word_counter = 0
                terms = set()
                factor_count += 1

        # partial factors for the last segment computed as the ratio of how far away ttr is from
        # unit, to how far away threshold is to unit
        if word_counter > 0:
            factor_count += (1-ttr) / (1 - threshold)

        # ttr never drops below threshold by end of text
        if factor_count == 0:
            ttr = lex.terms / lex.words
            if ttr == 1:
                factor_count += 1
            else:
                factor_count += (1-ttr) / (1 - threshold)

        return len(lex.wordlist) / factor_count

    ll = '\n'.join(sentences)
    lex = lr(ll)
    return lex.mtld()
#    return ld.mtld(ll)
    
def get_vocabulary(sentence_array):
    ''' Compute vocabulary

        :param sentence_array: a list of sentences
        :returns: a list of tokens
    '''
    data_vocabulary = {}
    total = 0
    
    for sentence in sentence_array:
        for token in sentence.strip().split():
            if token not in data_vocabulary:
                data_vocabulary[token] = 1 #/len(line.strip().split())
            else:
                data_vocabulary[token] += 1 #/len(line.strip().split())
            total += 1
            
    return total, data_vocabulary

def compute_ld_metric(metric_func, sentences, sample_idxs, iters):
    ''' Computing metric

        :param metric_func: get_bleu or get_ter_multeval
        :param sys: the sampled sentences from the translation
        :param sample_idxs: indexes for the sample (list)
        :param iters: number of iterations
        :returns: a socre (float)
    '''
    # 5. let's get the measurements for each sample
    scores = Parallel(n_jobs=-1)(delayed(eval(metric_func))([sentences[j] for j in sample_idxs[i]]) for i in range(iters))

    return scores

def compute_gram_diversity(sentences, lang="en", system_name="", freq_voc=None):
    ''' Computing metric

        :param metric_func: get_bleu or get_ter_multeval
        :param sys: the sampled sentences from the translation
        :param sample_idxs: indexes for the sample (list)
        :param iters: number of iterations
        :returns: a socre (float)
    '''
    nlpD = spacy_udpipe.load(lang).tokenizer
    nlpD.max_length = 300000000

    lemmas = get_lemmas(sentences, nlpD, system_name, freq_voc)

    return (compute_simpDiv(lemmas), compute_invSimpDiv(lemmas), compute_shannonDiv(lemmas))

def textToLFP(sentences, step=1000, last=2000):
    '''we are not lowercasing, tokenizing, removing stopwords, numerals etc.
    this is because we are looking into algorithmic bias and as such into the effect of the algorithm
    on the text it is offered. The text is already tokenized. Might add Lowercasing too.'''

    #create Frequency Dictionary
    fdist = FreqDist(" ".join(sentences).split()) # our text is already tokenized. We merge all sentences together
                                                  # and create one huge list of tokens.

    # get size range
    end = last + step
    sizes = list(range(0, end, step))

    #Get words for every frequency band
    freqs = [fdist.most_common(size+step)[size:size+step] for size in sizes[:-1]]
    freqs.append(fdist.most_common()[last:])

    #total tokens
    totalCount=fdist.N()

    #percentage frequency band
    percs = [sum([count for (_word,count) in freq])/totalCount for freq in freqs]

    #plot
    #plot_freqdist_freq(fdist, 20)

    return percs

