import re
import sys
import gzip
from nltk.stem import WordNetLemmatizer

from word_pos import word_pos, read_word_pos

def flatten(l):
    return [item for sublist in l for item in sublist]

class PPDB:
    def __init__(self, ppdbfile, score='PPDB2.0Score',
                 singleword=False, lemmatize=False):
        self.ppdbfile = ppdbfile
        self.ppsets = {}
        self.vocab = set([])
        self.scoretype = score
        self.singleword = singleword
        self.lemmatize = lemmatize

    def read_vocabfile(self, vocabfile):
        self.vocab = set([read_word_pos(w.strip())
                          for w in open(vocabfile,'rU').readlines()])

    def set_vocab(self, vocablist):
        '''
        Load vocab from a set of word_pos namedtuples
        :param vocablist:
        :return:
        '''
        self.vocab = set(vocablist)

    def load_paraphrases(self, secondorder=True):
        '''
        Loads sets of PPDB paraphrases for words contained in vocabfile.
        The vocabfile should contain one word per line, formatted as
        word_POS or multi_word_POS, e.g.

        horse_NN
        horses_NNS
        marine_life_NP

        '''
        if len(self.vocab) == 0:
            sys.stderr.write('Vocab is zero-length. Load vocab before calling'
                             'load_paraphrases on this object.')
            return
        sys.stderr.write('Loading first- and second-order paraphrases '
                         'for %d vocabulary terms...' % len(self.vocab))
        firstorderpplists = fetch_scored_pp_lists_fromfile(list(self.vocab), self.ppdbfile,
                                                           scores=[self.scoretype],
                                                           singlewordonly=self.singleword,
                                                           lemmatize=self.lemmatize)
        self.ppsets = {pp: {ppp: scr[0] for ppp, scr in pplist}
                       for pp, pplist in firstorderpplists.items()}
        if secondorder:
            secondorderpps = set(flatten([[word_pos(pp, wt.pos) for pp in pdict]
                                          for wt,pdict in self.ppsets.items() if len(pdict)>0]))
            secondorderpplists = fetch_scored_pp_lists_fromfile(list(secondorderpps), self.ppdbfile,
                                                                scores=[self.scoretype],
                                                                singlewordonly=self.singleword,
                                                                lemmatize=self.lemmatize)
            secondorderppsets = {pp: {ppp: scr[0] for ppp, scr in pplist}
                                 for pp, pplist in secondorderpplists.items()}
            self.ppsets.update(secondorderppsets)
        sys.stderr.write('done\n')


def fetch_scored_pp_lists_fromfile(wposlist, masterfile, scores=['PPDB2.0Score'],
                                   singlewordonly=False, lemmatize=False):
    """
    Pull scored pp lists for all words/pos in wtypelist from PPDB text file masterfile
    :param wposlist: list of word_pos namedtuples
    :param masterfile: PPDB file, downloaded from paraphrase.org
    :param scores: list of score types as they appear in the PPDB master list
    :return: dict of wtype -> list of tuple(word, [scores<float>])
    """

    words = set([wp.word for wp in wposlist])
    lem = WordNetLemmatizer()
    if lemmatize:
        lem_words = set([lem.lemmatize(w.decode('utf8')) for w in words])
    else:
        lem_words = words

    pp_lists = {wp: [] for wp in wposlist}

    score_search = {  # Add more as needed
        'PPDB2.0Score': re.compile('PPDB2.0Score=\d+[.\d]*'),
        'PPDB1.0Score': re.compile('PPDB1.0Score=\d+[.\d]*'),
        'RarityPenalty': re.compile('RarityPenalty=\d+[.\d]*'),
        'AGigaSim': re.compile('AGigaSim=\d+[.\d]*'),
        'GoogleNgramSim': re.compile('GoogleNgramSim=\d+[.\d]*'),
        'Independent': re.compile('Independent=\d+[.\d]*')
    }
    sys.stderr.write("Fetching PPDB paraphrases from file...\n")

    wposset = set(wposlist)
    if masterfile[-2:] == 'gz':
        fin = gzip.open(masterfile, 'r')
    else:
        fin = open(masterfile, 'rU')
    for line in fin:
        entry = line.split('|||')
        try:
            if singlewordonly and len(entry[2].strip().split()) > 1:
                continue
            else:
                if not lemmatize:
                    w1 = entry[1].strip()
                    if w1 in words:
                        pos = entry[0].strip().replace('[','').replace(']','')
                        if word_pos(w1, pos) in wposset:  # we have a match
                            pterm = '_'.join(entry[2].strip().split())  # underscore spaces
                            try:
                                scorevec = [float(re.search(score_search[s], entry[3]).group(0).split('=')[1]) for s in scores]
                                pp_lists[word_pos(w1, pos)].append(tuple([pterm, tuple(scorevec)]))
                            except AttributeError:
                                continue
                else:
                    w1 = entry[1].strip()
                    if lem.lemmatize(w1.decode('utf8')) in lem_words:
                        pos = entry[0].strip().replace('[','').replace(']','')
                        if word_pos(w1, pos) in wposset:
                            pterm = '_'.join(entry[2].strip().split())
                            try:
                                scorevec = [float(re.search(score_search[s], entry[3]).group(0).split('=')[1]) for s in scores]
                                pp_lists[word_pos(w1, pos)].append(tuple([pterm, tuple(scorevec)]))
                            except AttributeError:
                                continue
        except KeyError:
            continue
        except IndexError:
            continue
    fin.close()
    return pp_lists