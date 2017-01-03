###########################################
# lexsub_addcos_ppdb.py
###########################################
#
# This script ranks lexical substitution candidates based on the AddCos metric 
# (Melamud et al. A simple word embedding model for lexical substitution, 2015). The 
# substitution candidates are taken to be the PPDB paraphrases for the target word. 
# 
# This script is specifically for test sentences/instances that have been POS-tagged, with 
# each token in the format `word_POS`. Accordingly the word embedding model used for 
# lexical substitution should have its vocabulary in the same format, `word_POS`.
# 
# Test sentences should be stored in a text file with the following (tab-separated) format, 
# where `idx` indicates the  index (starting from 0) of the target word to be substituted 
# within in the sentence, and `tgt.P` indicates the target with a simple part of speech 
# label, i.e. `bug.N`:
# 
# ```
# <tgt.P>    <sentence_id>    <idx>    <sent_NN toks_NNS ,_, tagged_VBD ...>
# 
# ```
# 
# Script usage:
# 
# ```
# python lexsub_addcos_ppdb.py <LEXSUBFILE> <PPDBFILE> <EMBEDDINGFILE> <OUTFILE> [<MINSCORE>]
# ```
# where `LEXSUBFILE` is the tab-separated file containing (pos-tagged) test instances in 
# the above format; `PPDBFILE` is a PPDB file (i.e. XL or XXL) as available for download
# from <URL>, `EMBEDDINGFILE` is a file 
# containing word embeddings in gensim word2vec format, and `OUTFILE` is the name of the 
# destination file to write the results.
#  
# Optionally specify a `MINSCORE` which is the minimum PPDB score for a paraphrase to be 
# considered a potential substitute.

import sys
import gensim
import numpy as np
from collections import namedtuple, Counter

import addcos
import ppdb

def splitpop(string, delimiter):
    """
    Splits a string along a delimiter, and returns the
    string without the last field, and the last field.
    >splitpop('hello.world.test', '.')
    'hello.world', 'test'
    """
    if delimiter not in string:
        string += delimiter
    fields = string.split(delimiter)
    return delimiter.join(fields[:-1]), fields[-1]

def read_semeval_tsv(filename):
    '''
    Yields an iterator over word-contexts in SEMEVAL tsv file, the format used
    in PIC a different word. This code is lifted from that repo
    https://github.com/stephenroller/naacl2016/blob/master/lexsub.py
    :param filename: str
    :return:
    '''
    with open(filename, 'rU') as fin:
        for line in fin:
            line = line.strip()
            if not line: continue
            if '\t' not in line:
                continue
            target, ident, index, sentence = line.split('\t')
            ident = int(ident)
            index = int(index)
            taggedtoks = sentence.split()
            yield (target, ident, index, taggedtoks)
            
word_pos = namedtuple("word_pos", "word, pos")

def read_word_pos(s):
    '''
    Read word_pos from a string formatted as word_POS
    :param s: string
    :return: word_pos
    '''
    word = '_'.join(s.split('_')[:-1])
    pos = s.split('_')[-1].upper()
    return word_pos(word, pos)

def to_str(wp):
    return '_'.join((wp.word, wp.pos))

def get_base_form(word_dot_pos):
    poslookup = {'n': 'NN', 'r': 'RB', 'j': 'JJ', 'a': 'JJ', 'v': 'VB'}
    w, p = splitpop(word_dot_pos, '.')
    return '_'.join((w, poslookup[p]))

def lexsub_score_list(sent_toks, idx, sublist, preppedmodel, cwin=2):
    '''
    Given sentence tokens, the index of the target word, a list of possible substitutions,
    and an embedding model, compute the AddCos lexical substitution score (Melamud et al. 2015)
    for each possible substitution given the target and context
    :param sent_toks: list of str
    :param idx: int
    :param sublist: list of str
    :param preppedmodel: result of addcos.prep_model run on gensim Word2Vec embedding model
    :param cwin: int
    :return: dict {word: score}'
    '''
    # Unpack prepped model
    wordvecs_norm, wordvecs, ctxvecs_norm, ctxvecs, w2i, w2f = preppedmodel
    
    # Get context and target
    context = addcos.get_context_win(sent_toks, idx, cwin)
    target = sent_toks[idx]
    tgt_pos = splitpop(target, '_')[-1]
    
    # Compute addCos rankings
    addcos_ranks = {}
    C = np.array([ctxvecs_norm[w2i.get(c,0)] for c in context])
    t = wordvecs_norm[w2i.get(target, 0)]
    
    for sub in sublist:
        sub_pos = '_'.join((sub, tgt_pos))
        sub_vec = wordvecs_norm[w2i.get(sub_pos, 0)]
        addcos_ranks[sub] = addcos.sim_add(sub_vec, t, C)
        
    return addcos_ranks
    
def main(sentsfile, ppdbfile, modelfile, outfile, MINSCORE):
#     sentsfile = '/scratch-shared/users/acocos/ppdb_wsd_sup/eval/lexsub/semeval_all/sentences.tagged.tsv'
#     ppdbfile = '/scratch-shared/users/acocos/ppdb_wn_map/data/ppdb-2.0-xxl-lexical.gz'
#     modelfile = '/scratch-shared/users/acocos/ppdb_wn_map/data/word-pos.agiga.4b'# '/nlp/users/acocos/paraphrase_wsd/vecs/word-pos.agiga.4b'
#     
    ## load sentences and targets
    orig_tgts = {}
    sents = {}
    for tgt, idnt, idx, toks in read_semeval_tsv(sentsfile):
        orig_tgt = toks[idx]
        orig_tgts[idnt] = orig_tgt
        sents[idnt] = (idx, toks, tgt)
    
    ## load PPDB data
    vocab = set(orig_tgts.values()) | set([get_base_form(s[-1]) for s in sents.values()])
    wplist = [read_word_pos(w) for w in vocab]
    ppdblist = ppdb.fetch_scored_pp_lists_fromfile(wplist, ppdbfile,
                                              singlewordonly=True)
    
    ## For each sentence, generate candidate substitutions from the PPDB paraphrases of the target. 
    ##   First try finding paraphrases for original target word_POS...if there are none, use the base form.     
    ppdb_candidates = {}
    for i, (idx, toks, tgt) in sents.items():
        tgt_wp = read_word_pos(orig_tgts[i])
        cands = ppdblist[tgt_wp]
        if len(cands) == 0:
            tgt_wp = read_word_pos(get_base_form(tgt))
            cands = ppdblist[tgt_wp]
            if len(cands) == 0:
                sys.stderr.write('Could not find PPDB candidates for target %s or its base form %s\n' % (to_str(read_word_pos(orig_tgts[i])), to_str(tgt_wp)))
        good_cands = [w for w,s in cands if s[0] >= MINSCORE]
        if len(good_cands) == 0:
            sys.stderr.write('No PPDB candidates for target %s after filtering for minscore %0.2f\n' % (to_str(tgt_wp), MINSCORE))
        ppdb_candidates[i] = good_cands
    
    ## load model
    embedmodel = gensim.models.Word2Vec.load(modelfile)
    preppedmodel = addcos.prep_model(embedmodel)
    
    fout = open(outfile, 'w')
    
    for idnt, (idx, toks, tgt) in sents.items():
        cands = ppdb_candidates[idnt]
        lexsub_scores = lexsub_score_list(toks, idx, cands, preppedmodel)
        print >> fout, result_str(idnt, tgt, lexsub_scores)
        
    fout.close()

def result_str(idnt, tgt, lexsubdict):
    idnt = str(idnt)
    w, p = splitpop(tgt, '.')
    s = '--'.join((idnt, p, w)) + ' :: '
    for sub, scr in Counter(lexsubdict).most_common():
        s += ' '+' '.join((sub, str(scr), '//'))
    return s
    
if __name__=="__main__":
    SENTENCE_FILE = sys.argv[1]
    PPDB_FILE = sys.argv[2]
    MODEL_FILE = sys.argv[3]
    OUTFILE = sys.argv[4]
    
    if len(sys.argv) > 5:
        MINSCORE = float(sys.argv[5])
    else:
        MINSCORE = 0.0
        
    main(SENTENCE_FILE, PPDB_FILE, MODEL_FILE, OUTFILE, MINSCORE)