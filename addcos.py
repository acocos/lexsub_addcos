import numpy as np
from sklearn.preprocessing import normalize

def get_context_win(toks, ind, N):
    beforewin = list(toks[max(0, ind-N) : ind])
    afterwin = list(toks[ind+1 : min(len(toks), ind+N+1)])
    return beforewin + afterwin

def prep_model(model):
    '''
    Pre-normalize some things to make later calculations faster
    :param model: gensim Word2Vec model
    :return:
    '''
    dim = model.syn0.shape[1]
    wordvecs = model.syn0
    wordvecs = np.vstack([np.zeros(dim), wordvecs])
    wordvecs_norm = normalize(model.syn0, axis=1)
    wordvecs_norm = np.vstack([np.zeros(dim), wordvecs_norm])
    ctxvecs = model.syn1neg
    ctxvecs = np.vstack([np.zeros(dim), ctxvecs])
    ctxvecs_norm = normalize(model.syn1neg, axis=1)
    ctxvecs_norm = np.vstack([np.zeros(dim), ctxvecs_norm])
    w2i = {w: i+1 for i,w in enumerate(model.index2word)}
    w2c = {w: model.vocab[w].count for w in model.vocab}
    model_vocab_sum = sum(w2c.values())
    w2f = {w: float(w2c[w]) / model_vocab_sum for w in model.vocab}
    return (wordvecs_norm, wordvecs, ctxvecs_norm, ctxvecs, w2i, w2f)

def pcos(u,v):
    ''' Assumes vectors already L2 normalized
    :param u:
    :param v:
    :return:
    '''
    return ( np.dot(u,v) + 1 ) / 2.0

def sim_mult(s, t, C):
    '''
    Contextual similarity metric, compares potential substitute vec (s)
    to target vec (t) and Context vecs (C)
    :param s: np.array (1xd)
    :param t: np.array (1xd)
    :param C: np.array (n x d) where n is number of words in context
    :return:
    '''
    n = C.shape[0]
    left = pcos(s, t)
    if n == 0:
        return left
    right = np.prod(pcos(s, C.T))
    return (left * right)**(n+1)


def sim_add(s, t, C):
    '''
    Contextual similarity metric, compares potential substitute vec (s)
    to target vec (t) and Context vecs (C)
    :param s: np.array (1xd)
    :param t: np.array (1xd)
    :param C: np.array (n x d) where n is number of words in context
    :return:
    '''
    n = C.shape[0]
    left = np.dot(s, t)  # assumes already normalized
    if n == 0:
        return left
    right = np.sum(np.dot(s, C.T))
    return (left + right) / (n + 1)

