from collections import namedtuple

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
    return '_'.join([wp.word, wp.pos])