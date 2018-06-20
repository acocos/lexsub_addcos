# Lexical Substitution from PPDB

## Overview

This script ranks lexical substitution candidates based on the AddCos metric (Melamud et 
al. [A simple word embedding model for lexical substitution](http://www.aclweb.org/anthology/W15-1501), 2015).
The substitution candidates are taken to be the PPDB paraphrases for the target word. 

This script is specifically for test sentences/instances that have been POS-tagged, with 
each token in the format `word_POS`. Accordingly the word embedding model used for 
lexical substitution should have its vocabulary in the same format, `word_POS`.

Test sentences should be stored in a text file with the following (tab-separated) format, 
where `idx` indicates the  index (starting from 0) of the target word to be substituted 
within in the sentence, and `tgt.P` indicates the target with a simple part of speech 
label, i.e. `bug.N`:

```
<tgt.P>    <sentence_id>    <idx>    <sent_NN toks_NNS ,_, tagged_VBD ...>

```

### Script usage:

```
python lexsub_addcos_ppdb.py <LEXSUBFILE> <PPDBFILE> <EMBEDDINGFILE> <OUTFILE> [<MINSCORE>]
```
where `LEXSUBFILE` is the tab-separated file containing (pos-tagged) test instances in 
the above format; `PPDBFILE` is a PPDB file (i.e. XL or XXL) as available for download
from [the PPDB site](http://paraphrase.org/#/download), `EMBEDDINGFILE` is a file 
containing word embeddings in gensim word2vec format, and `OUTFILE` is the name of the 
destination file to write the results.
 
Optionally specify a `MINSCORE` which is the minimum PPDB score for a paraphrase to be 
considered a potential substitute.

### Note

This code provides the implementation of the AddCos metric as used in the paper:

```
Marianna Apidianaki, Guillaume Wisniewski, Anne Cocos, and Chris Callison-Burch. 2018. Automated Paraphrase Lattice Creation for HyTER Machine Translation Evaluation. In Proceedings of NAACL 2018 (Short Papers). New Orleans, LA.
```

If you are looking for a full implementation of the `hytera` pipeline from that paper (in which this code is just part of the pipeline), you can find it [here](https://github.com/acocos/pp-lexsub-hytera/).