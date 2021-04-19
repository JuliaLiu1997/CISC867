"""
Script to process XML files from BNC corpus and save data into Python Pickle format
that's faster to read.

Usage:
  python reproduction/process_bnc.py --bnc_dir=data/bnc/download/Texts --to=data/bnc/bnc.pkl
"""

import argparse
import nltk.corpus.reader.bnc
import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--bnc_dir', type=str)
parser.add_argument('--to_file', type=str)
args = parser.parse_args()
print(args)

bnc_reader = nltk.corpus.reader.bnc.BNCCorpusReader(root=args.bnc_dir, fileids=r'[a-z]{3}/\w*\.xml')

# bnc_reader.sents(): list of [word1, word2, ...]
# bnc_reader.tagged_sents(): list of [(lemma1, pos1), (lemma2, pos2), ...]
# There is no way to get all three at the same time, so we must zip and traverse both lists simultaneously.
sents = bnc_reader.sents()
tagged_sents = bnc_reader.tagged_sents(stem=True)

sentences = []
for sent, tagged_sent in tqdm.tqdm(zip(sents, tagged_sents)):
  sentence = []
  for word, (lemma, pos) in zip(sent, tagged_sent):
    # Replace SUBST -> NOUN to be consistent with UD tags
    if pos == 'SUBST':
      pos = 'NOUN'
    sentence.append({'word': word, 'lemma': lemma, 'pos': pos})
  sentences.append(sentence)

with open(args.to_file, 'wb') as f:
  pickle.dump(sentences, f)

print('Done')
