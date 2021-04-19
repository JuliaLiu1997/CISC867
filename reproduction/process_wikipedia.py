"""
Script to process multilingual Wikipedia files and save data into Python Pickle format.

Usage:
  python reproduction/process_wikipedia.py --wiki_dir=data/wiki/ --ud_dir=data/ud_all/ud-treebanks-v2.5/ --dest_dir=data/wiki/processed_udpipe/ --lang=fr --model=udpipe --tokens 10000000
"""

import collections
import argparse
import pickle
import random
import glob
import tqdm
import os
import spacy_udpipe
import stanza
import corpus
import const

# Setup arguments
parser = argparse.ArgumentParser()
parser.add_argument('--wiki_dir', type=str)
parser.add_argument('--ud_dir', type=str)
parser.add_argument('--dest_dir', type=str)
parser.add_argument('--lang', type=str)
parser.add_argument('--model', choices=['udpipe', 'stanza'])
parser.add_argument('--tokens', type=int)
args = parser.parse_args()
print(args)

if args.model == 'udpipe':
  spacy_udpipe.download(args.lang)
  nlp = spacy_udpipe.load(args.lang)
elif args.model == 'stanza':
  stanza.download(args.lang)
  nlp = stanza.Pipeline(args.lang, use_gpu=True)

# Process UD to get lemma mappings
lang_name_full = const.LANGUAGES[args.lang]
ud_treebanks = corpus.group_treebanks_by_language(args.ud_dir)[lang_name_full]
print('Processing UD:', lang_name_full)
ud_corpus = corpus.POSCorpus.create_from_ud(data_file_list=ud_treebanks)
ud_corpus._setup_lemma_merges()

# Process Wikipedia
ALL_LINES = []
for fname in glob.glob(args.wiki_dir + '/' + args.lang + '/**/*'):
  print(fname)
  with open(fname, encoding='utf-8', mode = 'r') as f:
    contents = f.read()
  for line in contents.split('\n'):
    if line.strip() == '':
      continue
    if line.startswith('<doc') or line.startswith('</doc'):
      continue
    ALL_LINES.append(line)

print('Lines:', len(ALL_LINES))
random.seed(12345)
random.shuffle(ALL_LINES)

def process_line(line):
  return nlp(line)

token_count = 0
pos_counts = collections.defaultdict(int)

sentences = []
pbar = tqdm.tqdm(total=args.tokens)

if args.model == 'udpipe':
  for doc in map(process_line, ALL_LINES):
    if token_count > args.tokens:
      break
    for sent in doc.sents:
      if len(sent) < 5:
        continue

      sentence = []
      for token in sent:
        token_count += 1
        token_lemma = None

        # Only assign lemma to nouns and verbs.
        # Try using UD lemma merger, otherwise fallback to udpipe lemma output.
        if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
          token_lemma = ud_corpus.merged_lemma_table.get(token.text.lower())
          if token_lemma is None:
            token_lemma = token.lemma_.lower()

        sentence.append({'word': token.text, 'lemma': token_lemma, 'pos': token.pos_})
        pos_counts[token.pos_] += 1

      pbar.update(len(sentence))
      sentences.append(sentence)

elif args.model == 'stanza':
  for doc in map(process_line, ALL_LINES):
    if token_count > args.tokens:
      break
    for sent in doc.sentences:
      if len(sent.words) < 5:
        continue

      sentence = []
      for token in sent.words:
        token_count += 1
        token_lemma = None

        # Only assign lemma to nouns and verbs.
        # Try using UD lemma merger, otherwise fallback to udpipe lemma output.
        if token.upos == 'NOUN' or token.upos == 'VERB':
          token_lemma = ud_corpus.merged_lemma_table.get(token.text.lower())
          if token_lemma is None:
            token_lemma = token.lemma.lower()

        sentence.append({'word': token.text, 'lemma': token_lemma, 'pos': token.upos})
        pos_counts[token.upos] += 1

      pbar.update(len(sentence))
      sentences.append(sentence)

pbar.close()

print('Tokens:', token_count)
for tok, tokcount in pos_counts.items():
  print('%s: %d' % (tok, tokcount))

save_file = args.dest_dir + '/' + args.lang + '.pkl'
print('Saving:', save_file)
with open(save_file, 'wb') as f:
  pickle.dump(sentences, f)
print('Done.')
