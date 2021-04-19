from collections import defaultdict, Counter
from disjoint_set import DisjointSet
import numpy as np
import pandas as pd
import conllu
import pickle
import os
import glob


# Crawl the UD direrctory and get a list of all conllu files, grouped by language
def group_treebanks_by_language(ud_path):
  ud_files = defaultdict(list)
  for ud_corpus_name in os.listdir(ud_path):
    language_name = ud_corpus_name[3:].split('-')[0].replace('_', ' ')
    for conllu_file in glob.glob(ud_path + ud_corpus_name + '/*.conllu'):
      ud_files[language_name].append(conllu_file)
  return ud_files



class POSCorpus(object):
  """Corpus for analyzing POS flexibility. After creation, corpus.sentences should consist of
  a list of sentences, each with a list of words. Example structure:
  [
    [
      {'word': "I", 'lemma': "i", 'pos': "PRON"},
      {'word': "love", 'lemma': "love", 'pos': "VERB"},
      {'word': "cats", 'lemma': "cat", 'pos': "NOUN"},
    ]
  ]
  """
  def __init__(self):
    self.lemma_merge_ds = None

  @classmethod
  def create_from_ud(cls, data_file_list):
    """Initialize corpus from a path to a file in conllu format"""
    corpus = POSCorpus()
    corpus.sentences = []

    for data_file_path in data_file_list:
      with open(data_file_path, "r", encoding="utf-8") as data_file:
        data = data_file.read()
        data = conllu.parse(data)

      for token_list in data:
        sentence = []
        for token in token_list:
          pos = token['upostag']
          lemma = token['lemma']
          word = token['form']
          # Sometimes the corpus doesn't have words, only underscores
          if word == '_' or lemma == '_':
            continue
          sentence.append({'word': word, 'lemma': lemma, 'pos': pos})
        if len(sentence) > 0:
          corpus.sentences.append(sentence)

    return corpus
  

  @classmethod
  def create_from_pickle(cls, data_file_path):
    """Initialize corpus from pkl file, containing lists of sentences"""
    corpus = POSCorpus()
    with open(data_file_path, 'rb') as f:
      corpus.sentences = pickle.load(f)
    return corpus


  def get_per_lemma_stats(self, flexibility_threshold=0.05):
    # Gather usages of each lemma
    # {lemma -> (POS, word)}
    lemma_forms = defaultdict(list)
    for sentence in self.sentences:
      for token in sentence:
        lemma = token['lemma']
        word = token['word']
        pos = token['pos']
        lemma_forms[lemma].append((pos, word))

    # Noun/Verb statistics for each lemma
    lemma_count_df = []
    for lemma, lemma_occurrences in lemma_forms.items():
      noun_count = len([word for (pos, word) in lemma_occurrences if pos == 'NOUN'])
      verb_count = len([word for (pos, word) in lemma_occurrences if pos == 'VERB'])
      lemma_count_df.append({'lemma': lemma, 'noun_count': noun_count, 'verb_count': verb_count})
    lemma_count_df = pd.DataFrame(lemma_count_df)

    lemma_count_df = lemma_count_df[lemma_count_df['noun_count'] + lemma_count_df['verb_count'] > 0]
    lemma_count_df['majority_tag'] = np.where(lemma_count_df['noun_count'] >= lemma_count_df['verb_count'], 'NOUN', 'VERB')
    lemma_count_df['total_count'] = lemma_count_df[['noun_count', 'verb_count']].sum(axis=1)
    lemma_count_df['minority_count'] = lemma_count_df[['noun_count', 'verb_count']].min(axis=1)
    lemma_count_df['minority_ratio'] = lemma_count_df['minority_count'] / lemma_count_df['total_count']
    lemma_count_df['is_flexible'] = lemma_count_df['minority_ratio'] > flexibility_threshold

    return lemma_count_df


  # Helper iterator to return N/V words and lemmas in corpus, lowercased
  def _iterate_words(self):
    for sentence in self.sentences:
      for token in sentence:
        if token['pos'] in ['NOUN', 'VERB']:
          yield token['word'].lower(), token['lemma'].lower(), token['pos']


  def _setup_lemma_merges(self):
    self.lemma_merge_ds = DisjointSet()
    for word, lemma, _ in self._iterate_words():
      self.lemma_merge_ds.union(word, lemma)

    # Group words that share the same lemma
    lemma_counter = Counter()
    for _, lemma, _ in self._iterate_words():
      lemma_counter[lemma] += 1

    lemma_groups = defaultdict(set)
    for word, lemma, _ in self._iterate_words():
      lemma_groups[self.lemma_merge_ds.find(word)].add(word)

    # Name of the group is the most frequent lemma in the group
    # Eg: [voyage, voyages, voyagerai, ...] should map to the same lemma
    self.merged_lemma_table = {}
    for word, lemma, _ in self._iterate_words():
      if word in self.merged_lemma_table:
        continue
      maxn, maxw = 0, None
      for w in lemma_groups[self.lemma_merge_ds.find(word)]:
        if lemma_counter[w] > maxn:
          maxn = lemma_counter[w]
          maxw = w
      self.merged_lemma_table[word] = maxw


  def get_lemma_stats_merge_method(self, flexibility_threshold=0.05):
    if self.lemma_merge_ds is None:
      self._setup_lemma_merges()

    # Gather usages of each lemma
    # {lemma -> (POS, word)}
    lemma_forms = defaultdict(list)
    for word, lemma, pos in self._iterate_words():
      lemma_forms[self.lemma_merge_ds.find(lemma)].append((pos, word))

    # Noun/Verb statistics for each lemma
    lemma_count_df = []
    for lemma, lemma_occurrences in lemma_forms.items():
      noun_count = len([word for (pos, word) in lemma_occurrences if pos == 'NOUN'])
      verb_count = len([word for (pos, word) in lemma_occurrences if pos == 'VERB'])
      lemma_count_df.append({'lemma': self.merged_lemma_table[lemma], 'noun_count': noun_count, 'verb_count': verb_count})
    lemma_count_df = pd.DataFrame(lemma_count_df)

    lemma_count_df = lemma_count_df[lemma_count_df['noun_count'] + lemma_count_df['verb_count'] > 0]
    lemma_count_df['majority_tag'] = np.where(lemma_count_df['noun_count'] >= lemma_count_df['verb_count'], 'NOUN', 'VERB')
    lemma_count_df['total_count'] = lemma_count_df[['noun_count', 'verb_count']].sum(axis=1)
    lemma_count_df['minority_count'] = lemma_count_df[['noun_count', 'verb_count']].min(axis=1)
    lemma_count_df['minority_ratio'] = lemma_count_df['minority_count'] / lemma_count_df['total_count']
    lemma_count_df['is_flexible'] = lemma_count_df['minority_ratio'] > flexibility_threshold

    return lemma_count_df
