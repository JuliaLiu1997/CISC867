# Run CompareEmbeddings notebook for multiple models and layers
# python run_mturk_correlations.py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats

import corpus
import semantic_embedding


# Parse the corpus
BNC_FILE = "../data/bnc/bnc.pkl"
corpus = corpus.POSCorpus.create_from_pickle(data_file_path=BNC_FILE)

annotation_df = pd.read_csv('../data/annotations/mturk.csv')
relevant_lemmas = annotation_df.lemma.tolist()


# Filter sentences containing lemmas we care about
sentences_with_relevant_lemmas = []
for sentence in corpus.sentences:
  for tok in sentence:
    if tok['lemma'] in relevant_lemmas:
      sentences_with_relevant_lemmas.append(sentence)
      break


def run_model(outfile, model_name, layer):
  print('Running:', model_name, layer)
  embedder = semantic_embedding.SemanticEmbedding(sentences_with_relevant_lemmas)
  embedder.init_bert(model_name=model_name, layer=layer)
  annotation_df[['nv_cosine_similarity', 'n_variation', 'v_variation']] = \
      annotation_df.apply(lambda row: embedder.get_contextual_nv_similarity(row.lemma, method="bert"),
                         axis=1, result_type="expand")

  # Run NV similarity
  corr = scipy.stats.spearmanr(annotation_df.mean_score, annotation_df.nv_cosine_similarity)[0]
  outfile.write('Model %s, Layer %d: %0.9f\n' % (model_name, layer, corr))

  plt.clf()
  plot = sns.boxplot(annotation_df.mean_score, annotation_df.nv_cosine_similarity)
  plot.set_title('%s layer %d, corr = %0.9f' % (model_name, layer, corr))
  plot.get_figure().savefig('figs/%s_%d.png' % (model_name, layer))


with open('results.txt', 'w') as outfile:
  for model in ['bert-base-uncased', 'bert-base-multilingual-cased', 'xlm-roberta-base']:
    for layer in range(13):
      run_model(outfile, model, layer)
