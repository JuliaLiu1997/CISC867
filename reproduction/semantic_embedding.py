import numpy as np
import pandas as pd
import allennlp.commands.elmo
import sklearn.metrics
import torch
import transformers
import tqdm


# Throw when we encounter an issue processing a lemma
class InvalidLemmaException(Exception):
  pass


class SemanticEmbedding:
  def __init__(self, sentences):
    self.sentences = sentences

  def init_elmo(self, layer=2):
    """Init here because batching required for efficiency
    @param layer = one of [0, 1, 2]
    """
    self.elmo = allennlp.commands.elmo.ElmoEmbedder(cuda_device=0)
    self.elmo_layer = layer


  def init_bert(self, model_name='bert-base-uncased', layer=12):
    """Initialize BERT model (but don't compute anything)
    @param model_name = either bert-base-uncased or bert-base-multilingual-cased
    @param layer = integer between 0 and 12
    """
    self.model_name = model_name
    self.bert_layer = layer
    bert_config = transformers.AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    self.bert_model = transformers.AutoModel.from_pretrained(model_name, config=bert_config).cuda()
    self.bert_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

  
  # Helper function for padding input for BERT so that we can batch it
  # Truncate to 100 tokens at most to avoid memory problems
  def _convert_to_bert_input(self, batch_tokens):
    def pad_to_length(tokens, desired_len):
      return tokens + (['[PAD]'] * (desired_len - len(tokens)))
    sentences = [' '.join(t['word'] for t in sentence_tokens) for sentence_tokens in batch_tokens]
    bert_tokens = [self.bert_tokenizer.tokenize(sentence)[:100] for sentence in sentences]
    max_len = max([len(tokens) for tokens in bert_tokens])
    padded_tokens = [pad_to_length(tokens, max_len) for tokens in bert_tokens]
    padded_ids = [self.bert_tokenizer.encode(tokens, add_special_tokens=False) for tokens in padded_tokens]
    attn_mask = [[1 if token != '[PAD]' else 0 for token in tokens] for tokens in padded_tokens]
    return padded_tokens, padded_ids, attn_mask


  # Check if wordpiece matches with a word at position i
  # Eg: ['my', 'cat', 'is', 'named', 'xiao', '##nu', '##an', '##hu', '##o']
  # Word:                            'xiaonuanhuo'
  def _match_wordpiece(self, wordpiece_tokens, word, i):
    if word == None:
      return None

    word = word.lower()
    wordpiece_tokens = [wp.lower() for wp in wordpiece_tokens]

    if not word.startswith(wordpiece_tokens[i]):
      return None
    if word == wordpiece_tokens[i]:
      return [i]

    # Get the whole word, then compare
    whole_word = wordpiece_tokens[i]
    match_ixs = [i]
    for j in range(i+1, len(wordpiece_tokens)):
      if wordpiece_tokens[j].startswith('##'):
        whole_word += wordpiece_tokens[j][2:]
        match_ixs.append(j)
      else:
        whole_word += wordpiece_tokens[j]
        match_ixs.append(j)

      if len(whole_word) >= len(word):
        break

    if whole_word == word:
      return match_ixs
    else:
      return None


  # Convert XLM tokens to BERT-like format. Example:
  # BERT: ['util', '##iza', '##tion']
  # XLM:  ['_util', 'iza', 'tion']
  def _convert_xlm_token_to_bert(self, tok):
    if tok[0] == '[':
      return tok
    if tok[0] == '▁':
      return tok[1:]
    else:
      return '##' + tok


  def get_bert_embeddings_for_lemma(self, lemma):
    # Gather sentences that are relevant
    relevant_sentences = []
    sentence_indices = []
    for sentence_ix, sentence in enumerate(self.sentences):
      if any([t['lemma'] == lemma for t in sentence]):
        sentence_indices.append(sentence_ix)
        relevant_sentences.append(sentence)

    print('Processing lemma: %s (%d instances)' % (lemma, len(relevant_sentences)))

    noun_embeddings = []
    verb_embeddings = []
    noun_indices = []
    verb_indices = []

    # Compute BERT embeddings in batches
    BATCH_SIZE = 32
    for batch_ix in range(0, len(relevant_sentences), BATCH_SIZE):
      batch_sentences = relevant_sentences[batch_ix : batch_ix+BATCH_SIZE]
      batch_tokens, padded_ids, attn_mask = self._convert_to_bert_input(batch_sentences)
      batch_embeddings = self.bert_model(
        torch.tensor(padded_ids).cuda(),
        attention_mask=torch.tensor(attn_mask).cuda()
      )[2][self.bert_layer]
      batch_embeddings = batch_embeddings.cpu().detach().numpy()

      # Process one sentence at a time.
      # Need to do a two-step matching process because the BERT embeddings correspond to
      # WordPiece tokens, which don't always match up with our tokens.
      for ix in range(len(batch_tokens)):
        token_list = batch_sentences[ix]
        wordpiece_tokens = batch_tokens[ix]
        embeddings = batch_embeddings[ix]
        original_sentence_ix = sentence_indices[batch_ix + ix]

        if self.model_name.startswith('xlm'):
          wordpiece_tokens = [self._convert_xlm_token_to_bert(t) for t in wordpiece_tokens]

        assert len(wordpiece_tokens) == len(embeddings)

        # Find word in the sentence that has the lemma
        pos = None
        lemma_form = None
        for i in range(len(token_list)):
          if token_list[i]['lemma'] == lemma:
            if token_list[i]['pos'] in ['NOUN', 'VERB']:
              pos = token_list[i]['pos']
              lemma_form = token_list[i]['word']
              break

        # Get the embedding that matches token
        for i in range(len(wordpiece_tokens)):
          match_ixs = self._match_wordpiece(wordpiece_tokens, lemma_form, i)
          if match_ixs:
            token_embedding = embeddings[match_ixs[0]]
            #token_embedding = np.mean([embeddings[i] for i in match_ixs], axis=0)
            if pos == 'NOUN':
              noun_embeddings.append(token_embedding)
              noun_indices.append(original_sentence_ix)
              break
            elif pos == 'VERB':
              verb_embeddings.append(token_embedding)
              verb_indices.append(original_sentence_ix)
              break

    if noun_embeddings == [] or verb_embeddings == []:
      print('Error with lemma:', lemma)
      raise InvalidLemmaException()

    noun_embeddings = np.vstack(noun_embeddings)
    verb_embeddings = np.vstack(verb_embeddings)
    return noun_embeddings, verb_embeddings, noun_indices, verb_indices


  def get_elmo_embeddings_for_lemma(self, lemma):
    # Gather sentences that are relevant
    relevant_sentences = []
    sentence_indices = []
    for sentence_ix, sentence in enumerate(self.sentences):
      if any([t['lemma'] == lemma for t in sentence]):
        sentence_indices.append(sentence_ix)
        relevant_sentences.append(sentence[:100])

    print('Processing lemma: %s (%d instances)' % (lemma, len(relevant_sentences)))

    noun_embeddings = []
    verb_embeddings = []
    BATCH_SIZE = 32
    for batch_ix in range(0, len(relevant_sentences), BATCH_SIZE):
      batch_sentences = relevant_sentences[batch_ix : batch_ix+BATCH_SIZE]
      batch_tokens = [[t['word'] for t in sentence] for sentence in batch_sentences]
      batch_embeddings = self.elmo.embed_batch(batch_tokens)
      # Only take embeddings from specified ELMo layer
      batch_embeddings = [x[self.elmo_layer] for x in batch_embeddings]

      for sentence_ix in range(len(batch_sentences)):
        token_list = batch_sentences[sentence_ix]
        for token_ix in range(len(token_list)):
          if token_list[token_ix]['lemma'] == lemma:
            if token_list[token_ix]['pos'] == 'NOUN':
              noun_embeddings.append(batch_embeddings[sentence_ix][token_ix])
              break
            elif token_list[token_ix]['pos'] == 'VERB':
              verb_embeddings.append(batch_embeddings[sentence_ix][token_ix])
              break

    noun_embeddings = np.vstack(noun_embeddings)
    verb_embeddings = np.vstack(verb_embeddings)
    return noun_embeddings, verb_embeddings


  def get_contextual_nv_similarity(self, lemma, method):
    """Compute cosine similarity between noun and verb embeddings, for a given lemma.
    Method can be 'elmo' or 'bert'.

    Returns: (n/v similarity, n-variation, v-variation)
    """
    if method == 'elmo':
      noun_embeddings, verb_embeddings = self.get_elmo_embeddings_for_lemma(lemma)
    elif method == 'bert':
      try:
        noun_embeddings, verb_embeddings, _, _ = self.get_bert_embeddings_for_lemma(lemma)
      except InvalidLemmaException:
        return None, None, None
    else:
      assert(False)

    # Sample majority class to be the same size
    minority_size = min(noun_embeddings.shape[0], verb_embeddings.shape[0])
    noun_embeddings = noun_embeddings[np.random.choice(noun_embeddings.shape[0], minority_size, replace=False), :]
    verb_embeddings = verb_embeddings[np.random.choice(verb_embeddings.shape[0], minority_size, replace=False), :]
    
    avg_noun_embedding = np.mean(noun_embeddings, axis=0)
    avg_verb_embedding = np.mean(verb_embeddings, axis=0)
    nv_similarity = float(
      sklearn.metrics.pairwise.cosine_similarity(
        avg_noun_embedding[np.newaxis,:],
        avg_verb_embedding[np.newaxis,:]
      )
    )

    n_variation = np.mean(np.sum((noun_embeddings - avg_noun_embedding)**2, axis=1)**0.5)
    v_variation = np.mean(np.sum((verb_embeddings - avg_verb_embedding)**2, axis=1)**0.5)

    return nv_similarity, n_variation, v_variation
