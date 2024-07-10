from sklearn.metrics.pairwise import cosine_similarity

def inspect_direction_representing_words(word_vectors, all_word_embeddings, num_tops, tokenizer):
  sims = cosine_similarity(all_word_embeddings, word_vectors)
  top_ids = sims.argsort(axis=0)[:-num_tops-1:-1].T
  return [tokenizer.convert_ids_to_tokens(ids) for ids in top_ids]  