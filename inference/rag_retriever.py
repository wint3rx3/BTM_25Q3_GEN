from embedding_util import encode_texts, load_embeddings
import numpy as np

def retrieve_rules(input_text, rules, top_k=5, embed_path='./inference/rule_embeds.npy'):
    rule_embeds = load_embeddings(embed_path)
    input_embed = encode_texts([input_text])[0]
    sims = np.dot(rule_embeds, input_embed) / (np.linalg.norm(rule_embeds, axis=1) * np.linalg.norm(input_embed))
    top_idx = sims.argsort()[-top_k:][::-1]
    return [rules[i] for i in top_idx]