from inference.retrieval.rule_loader import load_rules
import numpy as np
from inference..embedding.embedding_util import encode_texts, save_embeddings

# 규정 파일 경로 리스트
rule_filepaths = [
    './docs/띄어쓰기.json',
    './docs/한글_맞춤법_규정.json',
    './docs/표준어__규정.json',
    './docs/외래어__표기법.json',
    './docs/문장_부호.json'
]

# 규정 전체 로드
rules = load_rules(rule_filepaths)

# 10개만 임베딩
rules = rules[:10]

# 임베딩 생성
texts = [rule.get("content", "") for rule in rules]
embeds = encode_texts(texts)

# npy로 저장
np.save("rule_embeds.npy", embeds)