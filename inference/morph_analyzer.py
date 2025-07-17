from konlpy.tag import Mecab

mecab = Mecab()

def analyze_morphs(text):
    # 어절 단위로 분리
    eojuls = text.strip().split()
    # 각 어절별로 형태소 분석
    morphs_by_eojul = []
    for eojul in eojuls:
        morphs = mecab.pos(eojul)
        morphs_by_eojul.append(morphs)
    return morphs_by_eojul