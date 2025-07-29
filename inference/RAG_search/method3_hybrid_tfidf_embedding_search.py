"""
Method 3: 하이브리드 TF-IDF + 임베딩 검색
- TF-IDF + Qwen3 임베딩 + Jaccard 유사도 결합
- 가중치 기반 점수 블렌딩
- 다단계 후보 선별 과정
- TF-IDF로 후보 선별 → 임베딩 유사도 → Jaccard 유사도 → 가중치 결합
"""

import ast
import json
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class HybridTFIDFEmbeddingSearch:
    def __init__(self, embedding_model_name='Qwen/Qwen3-Embedding-4B', device=None, score_threshold=0.2):
        """
        하이브리드 TF-IDF + 임베딩 검색 초기화
        
        Args:
            embedding_model_name (str): 사용할 임베딩 모델명
            device (str): 사용할 디바이스
            score_threshold (float): 최종 점수 임계값 (기본값: 0.2)
        """
        self.embedding_model_name = embedding_model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        
        # 임베딩 모델 로드
        self._load_embedding_model()
        
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name, 
            use_fast=True,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.embedding_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        
        print(f"임베딩 모델 로드 완료: {self.embedding_model_name}")
    
    def load_data(self, data_paths):
        """
        규정 데이터 로드 및 전처리
        
        Args:
            data_paths (dict): 데이터 파일 경로 딕셔너리
        """
        all_data = []
        
        for key, path in data_paths.items():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        
        self.df = pd.DataFrame(all_data)
        self.df['keywords'] = self.df['keywords'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
        self.df['conditions'] = self.df['conditions'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
        self.df['index'] = self.df.index
        
        # TF-IDF 벡터화
        self._build_tfidf_matrix()
        
        print(f"데이터 로드 완료: {len(self.df)}개 규정")
    
    def _build_tfidf_matrix(self):
        """TF-IDF 매트릭스 구축"""
        corpus = self.df['keywords'].astype(str).tolist()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), 
            min_df=1, 
            max_df=0.9
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print("TF-IDF 매트릭스 구축 완료")
    
    @torch.no_grad()
    def qwen3_embed(self, texts, batch_size=32, max_length=2048, l2_normalize=True):
        """
        텍스트 리스트를 Qwen3 임베딩 벡터로 변환
        
        Args:
            texts (list): 텍스트 리스트
            batch_size (int): 배치 크기
            max_length (int): 최대 토큰 길이
            l2_normalize (bool): L2 정규화 여부
            
        Returns:
            np.ndarray: 임베딩 벡터 배열
        """
        vecs = []
        device = next(self.model.parameters()).device
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True,
                max_length=max_length, 
                return_tensors="pt"
            ).to(device)
            
            out = self.model(**enc)
            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            
            # Mean pooling
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            v = pooled.detach().float().cpu().numpy()
            
            if l2_normalize:
                n = np.linalg.norm(v, axis=1, keepdims=True)
                n[n == 0] = 1.0
                v = v / n
            
            vecs.append(v)
        
        return np.vstack(vecs)
    
    def compute_jaccard(self, qset, dset):
        """
        두 세트 간의 Jaccard 유사도 계산
        
        Args:
            qset (set): 쿼리 키워드 세트
            dset (set): 문서 키워드 세트
            
        Returns:
            float: Jaccard 유사도
        """
        inter = qset & dset
        union = qset | dset
        return len(inter) / len(union) if union else 0.0
    
    def hybrid_search(self, keywords_list, tfidf_k=50, top_k=10,
                     weight_tfidf=0.5, weight_emb=0.3, weight_jac=0.2):
        """
        하이브리드 검색 수행
        
        Args:
            keywords_list (list): 검색 키워드 리스트
            tfidf_k (int): TF-IDF 후보 개수
            top_k (int): 최종 반환 결과 개수
            weight_tfidf (float): TF-IDF 가중치
            weight_emb (float): 임베딩 가중치
            weight_jac (float): Jaccard 가중치
            
        Returns:
            tuple: (검색된 DataFrame, 점수 리스트)
        """
        if self.df is None or self.tfidf_matrix is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요.")
        
        # 1. TF-IDF 검색
        query = " ".join(keywords_list)
        query_vec = self.vectorizer.transform([query])
        tfidf_scores_full = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # 후보 선별
        num_candidates = min(tfidf_k, len(tfidf_scores_full))
        cand_idx = np.argsort(tfidf_scores_full)[::-1][:num_candidates]
        
        if len(cand_idx) == 0:
            return pd.DataFrame(), []
        
        # TF-IDF 점수 정규화
        tfidf_scores = tfidf_scores_full[cand_idx]
        tfidf_scores = MinMaxScaler().fit_transform(tfidf_scores.reshape(-1, 1)).flatten()
        
        # 2. 임베딩 유사도 계산
        texts = [self.df.iloc[i]['keywords'] for i in cand_idx]
        db_emb = self.qwen3_embed(texts, batch_size=32)
        query_emb = self.qwen3_embed([query], batch_size=1)[0]
        
        # 코사인 유사도 (L2 정규화했으므로 내적 = 코사인)
        emb_scores = db_emb @ query_emb
        
        # 3. Jaccard 유사도 계산
        qset = set(keywords_list)
        jac_scores = np.array([
            self.compute_jaccard(
                qset, 
                set(map(str.strip, self.df.iloc[i]['keywords'].split(',')))
            )
            for i in cand_idx
        ])
        
        # 4. 가중치 결합
        final_scores = (
            weight_tfidf * tfidf_scores +
            weight_emb * emb_scores +
            weight_jac * jac_scores
        )
        
        # 5. Top-k 정렬 및 임계값 필터링
        # 임계값 이상만 필터링
        valid_indices = final_scores >= self.score_threshold
        if not np.any(valid_indices):
            return pd.DataFrame(), []
        
        filtered_scores = final_scores[valid_indices]
        filtered_indices = cand_idx[valid_indices]
        
        # Top-k 선택
        num_top_results = min(top_k, len(filtered_scores))
        top_positions = np.argsort(filtered_scores)[::-1][:num_top_results]
        
        top_indices = filtered_indices[top_positions]
        top_scores = filtered_scores[top_positions]
        
        return self.df.iloc[top_indices], top_scores.tolist()
    
    def string_list_to_actual_list(self, string_list):
        """
        문자열 형태의 리스트를 실제 파이썬 리스트로 변환
        
        Args:
            string_list (str or list): 문자열 형태의 리스트 또는 이미 리스트인 경우
            
        Returns:
            list: 변환된 리스트
        """
        # 이미 리스트인 경우 그대로 반환
        if isinstance(string_list, list):
            return string_list
        
        # 문자열인 경우 파싱 시도
        try:
            return ast.literal_eval(string_list)
        except (ValueError, SyntaxError):
            cleaned = string_list.strip('[]').strip()
            return [item.strip().strip("'\"") for item in cleaned.split(',')] if cleaned else []
    
    def batch_search(self, keywords_data, tfidf_k=50, top_k=10,
                    weight_tfidf=0.5, weight_emb=0.3, weight_jac=0.2):
        """
        키워드 데이터에 대한 배치 검색
        
        Args:
            keywords_data (pd.DataFrame or list): 키워드 데이터
            tfidf_k (int): TF-IDF 후보 개수
            top_k (int): 최종 반환 결과 개수
            weight_tfidf (float): TF-IDF 가중치
            weight_emb (float): 임베딩 가중치  
            weight_jac (float): Jaccard 가중치
            
        Returns:
            pd.DataFrame: 검색 결과 데이터프레임
        """
        index_list_results = []
        score_list_results = []
        
        # keywords_data가 DataFrame인 경우
        if isinstance(keywords_data, pd.DataFrame):
            # 컬럼명 유연하게 처리
            if 'keyword' in keywords_data.columns:
                keyword_column = keywords_data['keyword']
            elif 'keywords' in keywords_data.columns:
                keyword_column = keywords_data['keywords']
            else:
                # 첫 번째 컬럼 사용
                keyword_column = keywords_data.iloc[:, 0]
        else:
            keyword_column = keywords_data
        
        for i in tqdm(range(len(keyword_column))):
            if isinstance(keywords_data, pd.DataFrame):
                keywords_list = self.string_list_to_actual_list(keyword_column.iloc[i])
            else:
                keywords_list = keyword_column[i] if isinstance(keyword_column[i], list) else [keyword_column[i]]
            
            # 하이브리드 검색 수행
            top_rows, scores = self.hybrid_search(
                keywords_list, 
                tfidf_k=tfidf_k, 
                top_k=top_k,
                weight_tfidf=weight_tfidf,
                weight_emb=weight_emb,
                weight_jac=weight_jac
            )
            
            # 결과 저장
            top_indices = top_rows['index'].tolist() if not top_rows.empty else []
            index_list_results.append(top_indices)
            score_list_results.append(scores)
        
        # 결과 DataFrame 생성
        results_df = pd.DataFrame({
            'index_list': index_list_results,
            'score_list': score_list_results
        })
        
        # 점수 분포 분석 추가
        self._analyze_score_distribution(score_list_results, method_name="Method 3")
        
        return results_df
    
    def _analyze_score_distribution(self, scores_list, method_name="Method 3"):
        """
        점수 분포 분석 및 출력
        
        Args:
            scores_list (list): 각 쿼리별 점수 리스트
            method_name (str): 방법명
        """
        all_scores = []
        for scores in scores_list:
            all_scores.extend(scores)
        
        if not all_scores:
            print(f"=== {method_name}: 점수 데이터 없음 ===")
            return
        
        print(f"=== {method_name}: 점수 분포 분석 ===")
        print(f"총 검색 결과: {len(all_scores)}개")
        print(f"점수 범위: {np.min(all_scores):.3f} ~ {np.max(all_scores):.3f}")
        print(f"평균: {np.mean(all_scores):.3f}, 중앙값: {np.median(all_scores):.3f}")
        print(f"표준편차: {np.std(all_scores):.3f}")
        print(f"임계값({self.score_threshold}) 이상: {len([s for s in all_scores if s >= self.score_threshold])}개")
        print(f"95th percentile: {np.percentile(all_scores, 95):.3f}")
        print(f"90th percentile: {np.percentile(all_scores, 90):.3f}")
        print(f"75th percentile: {np.percentile(all_scores, 75):.3f}")
        print("=" * 40)
    
    def process_keyword_file(self, keyword_file_path, **search_params):
        """
        키워드 파일 처리 및 검색
        
        Args:
            keyword_file_path (str): 키워드 파일 경로 (JSON 또는 CSV)
            **search_params: 검색 파라미터
            
        Returns:
            pd.DataFrame: 검색 결과 데이터프레임
        """
        # 파일 확장자에 따라 다르게 로드
        if keyword_file_path.endswith('.json'):
            with open(keyword_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON에서 키워드 추출
            keywords_list = []
            for item in data:
                if 'output' in item and 'keyword' in item['output']:
                    keyword_str = item['output']['keyword']
                    keywords = [k.strip() for k in keyword_str.split(',')]
                    keywords_list.append(keywords)
                else:
                    # 키워드가 없으면 질문에서 추출
                    question = item['input']['question']
                    keywords_list.append([question])
                    
            # 컬럼명을 'keyword'로 통일하여 batch_search와 일치시킴
            keyword_df = pd.DataFrame({'keyword': keywords_list})
            
        else:
            # CSV 파일 로드
            keyword_df = pd.read_csv(keyword_file_path)
        
        # 배치 검색 수행
        results_df = self.batch_search(keyword_df, **search_params)
        
        return results_df


def main():
    """사용 예시"""
    # 검색기 초기화
    searcher = HybridTFIDFEmbeddingSearch(
        embedding_model_name='Qwen/Qwen3-Embedding-4B',
        score_threshold=0.2
    )
    
    # 데이터 로드
    data_paths = {
        'hangeul_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
        'standard_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
        'spacing_rule': 'docs/띄어쓰기.json',
        'punctuation_rule': 'docs/문장 부호.json',
        'foreign_rule': 'docs/외래어  표기법.json'
    }
    searcher.load_data(data_paths)
    
    # 키워드 파일 처리
    results_df = searcher.process_keyword_file(
        'result/predictions_with_keywords.json',
        tfidf_k=50,
        top_k=10,
        weight_tfidf=0.5,
        weight_emb=0.3,
        weight_jac=0.2
    )
    
    print("검색 결과:")
    print(results_df.head())
    
    # 📁 결과 저장 추가
    output_file = 'result/method3_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과가 저장되었습니다: {output_file}")
    
    # 요약 정보도 저장
    summary_file = 'result/method3_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== Method 3: 하이브리드 TF-IDF + 임베딩 검색 결과 요약 ===\n")
        f.write(f"총 쿼리 수: {len(results_df)}\n")
        
        # 결과가 있는 쿼리 수 계산
        results_with_data = len([r for r in results_df['index_list'] if r])
        f.write(f"결과가 있는 쿼리: {results_with_data}\n")
        
        # 점수 통계 계산
        all_scores = []
        for scores in results_df['score_list']:
            all_scores.extend(scores)
        
        if all_scores:
            import numpy as np
            f.write(f"총 검색 결과: {len(all_scores)}개\n")
            f.write(f"평균 점수: {np.mean(all_scores):.3f}\n")
            f.write(f"중앙값 점수: {np.median(all_scores):.3f}\n")
            f.write(f"최고 점수: {np.max(all_scores):.3f}\n")
            f.write(f"최저 점수: {np.min(all_scores):.3f}\n")
            f.write(f"표준편차: {np.std(all_scores):.3f}\n")
            f.write(f"임계값({searcher.score_threshold}) 이상: {len([s for s in all_scores if s >= searcher.score_threshold])}개\n")
            f.write(f"95th percentile: {np.percentile(all_scores, 95):.3f}\n")
            f.write(f"90th percentile: {np.percentile(all_scores, 90):.3f}\n")
            f.write(f"75th percentile: {np.percentile(all_scores, 75):.3f}\n")
            
            # 점수 구간별 분포
            f.write(f"\n점수 구간별 분포:\n")
            f.write(f"0.8 이상: {len([s for s in all_scores if s >= 0.8])}개\n")
            f.write(f"0.6-0.8: {len([s for s in all_scores if 0.6 <= s < 0.8])}개\n")
            f.write(f"0.4-0.6: {len([s for s in all_scores if 0.4 <= s < 0.6])}개\n")
            f.write(f"0.2-0.4: {len([s for s in all_scores if 0.2 <= s < 0.4])}개\n")
        else:
            f.write("검색 결과 없음\n")
    
    print(f"✅ 요약 정보가 저장되었습니다: {summary_file}")
    
    # 하이브리드 검색 설정 정보도 저장
    config_file = 'result/method3_config.txt'
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("=== Method 3: 하이브리드 검색 설정 정보 ===\n")
        f.write(f"임베딩 모델: {searcher.embedding_model_name}\n")
        f.write(f"디바이스: {searcher.device}\n")
        f.write(f"점수 임계값: {searcher.score_threshold}\n")
        f.write(f"\n검색 파라미터:\n")
        f.write(f"TF-IDF 후보 수 (tfidf_k): 50\n")
        f.write(f"최종 결과 수 (top_k): 10\n")
        f.write(f"\n가중치 설정:\n")
        f.write(f"TF-IDF 가중치: 0.5\n")
        f.write(f"임베딩 가중치: 0.3\n")
        f.write(f"Jaccard 가중치: 0.2\n")
        f.write(f"\n데이터 정보:\n")
        f.write(f"규정 데이터 수: {len(searcher.df)}개\n")
        f.write(f"TF-IDF 벡터 차원: {searcher.tfidf_matrix.shape}\n")
        
        # 입력 파일 정보
        import os
        input_file = 'result/predictions_with_keywords.json'
        if os.path.exists(input_file):
            f.write(f"입력 파일: {input_file}\n")
            f.write(f"입력 파일 크기: {os.path.getsize(input_file)} bytes\n")
    
    print(f"✅ 검색 설정 정보가 저장되었습니다: {config_file}")
    
    # 샘플 결과 출력
    print(f"\n📊 검색 결과 샘플 (처음 3개):")
    for i in range(min(3, len(results_df))):
        indices = results_df.iloc[i]['index_list']
        scores = results_df.iloc[i]['score_list']
        print(f"Query {i+1}: {len(indices)}개 결과, 최고 점수: {max(scores):.3f}" if scores else f"Query {i+1}: 결과 없음")
    
    return results_df


if __name__ == "__main__":
    results = main()
