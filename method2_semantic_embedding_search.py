"""
Method 2: 의미 기반 임베딩 검색
- Qwen3-Embedding-4B 모델 사용
- 교정 사유 ↔ 규정(rule + conditions) 간 의미적 유사도 계산
- FAISS 벡터 데이터베이스 활용
- 코사인 유사도 기반 상위 K개 검색
"""

import os
import json
import torch
import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
from dotenv import load_dotenv
import re

# .env 파일 로드
load_dotenv()


class SemanticEmbeddingSearch:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-4B', device=None, hf_token=None, score_threshold=0.3):
        """
        의미 기반 임베딩 검색 초기화
        
        Args:
            model_name (str): 사용할 임베딩 모델명
            device (str): 사용할 디바이스 ('cuda', 'cpu', None=자동선택)
            hf_token (str): Hugging Face 토큰
            score_threshold (float): 유사도 점수 임계값 (기본값: 0.3)
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        
        # Hugging Face 로그인
        if hf_token:
            login(hf_token)
        
        # 모델 로드
        self.tokenizer, self.model, self.device = self._load_model()
        self.df = None
        self.vectordb_dir = None
    
    def _load_model(self):
        """임베딩 모델 로드"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).eval()
        
        model = model.to(self.device)
        print(f"모델 로드 완료: {self.model_name} on {self.device}")
        
        return tokenizer, model, self.device
    
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
        
        print(f"데이터 로드 완료: {len(self.df)}개 규정")
    
    def _to_text(self, rule, conditions):
        """
        규정과 조건을 하나의 텍스트로 결합
        
        Args:
            rule: 규정 텍스트
            conditions: 조건 텍스트
            
        Returns:
            str: 결합된 텍스트
        """
        def normalize(x):
            if isinstance(x, list):
                return " ".join(map(str, x))
            return "" if pd.isna(x) else str(x)
        
        r = normalize(rule)
        c = normalize(conditions)
        return (r + ("\n" + c if c else "")).strip()
    
    @torch.no_grad()
    def encode_texts(self, texts, batch_size=32, normalize=True):
        """
        텍스트 리스트를 임베딩 벡터로 변환
        
        Args:
            texts (list): 텍스트 리스트
            batch_size (int): 배치 크기
            normalize (bool): L2 정규화 여부
            
        Returns:
            np.ndarray: 임베딩 벡터 배열
        """
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=2048
            ).to(self.device)
            
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            attn_mask = inputs['attention_mask'].unsqueeze(-1)  # [B, T, 1]
            
            # Mean pooling
            summed = (last_hidden * attn_mask).sum(dim=1)
            counts = attn_mask.sum(dim=1).clamp(min=1)
            emb = (summed / counts).detach().float().cpu().numpy()  # [B, H]
            vecs.append(emb)
        
        X = np.vstack(vecs).astype('float32')
        if normalize:
            faiss.normalize_L2(X)  # 코사인 유사도 = 내적
        return X
    
    def encode_query(self, query):
        """
        단일 쿼리를 임베딩 벡터로 변환
        
        Args:
            query (str): 쿼리 텍스트
            
        Returns:
            np.ndarray: 임베딩 벡터
        """
        return self.encode_texts([query], batch_size=1, normalize=True)[0]
    
    def build_faiss_index(self, vectordb_dir):
        """
        FAISS 인덱스 구축 및 저장
        
        Args:
            vectordb_dir (str): 벡터 DB 저장 디렉토리
        """
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요.")
        
        os.makedirs(vectordb_dir, exist_ok=True)
        self.vectordb_dir = vectordb_dir
        
        # 메타데이터 및 검색용 텍스트 생성
        docs = []
        texts = []
        
        for _, row in self.df.iterrows():
            text = self._to_text(row.get('rule', ''), row.get('conditions', ''))
            docs.append({
                "index": int(row['index']),
                "rule": row.get('rule', ''),
                "conditions": row.get('conditions', ''),
                "text": text
            })
            texts.append(text)
        
        print("임베딩 계산 중...")
        # 임베딩 계산
        X = self.encode_texts(texts, batch_size=64, normalize=True)
        
        # FAISS 인덱스 생성 (코사인 유사도 = 내적)
        d = X.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(X)
        
        # 저장
        faiss.write_index(index, os.path.join(vectordb_dir, "index.faiss"))
        with open(os.path.join(vectordb_dir, "docs.jsonl"), "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        print(f"FAISS 인덱스 저장 완료: {vectordb_dir}")
    
    def load_faiss_index(self, vectordb_dir):
        """
        저장된 FAISS 인덱스 로드
        
        Args:
            vectordb_dir (str): 벡터 DB 디렉토리
            
        Returns:
            tuple: (FAISS 인덱스, 메타데이터 리스트)
        """
        index = faiss.read_index(os.path.join(vectordb_dir, "index.faiss"))
        docs = []
        with open(os.path.join(vectordb_dir, "docs.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))
        return index, docs
    
    def search(self, query, vectordb_dir=None, top_k=10):
        """
        단일 쿼리에 대한 검색
        
        Args:
            query (str): 검색 쿼리
            vectordb_dir (str): 벡터 DB 디렉토리 (None이면 기본값 사용)
            top_k (int): 반환할 상위 결과 개수
            
        Returns:
            list: 검색 결과 리스트
        """
        if vectordb_dir is None:
            vectordb_dir = self.vectordb_dir
        
        if vectordb_dir is None:
            raise ValueError("vectordb_dir이 설정되지 않았습니다.")
        
        # FAISS 인덱스 로드
        index, docs = self.load_faiss_index(vectordb_dir)
        
        # 쿼리 임베딩 및 검색
        q = self.encode_query(query).reshape(1, -1).astype('float32')
        D, I = index.search(q, top_k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx == -1:
                continue
            meta = docs[idx]
            results.append({
                "rank": rank + 1,
                "score": float(score),
                "rule_index": meta["index"],
                "rule": meta["rule"],
                "conditions": meta["conditions"]
            })
        
        return results
    
    def batch_search(self, queries, vectordb_dir=None, top_k=10):
        """
        여러 쿼리에 대한 배치 검색
        
        Args:
            queries (list): 검색 쿼리 리스트
            vectordb_dir (str): 벡터 DB 디렉토리
            top_k (int): 반환할 상위 결과 개수
            
        Returns:
            pd.DataFrame: 검색 결과 데이터프레임
        """
        if vectordb_dir is None:
            vectordb_dir = self.vectordb_dir
        
        # FAISS 인덱스 로드
        index, docs = self.load_faiss_index(vectordb_dir)
        
        # 배치 임베딩
        Q = self.encode_texts(queries, batch_size=64, normalize=True)
        
        # 배치 검색
        D, I = index.search(Q, top_k)
        
        top_k_idx_list = []
        top_k_scores_list = []
        
        for scores, idxs in zip(D, I):
            # 유효하지 않은 인덱스(-1) 제외 및 임계값 필터링
            filtered = [(i, s) for i, s in zip(idxs, scores) if i != -1 and s >= self.score_threshold]
            
            if not filtered:
                top_k_idx_list.append([])
                top_k_scores_list.append([])
                continue
            
            # 메타데이터에서 원본 rule 인덱스 추출
            rule_indices = [docs[i]["index"] for i, _ in filtered]
            rule_scores = [float(s) for _, s in filtered]
            
            top_k_idx_list.append(rule_indices)
            top_k_scores_list.append(rule_scores)
        
        # 결과 데이터프레임 생성
        results_df = pd.DataFrame()
        results_df[f'top_{top_k}_rule_index'] = top_k_idx_list
        results_df[f'top_{top_k}_scores'] = top_k_scores_list
        
        # 점수 분포 분석 추가
        self._analyze_score_distribution(top_k_scores_list, method_name="Method 2")
        
        return results_df
    
    def _analyze_score_distribution(self, scores_list, method_name="Method 2"):
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
        
        import numpy as np
        
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
    
    def process_llm_predictions(self, predictions_path, vectordb_dir=None, top_k=10):
        """
        LLM 예측 결과에서 교정 사유 추출 및 검색
        
        Args:
            predictions_path (str): LLM 예측 결과 JSON 파일 경로
            vectordb_dir (str): 벡터 DB 디렉토리
            top_k (int): 반환할 상위 결과 개수
            
        Returns:
            pd.DataFrame: 검색 결과 데이터프레임
        """
        with open(predictions_path, 'r', encoding='utf-8') as f:
            llm_prediction = json.load(f)
        
        # 교정 사유 추출
        id_list = []
        question_list = []
        reason_list = []
        
        for pred in llm_prediction:
            id_val = pred['id']
            question = pred['input']['question']
            ans = pred['output']['answer']
            
            # '옳다' 부분 추출
            pat = r'(?:^|(?<=[.!?…。！？]))\s*(?:(?:"[^"]*"|\'[^\']*\'|["][^"]*["]|['][^']*[']))?[^.!?…。！？]*옳다[^.!?…。！？]*[.!?…。！？]'
            m = re.search(pat, ans)
            if m:
                answer = m.group(0)
                # '옳다.' 이후 사유 추출
                m_reason = re.search(r'(?<=옳다\.)\s*(.*)$', ans, flags=re.S)
                reason = m_reason.group(1) if m_reason else ""
            else:
                reason = ans
            
            id_list.append(id_val)
            question_list.append(question)
            reason_list.append(reason)
        
        # 사유 DataFrame 생성
        df_reason = pd.DataFrame({
            'id': id_list,
            'question': question_list,
            'reason': reason_list
        })
        
        # 배치 검색 수행
        search_results = self.batch_search(
            df_reason['reason'].fillna("").astype(str).tolist(),
            vectordb_dir,
            top_k
        )
        
        # 결과 결합
        result_df = pd.concat([df_reason, search_results], axis=1)
        
        return result_df


def main():
    """사용 예시"""
    # 검색기 초기화 (.env에서 HF_TOKEN 로드)
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("⚠️ 경고: .env 파일에 HF_TOKEN이 설정되지 않았습니다.")
    
    searcher = SemanticEmbeddingSearch(
        model_name='Qwen/Qwen3-Embedding-4B',
        hf_token=hf_token
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
    
    # FAISS 인덱스 구축
    vectordb_dir = "./vectordb_rules"
    searcher.build_faiss_index(vectordb_dir)
    
    # LLM 예측 결과 처리
    results_df = searcher.process_llm_predictions(
        'result/predictions.json',
        vectordb_dir,
        top_k=10
    )
    
    print("검색 결과:")
    print(results_df.head())
    
    # 📁 결과 저장 추가
    output_file = 'result/method2_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과가 저장되었습니다: {output_file}")
    
    # 요약 정보도 저장
    summary_file = 'result/method2_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== Method 2: 의미 기반 임베딩 검색 결과 요약 ===\n")
        f.write(f"총 쿼리 수: {len(results_df)}\n")
        
        # 결과가 있는 쿼리 수 계산
        results_with_data = len([r for r in results_df['top_10_rule_index'] if r])
        f.write(f"결과가 있는 쿼리: {results_with_data}\n")
        
        # 점수 통계 계산
        all_scores = []
        for scores in results_df['top_10_scores']:
            all_scores.extend(scores)
        
        if all_scores:
            import numpy as np
            f.write(f"총 검색 결과: {len(all_scores)}개\n")
            f.write(f"평균 점수: {np.mean(all_scores):.3f}\n")
            f.write(f"중앙값 점수: {np.median(all_scores):.3f}\n")
            f.write(f"최고 점수: {np.max(all_scores):.3f}\n")
            f.write(f"최저 점수: {np.min(all_scores):.3f}\n")
            f.write(f"임계값({searcher.score_threshold}) 이상: {len([s for s in all_scores if s >= searcher.score_threshold])}개\n")
            f.write(f"95th percentile: {np.percentile(all_scores, 95):.3f}\n")
            f.write(f"90th percentile: {np.percentile(all_scores, 90):.3f}\n")
        else:
            f.write("검색 결과 없음\n")
    
    print(f"✅ 요약 정보가 저장되었습니다: {summary_file}")
    
    # 벡터 DB 정보도 저장
    vectordb_info_file = 'result/method2_vectordb_info.txt'
    with open(vectordb_info_file, 'w', encoding='utf-8') as f:
        f.write("=== Method 2: 벡터 데이터베이스 정보 ===\n")
        f.write(f"벡터 DB 경로: {vectordb_dir}\n")
        f.write(f"사용 모델: {searcher.model_name}\n")
        f.write(f"디바이스: {searcher.device}\n")
        f.write(f"임계값: {searcher.score_threshold}\n")
        
        # 인덱스 파일 존재 확인
        import os
        index_file = os.path.join(vectordb_dir, "index.faiss")
        docs_file = os.path.join(vectordb_dir, "docs.jsonl")
        f.write(f"인덱스 파일 존재: {os.path.exists(index_file)}\n")
        f.write(f"문서 파일 존재: {os.path.exists(docs_file)}\n")
        
        if os.path.exists(index_file):
            f.write(f"인덱스 파일 크기: {os.path.getsize(index_file)} bytes\n")
        if os.path.exists(docs_file):
            f.write(f"문서 파일 크기: {os.path.getsize(docs_file)} bytes\n")
    
    print(f"✅ 벡터 DB 정보가 저장되었습니다: {vectordb_info_file}")
    
    return results_df


if __name__ == "__main__":
    results = main()
