import pandas as pd
import numpy as np
import json
import ast
import os
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset, Features, Value
from tqdm import tqdm

class RAGDocumentSelector:
    """교집합 개수에 따라 최적의 문서 조합을 선택하는 클래스"""
    
    def __init__(self, method1_path: str, method2_path: str, method3_path: str):
        """
        Args:
            method1_path: 키워드 룰 베이스 결과 CSV 경로
            method2_path: 의미 기반 임베딩 결과 CSV 경로 
            method3_path: 하이브리드 TF-IDF + 임베딩 결과 CSV 경로
        """
        self.method1_df = pd.read_csv(method1_path)
        self.method2_df = pd.read_csv(method2_path)
        self.method3_df = pd.read_csv(method3_path)
        
        # 교집합 계산
        self.intersection_df = self._calculate_intersections()
        
    def _extract_retrieved_docs(self, retrieved_indices_str):
        """retrieved_indices 문자열에서 문서 ID 리스트 추출"""
        try:
            if pd.isna(retrieved_indices_str):
                return []
            indices = ast.literal_eval(retrieved_indices_str)
            return indices
        except:
            return []
    
    def _calculate_intersections(self):
        """각 ID별로 세 방법 간의 문서 교집합 계산"""
        results = []
        
        # 공통 ID 찾기
        common_ids = set(self.method1_df['query_id']).intersection(
            set(self.method2_df['query_id']).intersection(set(self.method3_df['query_id']))
        )
        
        for query_id in common_ids:
            # 각 방법별 검색된 문서들
            m1_docs = set(self._extract_retrieved_docs(
                self.method1_df[self.method1_df['query_id'] == query_id]['retrieved_indices'].iloc[0]
            ))
            m2_docs = set(self._extract_retrieved_docs(
                self.method2_df[self.method2_df['query_id'] == query_id]['retrieved_indices'].iloc[0]
            ))
            m3_docs = set(self._extract_retrieved_docs(
                self.method3_df[self.method3_df['query_id'] == query_id]['retrieved_indices'].iloc[0]
            ))
            
            # 세 방법 모두의 교집합
            all_intersection = m1_docs.intersection(m2_docs).intersection(m3_docs)
            
            results.append({
                'query_id': query_id,
                'intersection_count': len(all_intersection),
                'intersection_docs': list(all_intersection),
                'm1_docs': list(m1_docs),
                'm2_docs': list(m2_docs), 
                'm3_docs': list(m3_docs)
            })
        
        return pd.DataFrame(results)
    
    def select_documents_for_query(self, query_id: int) -> List[int]:
        """
        쿼리 ID에 대해 교집합 개수에 따른 최적 문서 3개 선택
        
        Args:
            query_id: 쿼리 ID
            
        Returns:
            선택된 문서 ID 리스트 (최대 3개)
        """
        query_info = self.intersection_df[self.intersection_df['query_id'] == query_id].iloc[0]
        intersection_count = query_info['intersection_count']
        
        if intersection_count >= 3:
            # 교집합 3개 이상: 교집합 문서 3개 그대로 사용
            selected_docs = query_info['intersection_docs'][:3]
            
        elif intersection_count == 2:
            # 교집합 2개: 교집합 2개 + 하이브리드 방법 최상위 1개
            intersection_docs = query_info['intersection_docs']
            m3_row = self.method3_df[self.method3_df['query_id'] == query_id].iloc[0]
            m3_docs = self._extract_retrieved_docs(m3_row['retrieved_indices'])
            
            # 교집합에 없는 하이브리드 문서 중 최상위 1개 선택
            additional_docs = [doc for doc in m3_docs if doc not in intersection_docs]
            selected_docs = intersection_docs + additional_docs[:1]
            
        elif intersection_count == 1:
            # 교집합 1개: 교집합 1개 + 하이브리드 방법 최상위 2개
            intersection_docs = query_info['intersection_docs']
            m3_row = self.method3_df[self.method3_df['query_id'] == query_id].iloc[0]
            m3_docs = self._extract_retrieved_docs(m3_row['retrieved_indices'])
            
            # 교집합에 없는 하이브리드 문서 중 최상위 2개 선택
            additional_docs = [doc for doc in m3_docs if doc not in intersection_docs]
            selected_docs = intersection_docs + additional_docs[:2]
            
        else:
            # 교집합 0개: 하이브리드 최상위 2개 + 임베딩 최상위 1개
            m3_row = self.method3_df[self.method3_df['query_id'] == query_id].iloc[0]
            m2_row = self.method2_df[self.method2_df['query_id'] == query_id].iloc[0]
            
            m3_docs = self._extract_retrieved_docs(m3_row['retrieved_indices'])
            m2_docs = self._extract_retrieved_docs(m2_row['retrieved_indices'])
            
            # 하이브리드 최상위 2개
            hybrid_docs = m3_docs[:2]
            
            # 임베딩에서 하이브리드와 겹치지 않는 최상위 1개 선택
            embedding_additional = [doc for doc in m2_docs if doc not in hybrid_docs]
            selected_docs = hybrid_docs + embedding_additional[:1]
        
        return selected_docs[:3]  # 최대 3개까지만
    
    def get_query_text(self, query_id: int) -> str:
        """쿼리 ID에 해당하는 입력 텍스트 반환"""
        query_row = self.method1_df[self.method1_df['query_id'] == query_id].iloc[0]
        return query_row['input_text']


class QwenFinetunedRAGInference:
    """파인튜닝된 Qwen 모델을 사용한 RAG 추론 클래스"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: 파인튜닝된 Qwen 모델 경로
        """
        self.model_path = model_path
        
        # 문서 데이터 로드 (docs 폴더의 모든 JSON 파일을 하나로 통합)
        self.documents = self._load_documents()
        
        # 모델과 토크나이저 로드 (predict.py와 동일)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        
        # 파인튜닝된 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("파인튜닝된 모델을 로드했습니다.")
    
    def _load_documents(self) -> Dict[int, Dict[str, str]]:
        """docs 폴더의 모든 JSON 파일을 로드하여 통합 (RAG 검색과 동일한 순서로)"""
        import os
        import json
        import pandas as pd
        
        all_data = []
        
        # RAG 검색에서 사용하는 실제 데이터 경로와 순서 (중복 포함)
        data_paths = {
            'hangeul_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
            'standard_rule': 'docs/한글 맞춤법 , 표준어  규정.json',  # 중복!
            'spacing_rule': 'docs/띄어쓰기.json',
            'punctuation_rule': 'docs/문장 부호.json',
            'foreign_rule': 'docs/외래어  표기법.json'
        }
        
        # 각 규정 파일을 순서대로 로드 (RAG 검색과 동일)
        for key, relative_path in data_paths.items():
            # 코랩 환경에 맞게 경로 수정
            full_path = os.path.join("/content/BTM_25Q3_GEN", relative_path)
            print(f"파일 경로 확인: {full_path}")  # 디버깅용
            print(f"파일 존재 여부: {os.path.exists(full_path)}")  # 디버깅용
            
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
                    print(f"{key}: {len(data)}개 문서 로드")
            else:
                print(f"❌ 파일을 찾을 수 없습니다: {full_path}")
        
        # DataFrame 생성하여 인덱스 확인 (RAG 검색과 동일한 방식)
        df = pd.DataFrame(all_data)
        df['index'] = df.index
        
        # 인덱스를 기준으로 문서 딕셔너리 생성
        all_documents = {}
        for idx, row in df.iterrows():
            all_documents[idx] = {
                'original_id': row.get('id', ''),
                'rule': row.get('rule', ''),
                'content': row.get('content', ''),
                'keywords': row.get('keywords', []),
                'examples': row.get('examples', {}),
                'conditions': row.get('conditions', [])
            }
        
        print(f"총 {len(all_documents)}개의 문서를 로드했습니다.")
        print(f"문서 인덱스 범위: 0 ~ {len(all_documents)-1}")
        return all_documents
    
    def get_document_content(self, doc_id: int) -> str:
        """문서 ID에 해당하는 문서 내용 반환"""
        try:
            doc_info = self.documents[doc_id]
            # 여러 필드를 조합해서 완전한 문서 내용 생성
            content_parts = []
            
            if doc_info.get('original_id'):
                content_parts.append(f"규정 ID: {doc_info['original_id']}")
            
            if doc_info.get('rule'):
                content_parts.append(f"규정: {doc_info['rule']}")
                
            if doc_info.get('content'):
                content_parts.append(f"내용: {doc_info['content']}")
                
            if doc_info.get('keywords') and isinstance(doc_info['keywords'], list):
                content_parts.append(f"키워드: {', '.join(doc_info['keywords'])}")
                
            if doc_info.get('examples'):
                examples = doc_info['examples']
                if isinstance(examples, dict):
                    if examples.get('correct'):
                        content_parts.append(f"올바른 예: {', '.join(examples['correct'])}")
                    if examples.get('incorrect'):
                        content_parts.append(f"틀린 예: {', '.join(examples['incorrect'])}")
            
            return "\n".join(content_parts) if content_parts else f"문서 {doc_id}의 내용을 찾을 수 없습니다."
            
        except (KeyError, IndexError):
            return f"문서 {doc_id}를 찾을 수 없습니다."
    
    def create_rag_prompt(self, query: str, selected_doc_ids: List[int]) -> str:
        """RAG용 프롬프트 생성 (파인튜닝된 모델용)"""
        # 선택된 문서들의 내용 가져오기
        doc_contents = []
        for doc_id in selected_doc_ids:
            content = self.get_document_content(doc_id)
            if content:
                doc_contents.append(f"참고문서 {doc_id}: {content}")
        
        # 프롬프트 구성
        context = "\n\n".join(doc_contents)
        
        # 파인튜닝된 모델용 프롬프트 (기존 학습 형식과 유사하게)
        system_prompt = (
            "당신은 한국어 맞춤법·문장부호 교정 전문가입니다.\n"
            "다음 참고 문서들을 활용하여 질문에 답하세요.\n"
            "- '교정형': 틀린 부분을 바로잡고, 이유를 어문 규범 근거로 설명하세요.\n"
            "- '선택형': 보기 중 옳은 표현을 선택하고, 이유를 어문 규범 근거로 설명하세요.\n"
            "정답 문장을 자연스럽게 작성하세요."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"참고 문서:\n{context}\n\n질문: {query}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 256) -> str:
        """파인튜닝된 Qwen 모델로 답변 생성 (predict.py와 동일한 설정)"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=None,  # 명시적으로 None 설정
                top_p=None,        # 명시적으로 None 설정  
                top_k=None         # 명시적으로 None 설정
            )
        
        # 입력 부분 제거하고 생성된 답변만 추출
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return answer


def main():
    """메인 실행 함수 - 파인튜닝된 모델 버전 (대회 형식)"""
    
    # 파인튜닝된 Qwen RAG 추론기 초기화
    model_path = r"/content/BTM_25Q3_GEN/data/korean_language_rag_V1.0_test.json"
    
    print("파인튜닝된 Qwen 모델 로드 중...")
    qwen_rag = QwenFinetunedRAGInference(model_path)
    
    # 테스트 데이터 로드 (predict.py와 동일)
    test_path = "data/korean_language_rag_V1.0_test.json"
    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)

    features = Features({
        "id": Value("string"),
        "input": {
            "question": Value("string"),
            "question_type": Value("string")
        }
    })
    test_ds = Dataset.from_list(test_data).cast(features)
    
    # 문서 선택기 초기화 (RAG 검색 결과 활용)
    method1_path = r"/content/BTM_25Q3_GEN/result/RAG_result/method1_results.csv"
    method2_path = r"/content/BTM_25Q3_GEN/result/RAG_result/method2_results.csv"
    method3_path = r"/content/BTM_25Q3_GEN/result/RAG_result/method3_results.csv"
    
    print("문서 선택기 초기화 중...")
    doc_selector = RAGDocumentSelector(method1_path, method2_path, method3_path)
    
    # 결과 저장용 리스트
    results = []
    
    # 각 테스트 예제에 대해 RAG 추론 수행
    print("RAG 추론 시작...")
    for ex in tqdm(test_ds, desc="RAG 추론 진행"):
        # 테스트 ID를 RAG 검색 query_id로 변환 (예: "751" -> 1)
        test_id = int(ex["id"])
        query_id = test_id - 750  # RAG 검색 결과의 query_id는 1부터 시작
        
        # 해당 query_id에 대한 최적 문서 선택
        try:
            selected_docs = doc_selector.select_documents_for_query(query_id)
            
            # RAG 프롬프트 생성 및 추론
            prompt = qwen_rag.create_rag_prompt(ex["input"]["question"], selected_docs)
            answer = qwen_rag.generate_answer(prompt)
            
        except Exception as e:
            # RAG 검색 결과가 없는 경우 문서 없이 추론
            print(f"Warning: Query ID {query_id}에 대한 RAG 결과가 없습니다. 문서 없이 추론합니다.")
            prompt = qwen_rag.create_rag_prompt(ex["input"]["question"], [])
            answer = qwen_rag.generate_answer(prompt)
        
        # predictions.json과 동일한 구조로 저장 (메타데이터 제거)
        results.append({
            "id": ex["id"],
            "input": ex["input"],
            "output": {
                "answer": answer
            }
        })
    
    # 결과 저장 - predictions.json과 동일한 형식
    output_path = "predictions-finetuned.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ predictions-finetuned.json 생성 완료! (RAG Finetuned 모델)")
    print(f"총 {len(results)}개 테스트 예제 처리 완료")


if __name__ == "__main__":
    main()
