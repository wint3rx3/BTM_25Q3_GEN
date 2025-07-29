"""
Method 1: 룰 베이스 기반 키워드 검색
- 형태소 분석기(KoNLPy Okt) 사용
- 명사, 동사, 형용사만 추출하여 키워드 생성
- 하이브리드 점수 계산 시스템
- 임계값 기반 필터링 (70점 이상)
"""

import json
import re
import pandas as pd
import numpy as np
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KeywordRuleBasedSearch:
    def __init__(self, data_paths=None, threshold=70):
        """
        키워드 룰 베이스 검색 초기화
        
        Args:
            data_paths (dict): 데이터 파일 경로들
            threshold (float): 점수 임계값 (기본값: 70, 0~100 범위)
        """
        self.okt = Okt()
        self.threshold = threshold
        self.df = None
        self.entire_examples_keyword_no_dash = []
        self.entire_examples_no_dash_with_index = []
        self.example_to_indices = {}
        
        if data_paths:
            self.load_data(data_paths)
    
    def load_data(self, data_paths):
        """
        규정 데이터 로드 및 전처리
        
        Args:
            data_paths (dict): 데이터 파일 경로 딕셔너리
                - hangeul_rule: 한글 맞춤법 규정 파일 경로
                - standard_rule: 표준어 규정 파일 경로  
                - spacing_rule: 띄어쓰기 규정 파일 경로
        """
        all_data = []
        
        # 각 규정 파일 로드
        for key, path in data_paths.items():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        
        # DataFrame 생성 및 전처리
        self.df = pd.DataFrame(all_data)
        self.df['keywords'] = self.df['keywords'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
        self.df['conditions'] = self.df['conditions'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
        self.df['index'] = self.df.index
        
        # 예시 데이터 전처리
        self._preprocess_examples()
    
    def _preprocess_examples(self):
        """예시 데이터 전처리 및 인덱싱"""
        self.entire_examples_keyword_no_dash = []
        self.entire_examples_no_dash_with_index = []
        
        for index, examples in enumerate(self.df['examples']):
            if isinstance(examples, dict):
                all_examples = examples.get('correct', []) + examples.get('incorrect', [])
                for example in all_examples:
                    example_no_dash = example.replace('-', '')
                    self.entire_examples_keyword_no_dash.append(example_no_dash)
                    self.entire_examples_no_dash_with_index.append((example_no_dash, index))
        
        # 예시 -> 인덱스 매핑 생성
        self.example_to_indices = {}
        for example_no_dash, original_index in self.entire_examples_no_dash_with_index:
            self.example_to_indices.setdefault(example_no_dash, []).append(original_index)
    
    def extract_keywords(self, text, top_k=3):
        """
        주요 단어 추출 (명사, 동사, 형용사만)
        
        Args:
            text (str): 입력 텍스트
            top_k (int): 추출할 키워드 개수
            
        Returns:
            list: 추출된 키워드 리스트
        """
        pos_tags = self.okt.pos(text)
        keywords = [word for word, pos in pos_tags if pos in ['Noun', 'Verb', 'Adjective']]
        return keywords[:top_k]
    
    def hybrid_search(self, keywords, examples, top_k=3):
        """
        하이브리드 검색 (완전일치, 부분일치, 형태적 유사성 등)
        
        Args:
            keywords (list): 검색 키워드 리스트
            examples (list): 검색 대상 예시 리스트
            top_k (int): 반환할 상위 결과 개수
            
        Returns:
            list: 각 키워드별 매칭 결과 리스트
        """
        results = []
        
        for keyword in keywords:
            matches = []
            
            for example in examples:
                score = 0
                
                # 1. 완전 키워드 일치 (높은 점수)
                if keyword == example:
                    score += 100
                elif keyword in example or example in keyword:
                    score += 50
                
                # 2. 형태적 유사성
                if keyword in example or example in keyword:
                    score += 20
                
                # 공통 문자 개수
                common_chars = len(set(keyword) & set(example))
                if common_chars > 0:
                    score += common_chars * 2
                
                # 길이 유사성
                len_diff = abs(len(keyword) - len(example))
                if len_diff <= 2:
                    score += (3 - len_diff) * 5
                
                # 접두사/접미사 일치
                if len(keyword) >= 2 and len(example) >= 2:
                    if keyword.startswith(example[:2]) or example.startswith(keyword[:2]):
                        score += 15
                    if keyword.endswith(example[-2:]) or example.endswith(keyword[-2:]):
                        score += 15
                
                if score > 0:
                    matches.append((example, score))
            
            # 점수 기준 정렬 및 상위 k개 선택
            matches.sort(key=lambda x: x[1], reverse=True)
            results.append(matches[:top_k])
        
        return results
    
    def search(self, sentence, top_k_keywords=3, top_k_matches=10):
        """
        문장에 대한 검색 수행
        
        Args:
            sentence (str): 검색할 문장
            top_k_keywords (int): 추출할 키워드 개수
            top_k_matches (int): 각 키워드별 매칭 개수
            
        Returns:
            dict: 검색 결과 딕셔너리
        """
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요.")
        
        # 키워드 추출
        keywords = self.extract_keywords(sentence, top_k=top_k_keywords)
        
        # 하이브리드 검색
        results = self.hybrid_search(
            keywords, 
            self.entire_examples_keyword_no_dash, 
            top_k=top_k_matches
        )
        
        # 인덱스별 최고 점수 집계
        index2best = {}
        
        for matches in results:
            for ex, score in matches:
                if score < self.threshold:
                    continue
                
                for original_index in self.example_to_indices.get(ex, []):
                    prev = index2best.get(original_index, -np.inf)
                    if score > prev:
                        index2best[original_index] = score
        
        # 점수 내림차순으로 정렬
        sorted_items = sorted(index2best.items(), key=lambda x: x[1], reverse=True)
        retrieved_indices = [idx for idx, sc in sorted_items]
        retrieved_scores = [float(sc) for idx, sc in sorted_items]
        
        return {
            'input_sentence': sentence,
            'keywords': keywords,
            'retrieved_index': retrieved_indices,
            'retrieved_scores': retrieved_scores,
            'total_candidates': len(sorted_items),
            'above_threshold': len([s for s in retrieved_scores if s >= self.threshold])
        }
    
    def analyze_score_distribution(self, results_df):
        """
        검색 결과의 점수 분포 분석
        
        Args:
            results_df (pd.DataFrame): 검색 결과 데이터프레임
            
        Returns:
            dict: 분포 통계
        """
        all_scores = []
        # 통일된 컬럼명 사용
        score_column = 'retrieved_scores'
        if score_column in results_df.columns:
            for scores in results_df[score_column]:
                all_scores.extend(scores)
        
        if not all_scores:
            return {"message": "점수 데이터가 없습니다."}
        
        import numpy as np
        
        stats = {
            "total_results": len(all_scores),
            "min_score": float(np.min(all_scores)),
            "max_score": float(np.max(all_scores)),
            "mean_score": float(np.mean(all_scores)),
            "median_score": float(np.median(all_scores)),
            "std_score": float(np.std(all_scores)),
            "percentiles": {
                "25th": float(np.percentile(all_scores, 25)),
                "75th": float(np.percentile(all_scores, 75)),
                "90th": float(np.percentile(all_scores, 90)),
                "95th": float(np.percentile(all_scores, 95))
            },
            "threshold_analysis": {
                "above_70": len([s for s in all_scores if s >= 70]),
                "above_80": len([s for s in all_scores if s >= 80]),
                "above_90": len([s for s in all_scores if s >= 90]),
                "perfect_100": len([s for s in all_scores if s == 100])
            }
        }
        
        return stats

    def batch_search(self, sentences):
        """
        여러 문장에 대한 배치 검색
        
        Args:
            sentences (list): 검색할 문장 리스트
            
        Returns:
            pd.DataFrame: 검색 결과 데이터프레임 (통일된 형태)
        """
        results = []
        for i, sentence in enumerate(sentences):
            result = self.search(sentence)
            # 통일된 형태로 변환
            unified_result = {
                'query_id': i,
                'input_text': sentence,
                'retrieved_indices': result['retrieved_index'],
                'retrieved_scores': result['retrieved_scores'],
                'method': 'method1_keyword_rule_based'
            }
            results.append(unified_result)
        
        df = pd.DataFrame(results)
        
        # 분포 분석 추가
        score_stats = self.analyze_score_distribution(df)
        print("=== Method 1: 점수 분포 분석 ===")
        print(f"총 검색 결과: {score_stats.get('total_results', 0)}개")
        print(f"점수 범위: {score_stats.get('min_score', 0):.1f} ~ {score_stats.get('max_score', 0):.1f}")
        print(f"평균: {score_stats.get('mean_score', 0):.1f}, 중앙값: {score_stats.get('median_score', 0):.1f}")
        if 'threshold_analysis' in score_stats and isinstance(score_stats['threshold_analysis'], dict):
            threshold_analysis = score_stats['threshold_analysis']
            print(f"임계값(70) 이상: {threshold_analysis['above_70']}개")
            print(f"90점 이상: {threshold_analysis['above_90']}개")
            print(f"만점(100): {threshold_analysis['perfect_100']}개")
        print("=" * 40)
        
        return df
    
    def process_llm_predictions(self, predictions_path):
        """
        LLM 예측 결과에서 '~이 옳다', '~가 옳다' 부분 추출 및 검색
        
        Args:
            predictions_path (str): LLM 예측 결과 JSON 파일 경로
            
        Returns:
            pd.DataFrame: 검색 결과 데이터프레임
        """
        with open(predictions_path, 'r', encoding='utf-8') as f:
            llm_prediction = json.load(f)
        
        # '~이 옳다', '~가 옳다' 부분 추출
        processed_sentences = []
        for pred in llm_prediction:
            answer = pred['output']['answer']
            
            if '가 옳다.' in answer:
                input_text = answer.split('가 옳다.')[0]
            elif '이 옳다.' in answer:
                input_text = answer.split('이 옳다.')[0]
            else:
                input_text = answer
            
            processed_sentences.append(input_text.strip('"'))
        
        # 배치 검색 수행
        return self.batch_search(processed_sentences)


def main():
    """사용 예시"""
    # 데이터 파일 경로 설정
    data_paths = {
        'hangeul_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
        'standard_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
        'spacing_rule': 'docs/띄어쓰기.json',
        'punctuation_rule': 'docs/문장 부호.json',
        'foreign_rule': 'docs/외래어  표기법.json'
    }
    
    # 검색기 초기화 (임계값 70으로 유지)
    searcher = KeywordRuleBasedSearch(data_paths, threshold=70)
    
    # LLM 예측 결과 처리
    results_df = searcher.process_llm_predictions('result/predictions.json')
    
    print("검색 결과:")
    print(results_df.head())
    
    # 📁 결과 저장 추가
    output_file = 'result/RAG_result/method1_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과가 저장되었습니다: {output_file}")
    
    # 통일된 형태의 CSV도 저장 (교집합 분석용)
    try:
        from unified_csv_utils import create_unified_csv
        unified_output = 'result/RAG_result/method1_unified.csv'
        create_unified_csv(results_df, 'method1', unified_output)
    except ImportError:
        print("⚠️ unified_csv_utils를 찾을 수 없습니다. 통일된 CSV는 생성되지 않습니다.")
    
    return results_df


if __name__ == "__main__":
    results = main()
