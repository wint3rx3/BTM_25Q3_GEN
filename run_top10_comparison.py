"""
간단한 실행 스크립트 - 상위 10개 결과 비교
각 방법별로 상위 10개 결과를 뽑고 점수 분포를 확인합니다.
"""

import sys
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def run_method1():
    """Method 1 실행"""
    print("=" * 50)
    print("Method 1: 룰베이스 키워드 검색 (상위 10개)")
    print("=" * 50)
    
    try:
        from method1_keyword_rule_based_search import KeywordRuleBasedSearch
        
    # 데이터 경로 설정
    data_paths = {
        'hangeul_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
        'standard_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
        'spacing_rule': 'docs/띄어쓰기.json',
        'punctuation_rule': 'docs/문장 부호.json',
        'foreign_rule': 'docs/외래어  표기법.json'
    }
    
    # 검색기 초기화 (임계값 낮춤 - 더 많은 결과 확인용)
    searcher = KeywordRuleBasedSearch(data_paths, threshold=60)
    
    # 예측 결과 처리
    results = searcher.process_llm_predictions('result/predictions.json')
    
    print(f"총 쿼리 수: {len(results)}")
        
        # 결과가 있는 쿼리 수
        queries_with_results = len([r for r in results['retrieved_index'] if r])
        print(f"결과가 있는 쿼리: {queries_with_results}")
        
        # 샘플 결과 출력
        print("\n샘플 결과 (처음 3개):")
        for i in range(min(3, len(results))):
            keywords = results.iloc[i]['keywords']
            indices = results.iloc[i]['retrieved_index'][:10]  # 상위 10개만
            scores = results.iloc[i]['retrieved_scores'][:10]   # 상위 10개만
            
            print(f"\n쿼리 {i+1}:")
            print(f"  키워드: {keywords}")
            print(f"  검색 결과: {len(indices)}개")
            if scores:
                print(f"  점수 범위: {min(scores):.1f} ~ {max(scores):.1f}")
        
        # 결과 저장
        results.to_csv('method1_top10_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n✓ 결과 저장: method1_top10_results.csv")
        
        return results
        
    except Exception as e:
        print(f"✗ Method 1 실행 실패: {e}")
        return None


def run_method2():
    """Method 2 실행"""
    print("=" * 50)
    print("Method 2: 의미 기반 임베딩 검색 (상위 10개)")
    print("=" * 50)
    
    try:
        from method2_semantic_embedding_search import SemanticEmbeddingSearch
        
        # 데이터 경로 설정
        data_paths = {
            'hangeul_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
            'standard_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
            'spacing_rule': 'docs/띄어쓰기.json'
        }
        
        # 검색기 초기화
        # .env에서 HF_TOKEN 가져오기
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print("⚠️ 경고: .env 파일에 HF_TOKEN이 설정되지 않았습니다.")
        
        searcher = SemanticEmbeddingSearch(
            model_name='Qwen/Qwen3-Embedding-4B',
            hf_token=hf_token,
            score_threshold=0.3  # 임계값 추가
        )
        
        # 데이터 로드
        searcher.load_data(data_paths)
        
        # FAISS 인덱스 구축
        vectordb_dir = "./vectordb_rules_top10"
        if not os.path.exists(vectordb_dir):
            print("FAISS 인덱스 구축 중...")
            searcher.build_faiss_index(vectordb_dir)
        else:
            print("기존 FAISS 인덱스 사용")
        
        # 예측 결과 처리 (상위 10개)
        results = searcher.process_llm_predictions(
            'result/predictions.json',
            vectordb_dir,
            top_k=10
        )
        
        print(f"총 쿼리 수: {len(results)}")
        
        # 샘플 결과 출력
        print("\n샘플 결과 (처음 3개):")
        for i in range(min(3, len(results))):
            reason = results.iloc[i]['reason'][:100] + "..." if len(results.iloc[i]['reason']) > 100 else results.iloc[i]['reason']
            indices = results.iloc[i]['top_10_rule_index']
            scores = results.iloc[i]['top_10_scores']
            
            print(f"\n쿼리 {i+1}:")
            print(f"  교정사유: {reason}")
            print(f"  검색 결과: {len(indices)}개")
            if scores:
                print(f"  점수 범위: {min(scores):.3f} ~ {max(scores):.3f}")
        
        # 결과 저장
        results.to_csv('method2_top10_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n✓ 결과 저장: method2_top10_results.csv")
        
        return results
        
    except Exception as e:
        print(f"✗ Method 2 실행 실패: {e}")
        return None


def run_method3():
    """Method 3 실행"""
    print("=" * 50)
    print("Method 3: 하이브리드 TF-IDF + 임베딩 검색 (상위 10개)")
    print("=" * 50)
    
    try:
        from method3_hybrid_tfidf_embedding_search import HybridTFIDFEmbeddingSearch
        
        # 키워드 파일 확인
        keyword_file = 'result/predictions_with_keywords.json'
        if not os.path.exists(keyword_file):
            print(f"✗ 키워드 파일을 찾을 수 없습니다: {keyword_file}")
            return None
        
        # 데이터 경로 설정
        data_paths = {
            'hangeul_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
            'standard_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
            'spacing_rule': 'docs/띄어쓰기.json'
        }
        
        # 검색기 초기화
        searcher = HybridTFIDFEmbeddingSearch(
            embedding_model_name='Qwen/Qwen3-Embedding-4B',
            score_threshold=0.2  # 임계값 추가
        )
        
        # 데이터 로드
        searcher.load_data(data_paths)
        
        # 키워드 파일 처리 (상위 10개)
        results = searcher.process_keyword_file(
            keyword_file,
            tfidf_k=50,
            top_k=10,
            weight_tfidf=0.5,
            weight_emb=0.3,
            weight_jac=0.2
        )
        
        print(f"총 쿼리 수: {len(results)}")
        
        # 샘플 결과 출력
        print("\n샘플 결과 (처음 3개):")
        for i in range(min(3, len(results))):
            indices = results.iloc[i]['index_list']
            scores = results.iloc[i]['score_list']
            
            print(f"\n쿼리 {i+1}:")
            print(f"  검색 결과: {len(indices)}개")
            if scores:
                print(f"  점수 범위: {min(scores):.3f} ~ {max(scores):.3f}")
        
        # 결과 저장
        results.to_csv('method3_top10_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n✓ 결과 저장: method3_top10_results.csv")
        
        return results
        
    except Exception as e:
        print(f"✗ Method 3 실행 실패: {e}")
        return None


def main():
    """메인 실행 함수"""
    print("세 가지 RAG 방법 상위 10개 결과 추출 시작")
    print("=" * 60)
    
    results = {}
    
    # Method 1 실행
    results['method1'] = run_method1()
    
    print("\n" + "="*60 + "\n")
    
    # Method 2 실행
    results['method2'] = run_method2()
    
    print("\n" + "="*60 + "\n")
    
    # Method 3 실행 
    results['method3'] = run_method3()
    
    # 전체 요약
    print("\n" + "=" * 60)
    print("전체 실행 요약")
    print("=" * 60)
    
    for method, result in results.items():
        if result is not None:
            print(f"✓ {method}: 성공 ({len(result)}개 쿼리)")
        else:
            print(f"✗ {method}: 실패")
    
    print("\n모든 방법의 상위 10개 결과 추출 완료!")
    print("각 방법별 CSV 파일이 생성되었습니다.")
    
    return results


if __name__ == "__main__":
    results = main()
