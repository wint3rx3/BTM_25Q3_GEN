"""
통일된 점수 체계(0~1)로 세 방법 비교
- Method 1: 정규화된 하이브리드 점수 (0~1)
- Method 2: 코사인 유사도 (0~1) 
- Method 3: 가중치 결합 점수 (0~1)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from method1_keyword_rule_based_search import KeywordRuleBasedSearch
from method2_semantic_embedding_search import SemanticEmbeddingSearch
from method3_hybrid_tfidf_embedding_search import HybridTFIDFEmbeddingSearch

# .env 파일 로드
load_dotenv()

def run_all_methods():
    """세 방법 모두 실행하여 결과 비교"""
    
    # 공통 데이터 경로
    data_paths = {
        'hangeul_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
        'standard_rule': 'docs/한글 맞춤법 , 표준어  규정.json',
        'spacing_rule': 'docs/띄어쓰기.json',
        'punctuation_rule': 'docs/문장 부호.json',
        'foreign_rule': 'docs/외래어  표기법.json'
    }
    
    print("🚀 통일된 점수 체계(0~1)로 세 방법 실행 시작...")
    print("=" * 60)
    
    # Method 1: 정규화된 룰베이스 검색
    print("📌 Method 1: 정규화된 키워드 룰베이스 검색")
    method1 = KeywordRuleBasedSearch(data_paths, threshold=0.7)  # 0.7로 변경
    results1 = method1.process_llm_predictions('result/predictions.json')
    results1.to_csv('method1_normalized_results.csv', index=False)
    print("✅ Method 1 완료\n")
    
    # Method 2: 의미 기반 임베딩 검색
    print("📌 Method 2: 의미 기반 임베딩 검색")
    
    # .env에서 HF_TOKEN 가져오기
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("⚠️ 경고: .env 파일에 HF_TOKEN이 설정되지 않았습니다.")
    
    method2 = SemanticEmbeddingSearch(
        model_name='Qwen/Qwen3-Embedding-4B',
        hf_token=hf_token,
        score_threshold=0.3
    )
    method2.load_data(data_paths)
    vectordb_dir = "./vectordb_rules"
    method2.build_faiss_index(vectordb_dir)
    results2 = method2.process_llm_predictions('result/predictions.json', vectordb_dir, top_k=10)
    results2.to_csv('method2_results_top10.csv', index=False)
    print("✅ Method 2 완료\n")
    
    # Method 3: 하이브리드 TF-IDF + 임베딩 검색
    print("📌 Method 3: 하이브리드 TF-IDF + 임베딩 검색")
    method3 = HybridTFIDFEmbeddingSearch(
        embedding_model_name='Qwen/Qwen3-Embedding-4B',
        score_threshold=0.2
    )
    method3.load_data(data_paths)
    results3 = method3.process_keyword_file(
        'result/predictions_with_keywords.json',
        tfidf_k=50, top_k=10
    )
    results3.to_csv('method3_results_top10.csv', index=False)
    print("✅ Method 3 완료\n")
    
    print("🎯 모든 방법 실행 완료!")
    return results1, results2, results3

def analyze_unified_scores(results1, results2, results3):
    """통일된 점수 체계로 분석"""
    
    print("\n📊 통일된 점수 체계 분석 (0~1 범위)")
    print("=" * 50)
    
    # Method 1 점수 추출
    method1_scores = []
    for scores in results1['retrieved_scores']:
        method1_scores.extend(scores)
    
    # Method 2 점수 추출  
    method2_scores = []
    for scores in results2['top_10_scores']:
        method2_scores.extend(scores)
    
    # Method 3 점수 추출
    method3_scores = []
    for scores in results3['score_list']:
        method3_scores.extend(scores)
    
    # 비교 표 생성
    comparison_data = {
        'Method': ['Method 1 (정규화)', 'Method 2 (임베딩)', 'Method 3 (하이브리드)'],
        'Total Results': [len(method1_scores), len(method2_scores), len(method3_scores)],
        'Min Score': [f"{np.min(method1_scores):.3f}" if method1_scores else "N/A",
                     f"{np.min(method2_scores):.3f}" if method2_scores else "N/A", 
                     f"{np.min(method3_scores):.3f}" if method3_scores else "N/A"],
        'Max Score': [f"{np.max(method1_scores):.3f}" if method1_scores else "N/A",
                     f"{np.max(method2_scores):.3f}" if method2_scores else "N/A",
                     f"{np.max(method3_scores):.3f}" if method3_scores else "N/A"],
        'Mean': [f"{np.mean(method1_scores):.3f}" if method1_scores else "N/A",
                f"{np.mean(method2_scores):.3f}" if method2_scores else "N/A",
                f"{np.mean(method3_scores):.3f}" if method3_scores else "N/A"],
        'Std': [f"{np.std(method1_scores):.3f}" if method1_scores else "N/A",
               f"{np.std(method2_scores):.3f}" if method2_scores else "N/A", 
               f"{np.std(method3_scores):.3f}" if method3_scores else "N/A"],
        'Above 0.7': [len([s for s in method1_scores if s >= 0.7]),
                     len([s for s in method2_scores if s >= 0.7]),
                     len([s for s in method3_scores if s >= 0.7])],
        'Above 0.9': [len([s for s in method1_scores if s >= 0.9]),
                     len([s for s in method2_scores if s >= 0.9]),
                     len([s for s in method3_scores if s >= 0.9])]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 히스토그램
    plt.subplot(1, 3, 1)
    if method1_scores:
        plt.hist(method1_scores, bins=20, alpha=0.7, label='Method 1', color='blue')
    if method2_scores:
        plt.hist(method2_scores, bins=20, alpha=0.7, label='Method 2', color='red')
    if method3_scores:
        plt.hist(method3_scores, bins=20, alpha=0.7, label='Method 3', color='green')
    plt.xlabel('Score (0~1)')
    plt.ylabel('Frequency')
    plt.title('Score Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 박스플롯
    plt.subplot(1, 3, 2)
    all_scores = []
    labels = []
    if method1_scores:
        all_scores.append(method1_scores)
        labels.append('Method 1')
    if method2_scores:
        all_scores.append(method2_scores)
        labels.append('Method 2')
    if method3_scores:
        all_scores.append(method3_scores)
        labels.append('Method 3')
    
    if all_scores:
        plt.boxplot(all_scores, labels=labels)
        plt.ylabel('Score (0~1)')
        plt.title('Score Distribution (Box Plot)')
        plt.grid(True, alpha=0.3)
    
    # 임계값별 비교
    plt.subplot(1, 3, 3)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    method1_counts = [len([s for s in method1_scores if s >= t]) for t in thresholds] if method1_scores else [0]*5
    method2_counts = [len([s for s in method2_scores if s >= t]) for t in thresholds] if method2_scores else [0]*5
    method3_counts = [len([s for s in method3_scores if s >= t]) for t in thresholds] if method3_scores else [0]*5
    
    x = np.arange(len(thresholds))
    width = 0.25
    
    plt.bar(x - width, method1_counts, width, label='Method 1', color='blue', alpha=0.7)
    plt.bar(x, method2_counts, width, label='Method 2', color='red', alpha=0.7)
    plt.bar(x + width, method3_counts, width, label='Method 3', color='green', alpha=0.7)
    
    plt.xlabel('Threshold')
    plt.ylabel('Count Above Threshold')
    plt.title('Results Above Thresholds')
    plt.xticks(x, [f'≥{t}' for t in thresholds])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('unified_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 비교 결과 저장
    df_comparison.to_csv('unified_score_comparison.csv', index=False)
    
    return df_comparison

def main():
    """메인 실행 함수"""
    try:
        # 모든 방법 실행
        results1, results2, results3 = run_all_methods()
        
        # 통일된 점수 분석
        comparison_df = analyze_unified_scores(results1, results2, results3)
        
        print("\n🎉 분석 완료! 저장된 파일:")
        print("- method1_normalized_results.csv")
        print("- method2_results_top10.csv") 
        print("- method3_results_top10.csv")
        print("- unified_score_comparison.csv")
        print("- unified_score_comparison.png")
        
        return comparison_df
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    comparison_result = main()