"""
통일된 CSV 형태로 RAG 검색 결과를 저장하기 위한 유틸리티 함수들
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def create_unified_csv(results_df: pd.DataFrame, method_name: str, output_path: str) -> None:
    """
    RAG 검색 결과를 통일된 형태의 CSV로 저장
    
    Args:
        results_df (pd.DataFrame): 검색 결과 DataFrame
        method_name (str): 검색 방법명 (method1, method2, method3)
        output_path (str): 출력 CSV 파일 경로
    """
    
    # 통일된 컬럼명 확인
    required_columns = ['query_id', 'input_text', 'retrieved_indices', 'retrieved_scores', 'method']
    
    # 필수 컬럼이 있는지 확인
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 통일된 형태로 재정렬
    unified_df = results_df[required_columns].copy()
    
    # 추가 통계 정보 컬럼 생성
    unified_df['num_results'] = unified_df['retrieved_indices'].apply(len)
    unified_df['max_score'] = unified_df['retrieved_scores'].apply(
        lambda x: max(x) if x else 0.0
    )
    unified_df['min_score'] = unified_df['retrieved_scores'].apply(
        lambda x: min(x) if x else 0.0
    )
    unified_df['avg_score'] = unified_df['retrieved_scores'].apply(
        lambda x: np.mean(x) if x else 0.0
    )
    
    # CSV로 저장
    unified_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 통일된 형태의 결과가 저장되었습니다: {output_path}")


def analyze_intersection(csv_paths: List[str], output_path: str = None) -> Dict[str, Any]:
    """
    여러 RAG 검색 방법의 결과에서 교집합 분석
    
    Args:
        csv_paths (List[str]): 각 방법의 CSV 파일 경로 리스트
        output_path (str, optional): 교집합 분석 결과 저장 경로
        
    Returns:
        Dict[str, Any]: 교집합 분석 결과
    """
    
    # 각 방법의 결과 로드
    method_results = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        method_name = df['method'].iloc[0] if 'method' in df.columns else csv_path
        method_results[method_name] = df
    
    analysis_results = {
        'total_queries': 0,
        'method_coverage': {},
        'pairwise_intersections': {},
        'complete_intersection': [],
        'union_coverage': []
    }
    
    if not method_results:
        return analysis_results
    
    # 총 쿼리 수 (모든 방법이 동일해야 함)
    query_counts = [len(df) for df in method_results.values()]
    if len(set(query_counts)) > 1:
        print("⚠️ 경고: 방법별 쿼리 수가 다릅니다.")
    
    analysis_results['total_queries'] = max(query_counts) if query_counts else 0
    
    # 각 방법별 커버리지 분석
    for method_name, df in method_results.items():
        retrieved_docs = []
        for _, row in df.iterrows():
            if row['retrieved_indices']:  # 빈 리스트가 아닌 경우
                retrieved_docs.extend(eval(row['retrieved_indices']) if isinstance(row['retrieved_indices'], str) else row['retrieved_indices'])
        
        unique_docs = set(retrieved_docs)
        analysis_results['method_coverage'][method_name] = {
            'unique_documents': len(unique_docs),
            'total_retrievals': len(retrieved_docs),
            'queries_with_results': len([1 for _, row in df.iterrows() if row['retrieved_indices'] and len(eval(row['retrieved_indices']) if isinstance(row['retrieved_indices'], str) else row['retrieved_indices']) > 0])
        }
    
    # 쌍별 교집합 분석
    method_names = list(method_results.keys())
    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            method1, method2 = method_names[i], method_names[j]
            
            # 각 쿼리별로 교집합 계산
            query_intersections = []
            for query_id in range(min(len(method_results[method1]), len(method_results[method2]))):
                docs1 = method_results[method1].iloc[query_id]['retrieved_indices']
                docs2 = method_results[method2].iloc[query_id]['retrieved_indices']
                
                # 문자열인 경우 파싱
                if isinstance(docs1, str):
                    docs1 = eval(docs1) if docs1.strip() else []
                if isinstance(docs2, str):
                    docs2 = eval(docs2) if docs2.strip() else []
                
                intersection = set(docs1) & set(docs2)
                union = set(docs1) | set(docs2)
                
                query_intersections.append({
                    'query_id': query_id,
                    'intersection_size': len(intersection),
                    'union_size': len(union),
                    'jaccard_similarity': len(intersection) / len(union) if union else 0.0,
                    'method1_size': len(docs1),
                    'method2_size': len(docs2)
                })
            
            analysis_results['pairwise_intersections'][f"{method1}_vs_{method2}"] = {
                'query_level_analysis': query_intersections,
                'average_jaccard': np.mean([q['jaccard_similarity'] for q in query_intersections]),
                'average_intersection_size': np.mean([q['intersection_size'] for q in query_intersections]),
                'queries_with_intersection': len([q for q in query_intersections if q['intersection_size'] > 0])
            }
    
    # 전체 교집합 (모든 방법에서 공통으로 검색된 문서)
    if len(method_names) >= 2:
        for query_id in range(analysis_results['total_queries']):
            query_docs = []
            for method_name in method_names:
                if query_id < len(method_results[method_name]):
                    docs = method_results[method_name].iloc[query_id]['retrieved_indices']
                    if isinstance(docs, str):
                        docs = eval(docs) if docs.strip() else []
                    query_docs.append(set(docs))
            
            if query_docs:
                complete_intersection = set.intersection(*query_docs) if query_docs else set()
                union_docs = set.union(*query_docs) if query_docs else set()
                
                analysis_results['complete_intersection'].append({
                    'query_id': query_id,
                    'intersection_size': len(complete_intersection),
                    'union_size': len(union_docs),
                    'intersection_docs': list(complete_intersection)
                })
                
                analysis_results['union_coverage'].append({
                    'query_id': query_id,
                    'union_size': len(union_docs),
                    'union_docs': list(union_docs)
                })
    
    # 결과 저장
    if output_path:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            # JSON 직렬화를 위해 set을 list로 변환
            serializable_results = analysis_results.copy()
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"✅ 교집합 분석 결과가 저장되었습니다: {output_path}")
    
    return analysis_results


def print_intersection_summary(analysis_results: Dict[str, Any]) -> None:
    """
    교집합 분석 결과 요약 출력
    
    Args:
        analysis_results (Dict[str, Any]): analyze_intersection의 결과
    """
    print("=" * 50)
    print("📊 RAG 검색 방법별 교집합 분석 결과")
    print("=" * 50)
    
    print(f"총 쿼리 수: {analysis_results['total_queries']}")
    print()
    
    # 방법별 커버리지
    print("🔍 방법별 검색 성능:")
    for method, stats in analysis_results['method_coverage'].items():
        print(f"  {method}:")
        print(f"    - 고유 문서 수: {stats['unique_documents']}")
        print(f"    - 총 검색 결과: {stats['total_retrievals']}")
        print(f"    - 결과가 있는 쿼리: {stats['queries_with_results']}")
    print()
    
    # 쌍별 교집합
    print("🤝 방법 간 유사성:")
    for pair_name, stats in analysis_results['pairwise_intersections'].items():
        print(f"  {pair_name}:")
        print(f"    - 평균 Jaccard 유사도: {stats['average_jaccard']:.3f}")
        print(f"    - 평균 교집합 크기: {stats['average_intersection_size']:.1f}")
        print(f"    - 교집합이 있는 쿼리: {stats['queries_with_intersection']}")
    print()
    
    # 전체 교집합 통계
    if analysis_results['complete_intersection']:
        intersection_sizes = [q['intersection_size'] for q in analysis_results['complete_intersection']]
        union_sizes = [q['union_size'] for q in analysis_results['union_coverage']]
        
        print("🎯 전체 교집합 통계:")
        print(f"  - 교집합이 있는 쿼리: {len([s for s in intersection_sizes if s > 0])}")
        print(f"  - 평균 교집합 크기: {np.mean(intersection_sizes):.1f}")
        print(f"  - 평균 합집합 크기: {np.mean(union_sizes):.1f}")
        print(f"  - 최대 교집합 크기: {max(intersection_sizes)}")
    
    print("=" * 50)


# 사용 예시
if __name__ == "__main__":
    # 교집합 분석 예시
    csv_files = [
        'result/method1_results.csv',
        'result/method2_results.csv', 
        'result/method3_results.csv'
    ]
    
    try:
        results = analyze_intersection(csv_files, 'result/intersection_analysis.json')
        print_intersection_summary(results)
    except Exception as e:
        print(f"❌ 교집합 분석 중 오류: {e}")
