"""
RAG 검색 방법들의 교집합 분석 메인 스크립트
세 가지 검색 방법의 결과를 비교하고 교집합을 분석합니다.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from unified_csv_utils import analyze_intersection, print_intersection_summary, create_unified_csv


def ensure_unified_csvs():
    """
    각 방법의 결과를 통일된 형태로 변환 (필요한 경우)
    """
    methods = [
        ('method1', 'result/RAG_result/method1_results.csv', 'result/RAG_result/method1_unified.csv'),
        ('method2', 'result/RAG_result/method2_results.csv', 'result/RAG_result/method2_unified.csv'),
        ('method3', 'result/RAG_result/method3_results.csv', 'result/RAG_result/method3_unified.csv')
    ]
    
    unified_files = []
    
    for method_name, original_file, unified_file in methods:
        if os.path.exists(original_file) and not os.path.exists(unified_file):
            try:
                print(f"🔄 {method_name} 결과를 통일된 형태로 변환 중...")
                df = pd.read_csv(original_file)
                
                # 각 방법별로 컬럼명이 다를 수 있으므로 매핑
                if method_name == 'method1':
                    # Method 1은 이미 통일된 형태여야 함
                    if 'query_id' in df.columns:
                        create_unified_csv(df, method_name, unified_file)
                    else:
                        print(f"⚠️ {method_name}: 통일된 형태가 아닌 것 같습니다. 수동 변환이 필요합니다.")
                
                elif method_name == 'method2':
                    # Method 2도 이미 통일된 형태여야 함
                    if 'query_id' in df.columns:
                        create_unified_csv(df, method_name, unified_file)
                    else:
                        print(f"⚠️ {method_name}: 통일된 형태가 아닌 것 같습니다. 수동 변환이 필요합니다.")
                
                elif method_name == 'method3':
                    # Method 3도 이미 통일된 형태여야 함
                    if 'query_id' in df.columns:
                        create_unified_csv(df, method_name, unified_file)
                    else:
                        print(f"⚠️ {method_name}: 통일된 형태가 아닌 것 같습니다. 수동 변환이 필요합니다.")
                
            except Exception as e:
                print(f"❌ {method_name} 변환 실패: {e}")
        
        if os.path.exists(unified_file):
            unified_files.append(unified_file)
        elif os.path.exists(original_file):
            # 통일된 파일이 없으면 원본 파일 사용
            unified_files.append(original_file)
    
    return unified_files


def main():
    """
    교집합 분석 메인 함수
    """
    print("🚀 RAG 검색 방법 교집합 분석을 시작합니다...")
    print()
    
    # 결과 디렉토리 확인
    result_dir = Path("result/RAG_result")
    if not result_dir.exists():
        print("❌ result/RAG_result 디렉토리가 존재하지 않습니다.")
        print("먼저 각 RAG 검색 방법을 실행하여 결과를 생성하세요.")
        return
    
    # 통일된 형태의 CSV 파일 준비
    unified_files = ensure_unified_csvs()
    
    if len(unified_files) < 2:
        print("❌ 비교할 수 있는 결과 파일이 충분하지 않습니다.")
        print(f"찾은 파일: {unified_files}")
        return
    
    print(f"📂 분석할 파일들: {len(unified_files)}개")
    for file in unified_files:
        print(f"  - {file}")
    print()
    
    # 교집합 분석 수행
    try:
        analysis_results = analyze_intersection(
            unified_files, 
            'result/RAG_result/intersection_analysis.json'
        )
        
        # 결과 요약 출력
        print_intersection_summary(analysis_results)
        
        # 상세 분석 결과를 텍스트 파일로도 저장
        save_detailed_analysis(analysis_results, 'result/RAG_result/intersection_detailed.txt')
        
        print("\n✅ 교집합 분석이 완료되었습니다!")
        print("📁 결과 파일:")
        print("  - result/RAG_result/intersection_analysis.json (전체 분석 결과)")
        print("  - result/RAG_result/intersection_detailed.txt (상세 분석 보고서)")
        
    except Exception as e:
        print(f"❌ 교집합 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def save_detailed_analysis(analysis_results, output_path):
    """
    상세 분석 결과를 텍스트 파일로 저장
    
    Args:
        analysis_results (dict): analyze_intersection의 결과
        output_path (str): 출력 파일 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAG 검색 방법별 교집합 분석 상세 보고서\n")
        f.write("=" * 80 + "\n\n")
        
        # 기본 정보
        f.write(f"분석 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 쿼리 수: {analysis_results['total_queries']}\n")
        f.write(f"분석 방법 수: {len(analysis_results['method_coverage'])}\n\n")
        
        # 방법별 성능
        f.write("1. 방법별 검색 성능\n")
        f.write("-" * 40 + "\n")
        for method, stats in analysis_results['method_coverage'].items():
            f.write(f"\n[{method}]\n")
            f.write(f"  고유 문서 수: {stats['unique_documents']}\n")
            f.write(f"  총 검색 결과: {stats['total_retrievals']}\n")
            f.write(f"  결과가 있는 쿼리: {stats['queries_with_results']}\n")
            coverage_rate = stats['queries_with_results'] / analysis_results['total_queries'] * 100
            f.write(f"  커버리지 비율: {coverage_rate:.1f}%\n")
        
        # 쌍별 유사성
        f.write(f"\n\n2. 방법 간 유사성 분석\n")
        f.write("-" * 40 + "\n")
        for pair_name, stats in analysis_results['pairwise_intersections'].items():
            f.write(f"\n[{pair_name}]\n")
            f.write(f"  평균 Jaccard 유사도: {stats['average_jaccard']:.3f}\n")
            f.write(f"  평균 교집합 크기: {stats['average_intersection_size']:.1f}\n")
            f.write(f"  교집합이 있는 쿼리: {stats['queries_with_intersection']}\n")
            intersection_rate = stats['queries_with_intersection'] / analysis_results['total_queries'] * 100
            f.write(f"  교집합 비율: {intersection_rate:.1f}%\n")
        
        # 전체 교집합 통계
        if analysis_results['complete_intersection']:
            f.write(f"\n\n3. 전체 교집합 통계\n")
            f.write("-" * 40 + "\n")
            
            intersection_sizes = [q['intersection_size'] for q in analysis_results['complete_intersection']]
            union_sizes = [q['union_size'] for q in analysis_results['union_coverage']]
            
            queries_with_intersection = len([s for s in intersection_sizes if s > 0])
            f.write(f"  교집합이 있는 쿼리: {queries_with_intersection}\n")
            f.write(f"  교집합 비율: {queries_with_intersection / analysis_results['total_queries'] * 100:.1f}%\n")
            f.write(f"  평균 교집합 크기: {sum(intersection_sizes) / len(intersection_sizes):.1f}\n")
            f.write(f"  평균 합집합 크기: {sum(union_sizes) / len(union_sizes):.1f}\n")
            f.write(f"  최대 교집합 크기: {max(intersection_sizes)}\n")
            f.write(f"  최대 합집합 크기: {max(union_sizes)}\n")
            
            # 교집합 크기별 분포
            from collections import Counter
            intersection_dist = Counter(intersection_sizes)
            f.write(f"\n  교집합 크기별 분포:\n")
            for size in sorted(intersection_dist.keys()):
                count = intersection_dist[size]
                f.write(f"    크기 {size}: {count}개 쿼리\n")
        
        # 쿼리별 상세 분석 (처음 10개만)
        f.write(f"\n\n4. 쿼리별 상세 분석 (상위 10개 쿼리)\n")
        f.write("-" * 40 + "\n")
        
        if analysis_results['complete_intersection']:
            for i, query_analysis in enumerate(analysis_results['complete_intersection'][:10]):
                f.write(f"\n쿼리 {query_analysis['query_id']}:\n")
                f.write(f"  교집합 크기: {query_analysis['intersection_size']}\n")
                f.write(f"  합집합 크기: {query_analysis['union_size']}\n")
                if query_analysis['intersection_docs']:
                    f.write(f"  공통 문서 ID: {query_analysis['intersection_docs'][:5]}{'...' if len(query_analysis['intersection_docs']) > 5 else ''}\n")
        
        f.write(f"\n\n" + "=" * 80 + "\n")
        f.write("분석 완료\n")
        f.write("=" * 80 + "\n")


if __name__ == "__main__":
    main()
