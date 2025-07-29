"""
모든 RAG 검색 방법을 실행하고 교집합 분석을 수행하는 통합 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path
import time

def run_method(method_name, script_path):
    """
    개별 RAG 검색 방법 실행
    
    Args:
        method_name (str): 방법명
        script_path (str): 스크립트 경로
    """
    print(f"🚀 {method_name} 실행 중...")
    print(f"   스크립트: {script_path}")
    
    if not os.path.exists(script_path):
        print(f"❌ {method_name}: 스크립트 파일을 찾을 수 없습니다: {script_path}")
        return False
    
    try:
        start_time = time.time()
        
        # Python 스크립트 실행
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, encoding='utf-8')
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {method_name} 완료 (소요시간: {elapsed_time:.1f}초)")
            # 표준 출력의 마지막 몇 줄만 표시 (너무 길 수 있으므로)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                print("   출력:")
                for line in lines[-5:]:  # 마지막 5줄만
                    print(f"     {line}")
            return True
        else:
            print(f"❌ {method_name} 실행 실패 (코드: {result.returncode})")
            if result.stderr:
                print(f"   오류: {result.stderr[:500]}")  # 오류 메시지 일부만
            return False
            
    except Exception as e:
        print(f"❌ {method_name} 실행 중 예외 발생: {e}")
        return False


def check_prerequisites():
    """
    실행 전 전제조건 확인
    """
    print("🔍 전제조건 확인 중...")
    
    # 필요한 디렉토리 확인
    required_dirs = [
        'result',
        'result/RAG_result',
        'docs'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"📁 디렉토리 생성: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # 필요한 데이터 파일 확인
    required_files = [
        'result/predictions.json',
        'result/predictions_with_keywords.json',
        'docs/한글 맞춤법 , 표준어  규정.json',
        'docs/띄어쓰기.json',
        'docs/문장 부호.json',
        'docs/외래어  표기법.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️ 다음 파일들이 누락되었습니다:")
        for file in missing_files:
            print(f"   - {file}")
        print("필요한 데이터 파일을 준비한 후 다시 실행하세요.")
        return False
    
    print("✅ 전제조건 확인 완료")
    return True


def main():
    """
    통합 실행 메인 함수
    """
    print("🎯 RAG 검색 방법 통합 실행 및 교집합 분석")
    print("=" * 60)
    
    # 전제조건 확인
    if not check_prerequisites():
        return
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    rag_search_dir = current_dir / "inference" / "RAG_search"
    
    if not rag_search_dir.exists():
        rag_search_dir = current_dir / "RAG_search"
    
    if not rag_search_dir.exists():
        rag_search_dir = current_dir
    
    print(f"📂 작업 디렉토리: {rag_search_dir}")
    
    # 원래 디렉토리로 작업 디렉토리 변경
    original_dir = os.getcwd()
    os.chdir(rag_search_dir)
    
    try:
        # 각 RAG 검색 방법 실행
        methods = [
            ("Method 1 (키워드 룰베이스)", "method1_keyword_rule_based_search.py"),
            ("Method 2 (의미 임베딩)", "method2_semantic_embedding_search.py"),
            ("Method 3 (하이브리드)", "method3_hybrid_tfidf_embedding_search.py")
        ]
        
        successful_methods = []
        
        for method_name, script_name in methods:
            print(f"\n{'-' * 40}")
            success = run_method(method_name, script_name)
            if success:
                successful_methods.append(method_name)
            print(f"{'-' * 40}")
            
            # 각 방법 간에 잠시 대기 (시스템 리소스 고려)
            if method_name != methods[-1][0]:  # 마지막이 아니면
                print("⏳ 잠시 대기 중... (3초)")
                time.sleep(3)
        
        # 결과 요약
        print(f"\n📊 실행 결과 요약:")
        print(f"   성공한 방법: {len(successful_methods)}/3")
        for method in successful_methods:
            print(f"   ✅ {method}")
        
        failed_methods = len(methods) - len(successful_methods)
        if failed_methods > 0:
            print(f"   ❌ 실패한 방법: {failed_methods}개")
        
        # 교집합 분석 실행 (2개 이상 성공한 경우)
        if len(successful_methods) >= 2:
            print(f"\n{'=' * 60}")
            print("📈 교집합 분석 실행 중...")
            
            try:
                from run_intersection_analysis import main as run_intersection
                run_intersection()
            except Exception as e:
                print(f"❌ 교집합 분석 실행 실패: {e}")
                # 직접 스크립트 실행 시도
                try:
                    result = subprocess.run([
                        sys.executable, "run_intersection_analysis.py"
                    ], capture_output=True, text=True, encoding='utf-8')
                    
                    if result.returncode == 0:
                        print("✅ 교집합 분석 완료")
                        print(result.stdout)
                    else:
                        print(f"❌ 교집합 분석 스크립트 실행 실패: {result.stderr}")
                except Exception as e2:
                    print(f"❌ 교집합 분석 스크립트 실행 중 예외: {e2}")
        else:
            print(f"\n⚠️ 교집합 분석을 위해서는 최소 2개 방법이 성공해야 합니다.")
        
        print(f"\n🎉 모든 작업이 완료되었습니다!")
        print(f"📁 결과 확인: result/RAG_result/ 디렉토리")
        
    finally:
        # 원래 디렉토리로 복원
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
