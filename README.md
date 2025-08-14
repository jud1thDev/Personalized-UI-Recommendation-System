# Personalized UI Recommendation System

## 개요
사용자 행동 로그를 기반으로 홈 화면을 개인화하는 AI 시스템

## 주요 기능
1. **기능 노출 여부 예측** - 사용자가 실제로 사용할 기능만 표시
2. **UI 유형 예측** - 사용자 타입(일반/시니어)에 따른 최적의 컴포넌트 타입 선택
   - **일반 사용자**: list_single, list_multi, grid_multi, grid_full, banner
   - **시니어 사용자**: s_list_single, s_list_multi, s_grid_multi, s_grid_full
   - **공통**: exchange_stock_widget (환율·증시)
3. **기능 그룹 및 레이블 예측** - 서비스 클러스터와 소제목 자동 생성
4. **홈 화면 최적화** - 정보 밀도, 배치 순서, 레이아웃 자동 조정

## 시스템 구성
- **데이터**: CSV 기반 사용자 행동 로그 (세션, 클릭, 체류시간, 경로 등)
- **피처 엔지니어링**: 시간대, 세션 빈도, 경로 엔트로피, 기능별 사용 패턴
- **모델**: LightGBM 4.x (Binary/Multi-class/Regression)
- **출력**: 프론트엔드에서 바로 사용 가능한 JSON 구조

## 프로젝트 구조
```
Personalized-UI-Recommendation-System/
├── README.md
├── requirements.txt
├── .gitignore                
├── colab_main.ipynb          # 코랩용 Jupyter 노트북
├── .git/
│   └── hooks/
│       ├── pre-commit        # Linux/Mac용 노트북 출력 제거 훅
│       └── pre-commit.ps1   # Windows용 노트북 출력 제거 훅
└── ui_rec/
    ├── config/               # 설정 파일 (실행 시 자동 생성)
    │   ├── data.yaml
    │   └── model.yaml
    ├── data/
    │   ├── raw/              # 원본 데이터 (CSV 파일들)
    │   ├── processed/        # 전처리된 피처 데이터
    │   ├── models/           # 학습된 모델 파일들 (.joblib)
    │   ├── outputs/          # 예측 결과 (JSON)
    │   └── logs/             # 실행 로그
    └── src/
        ├── data/             # 데이터 생성 및 전처리
        │   └── generate_mock.py
        ├── features/          # 피처 엔지니어링
        │   └── build_features.py
        ├── models/            # 모델 학습 및 예측
        │   ├── common.py
        │   ├── exposure.py
        │   ├── rank.py
        │   ├── service_cluster.py
        │   ├── ui_grouping.py
        │   └── ui_type.py
        └── utils/             # 유틸리티 함수들
            ├── ai_label_generator.py
            ├── component_mapping.py
            ├── features.py
            ├── icons.py
            ├── io.py
            ├── logging.py
            ├── metrics.py
            ├── schema.py
            └── time.py
```

## Google Colab에서 실행

### **1. Google Colab 접속**
1. [Google Colab](https://colab.research.google.com/) 접속
2. Google 계정으로 로그인

### **2. 프로젝트 파일 업로드**
1. Colab에서 `파일` → `GitHub에서 노트북 가져오기` 클릭
2. 저장소 URL 입력: `jud1thDev/Personalized-UI-Recommendation-System`
3. `colab_main.ipynb` 선택

### **3. 실행 환경 설정**
1. **런타임 유형 변경**: `런타임` → `런타임 유형 변경`
2. 하드웨어 가속기: `GPU` 또는 `TPU` 선택 (권장)
3. 메모리: `고용량 RAM` 선택

### **4. 노트북 실행**

### **5. 실행 완료 확인**
모든 셀이 성공적으로 실행되면 `ui_home_outputs.json` 파일이 다운로드됨

