# Personalized UI Recommendation System

## 개요
다양한 연령대(시니어 포함)의 사용자 행동 로그를 기반으로 홈 화면을 개인화하는 AI 시스템

## 주요 기능
1. **기능 노출 여부 예측** - 사용자가 실제로 사용할 기능만 표시
2. **UI 유형 예측** - card, list_item, banner, icon 등 최적의 컴포넌트 타입 선택
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
└── ui_rec/
    ├── config/               # 설정 파일 (실행 시 자동 생성)
    └── src/
        ├── data/             # 데이터 생성
        ├── features/          # 피처 엔지니어링
        ├── models/            # 모델 학습
        └── utils/             # 유틸리티
```

## Google Colab에서 실행

### 1. 프로젝트 업로드
1. **GitHub에서 다운로드**
   ```bash
   !git clone https://github.com/jud1thDev/Personalized-UI-Recommendation-System.git
   ```

2. **Google Colab에서 파일 업로드**
   - `colab_main.ipynb` 파일을 Google Colab에 업로드
   - 또는 GitHub 저장소를 직접 연결

### 2. 실행 
노트북의 각 셀을 **순서대로** 실행

### 3. 결과 확인
- **자동 다운로드**: `ui_home_outputs.json` 파일이 자동으로 다운로드됨
- **파일 위치**: `ui_rec/data/outputs/ui_home_outputs.json`
- **내용**: 각 사용자별 개인화된 홈 화면 구성