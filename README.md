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
├── colab_main.py          # 코랩용 통합 실행 파일
├── COLAB_GUIDE.md         # 코랩 실행 가이드
├── ui_rec/
│   ├── config/            # 설정 파일 (실행 시 자동 생성)
│   ├── scripts/           # 실행 스크립트 (OS 따라서 run_all.ps1, run_all.sh)
│   └── src/              
│       ├── data/          # 데이터 생성
│       ├── features/      # 피처 엔지니어링
│       ├── models/        # 모델 학습
│       ├── inference/     # 추론
│       └── utils/         # 유틸리티
```

## 빠른 시작

### 1. 환경 설정
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 실행
```powershell
# Windows PowerShell
.\.ui_rec\scripts\run_all.ps1

# Linux/Mac
bash ui_rec/scripts/run_all.sh
```

```bash
python -m ui_rec.src.data.generate_mock      # 모의 데이터 생성
python -m ui_rec.src.features.build_features # 피처 생성
python -m ui_rec.src.models.exposure         # 노출 모델 학습
python -m ui_rec.src.models.ui_type          # UI 타입 모델 학습
python -m ui_rec.src.models.service_cluster  # 서비스 클러스터 예측 모델 학습
python -m ui_rec.src.models.rank             # 순위 모델 학습
python -m ui_rec.src.inference.predict       # 추론 및 JSON 출력 (ui_grouping.py 로직 포함)
```

## 코랩에서 실행
COLAB_GUIDE.md 참고

## 출력
- **결과**: `ui_rec/data/outputs/ui_home_outputs.json`
- **모델**: `ui_rec/data/models/`
- **데이터**: `ui_rec/data/raw/`, `ui_rec/data/processed/`