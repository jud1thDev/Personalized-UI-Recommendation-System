# 코랩ver

## 1. 코랩 노트북 생성
- [Google Colab](https://colab.research.google.com/) 접속
- 새 노트북 생성

## 2. 필요한 패키지 설치
```python
!pip install -q pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 lightgbm>=4.0.0 pyyaml>=6.0 tqdm>=4.65.0 joblib>=1.3.0
```

## 3. 파일 업로드
###  GitHub에서 클론
```python
!git clone https://github.com/jud1thDev/Personalized-UI-Recommendation-System.git
%cd Personalized-UI-Recommendation-System
```

## 4. 실행
```python
# 파일 실행
exec(open('colab_main.py').read())

# 또는 직접 실행
!python colab_main.py
```
