# Finance Time Series
- 이론 : [velog.io/@dlgkdis801/Node 10. Finance Time Series 데이터 활용하기](https://velog.io/@dlgkdis801/Node-10.-Finance-Time-Series-%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%99%9C%EC%9A%A9%ED%95%98%EA%B8%B0)
- 프로젝트
  - [1] Data Labeling
  - [2] Feature Engineering
  - [3] Model Training

## 순서
- Data Labeling
  - 필요 라이브러리 import
  - 데이터 불러옹기 & 시각화
  - Price Change Direction
  - Using Moving Average
  - Local Min-Max
  - Trend Scanning
- Feature Engineering
  - 환경 구성 및 데이터 불러오기
  - Technical Index
  - Feature Selection methods
    - MDI
    - MDA
    - RFE CV
    - SFS
    - SHAP
- Model Training
  - 환경 구성 및 데이터 불러오기
  - Purged K-fold for Cross-Validation
  - n_cv = 5로 실험!(LMS 환경은 기존 4로 진행하여 비교)
  - Model 적용
  - LMS 환경(기존 코드)와 결과 비교
  - 결과 분석
  - 추가 시도
- 추가: 튜닝 방식 바꿔보기_Optuna
  - Optuna 설치 & 적용
  - Optuna 하이퍼파라미터 튜닝 결과 분석
- 모델 변경
  - Logistic Regression
  - LightGBM
  - CatBoost
  - Naive Bayes
  - 새로운 모델들의 결과 분석
- 회고