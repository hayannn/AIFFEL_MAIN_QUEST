# 시계열 분류 프로젝트
- 프로젝트 목표
  - Objective 1: 비정상 데이터를 정상 데이터로 만들기
  - Objective 2: 분류 모델의 성능 높이기

## 순서
- 비정상 데이터를 정상 데이터로 만들기
- 분산을 일정하게(로그 변환, log transformation)
- 차분으로 추세 제거
- 계절 차분으로 계절성 제거
- 검정: 정상성 확인
- 시계열 분류 진행
  - 데이터 전처리(다운로드, 가공, 나누기)
  - Feature extraction
  - impute
  - 모델 적용: RandomForest,XGBoost -> score로 평가
  - 시각화: XGBoost plot_importance
  - Classification report(검증 및 분석)
