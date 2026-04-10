# IMBK_Bank_Customer_Churn_ML

# 🏦 고객 이탈 분류 ML 및 인사이트 분석

## 프로젝트 기간
2026.04.10

## 📌 프로젝트 개요
본 프로젝트는 은행 고객 데이터를 분석하여 이탈(Churn) 위험이 있는 고객을 사전에 예측하는 머신러닝 모델링 프로젝트입니다. 단순히 모델 성능을 높이는 것을 넘어, 아래 세 가지 핵심 목표를 달성하는 데 주안점을 두었습니다.

1. **데이터에 대한 깊은 이해 (Understanding)**
2. **풀 프로세스 코딩 구현 (Full-process Coding)**
3. **비즈니스 관점의 설명력 확보 (Explainability)**

- **데이터 출처**: [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data) (row: 10000, col:12)
- **개발자**: 박성준

<br/>

## 🛠 기술 스택 (Tech Stack)
- **Language**: `Python`
- **Environment**: `Jupyter Notebook`
- **Libraries**: `Pandas`, `NumPy`, `Scikit-learn`, `CatBoost`, `XGBoost`, `LightGBM` , 'seaborn' 

<br/>

## ⚙️ 데이터 전처리 (Data Preprocessing)
모델의 학습 효율과 성능을 극대화하기 위해 다음과 같은 전처리 파이프라인을 구축했습니다.

1. **불필요한 피처 제거**: 모델의 예측과 무관한 식별자 컬럼(`RowNumber`, `CustomerId`, `Surname`) 제거
2. **범주형 데이터 인코딩 (Encoding)**: 
   - 성별(`Gender`), 국가(`Geography`) 등의 명목형 변수는 모델이 인식할 수 있도록 **One-Hot Encoding** 처리
3. **수치형 데이터 스케일링 (Scaling)**:
   - 나이(`Age`), 잔고(`Balance`), 추정 급여(`EstimatedSalary`) 등 값의 편차가 큰 연속형 변수는 **StandardScaler**를 적용하여 분포를 조정
4. **클래스 불균형(Class Imbalance) 확인**: 이탈하지 않은 고객(0)이 이탈한 고객(1)보다 많은 불균형 데이터임을 확인하고, 이를 고려하여 교차 검증 및 평가 지표(F1 Score) 설정

<br/>

## 📊 탐색적 데이터 분석 (EDA) 및 해석
데이터 시각화 및 통계적 분석을 통해 고객 이탈에 영향을 미치는 주요 인사이트를 도출했습니다. 

- **나이(Age)와 이탈률**: 40~50대 중년층 고객 그룹에서 이탈률이 눈에 띄게 상승하는 경향을 확인했습니다.
- **활성 상태(IsActiveMember)**: 비활성 고객의 이탈 확률이 활성 고객 대비 유의미하게 높게 나타났습니다. 지속적인 서비스 참여 유도가 중요함을 시사합니다.
- **보유 상품 수(NumOfProducts)**: 금융 상품을 3개 이상 보유한 고객의 경우 오히려 이탈률이 급증하는 이상 현상(Anomaly)을 발견했습니다. 이는 특정 결합 상품의 만족도 저하 또는 복잡한 서비스 구조 때문일 것으로 해석됩니다.
- **성별 및 지역(Gender & Geography)**: 여성 고객의 이탈률이 남성보다 다소 높았으며, 특정 국가(예: 독일) 고객의 이탈 비중이 두드러졌습니다.

<br/>

## 💡 모델링 전략 (Modeling Strategy)

본 프로젝트에서는 단일 모델의 한계를 극복하고 예측의 안정성을 높이기 위해 **스태킹 앙상블(Stacking Ensemble)** 기법을 활용했습니다.

1. **Base Models**: 여러 머신러닝 알고리즘을 활용하여 개별 예측 수행 (검증 결과 단일 모델 중에서는 `CatBoost`가 F1 Score 0.6088로 가장 우수한 성능을 보임)
2. **Meta Model**: `Logistic Regression`을 최종 추정기(Final Estimator)로 사용하여 Base Model들의 예측 결과를 종합 및 최종 예측 도출

<br/>

## 📊 모델 성능 평가 (Results)

스태킹 앙상블을 통해 모든 개별 모델의 장점을 결합한 결과, 목표로 했던 베이스라인 수치(Accuracy 80%, F1 Score 57%)를 크게 상회하는 우수한 성능을 달성했습니다.

| Metric | Target Score | **Stacking Ensemble Score** |
| :--- | :---: | :---: |
| **Accuracy** (정확도) | 80.00% | **86.85%** |
| **F1 Score** | 57.00% | **60.33%** |

<br/>

<img width="757" height="550" alt="사후분석" src="https://github.com/user-attachments/assets/0eafc9d7-4b57-4f80-8ac7-5964856a5922" />

## 🚀 모델 고도화 및 사후 분석 (Advanced Modeling & Explainability)

### 🛠 1. AutoML & Hyperparameter Tuning
개별 모델의 성능을 한계까지 끌어올리기 위해 하이퍼파라미터 튜닝을 진행했습니다. (💡 팁: Optuna, GridSearchCV 등 사용하신 라이브러리 이름을 여기에 적어주세요)를 활용하여 데이터의 복잡한 패턴을 과적합 없이 학습할 수 있는 최적의 파라미터 공간을 탐색했습니다. 

### 🧱 2. Stacking Pipeline 구축
튜닝이 완료된 최적의 단일 모델(Base Models)들을 결합하여 자동화된 **Stacking Ensemble Pipeline**을 구축했습니다. 각 모델의 예측값을 새로운 피처로 삼아 Meta Model이 최종 판단을 내리도록 설계함으로써, 개별 알고리즘이 가진 편향(Bias)을 상호 보완하고 예측의 안정성과 일반화 성능을 극대화했습니다.

### 🔍 3. SHAP Value 기반 사후 분석 (Model Explainability)
앙상블 모델 특유의 '블랙박스(Black-box)' 한계를 극복하고, 프로젝트의 핵심 목표인 **'비즈니스 설명력(Explainability)'**을 확보하기 위해 **SHAP (SHapley Additive exPlanations)** 분석을 수행했습니다.

1. **나이 (Age)**: 이탈을 예측하는 가장 강력하고 지배적인 변수입니다. 붉은색 점(고령)이 기준선 우측(+)에 넓게 퍼져있는 것으로 보아, **나이가 많을수록 이탈 확률이 가파르게 상승**함을 알 수 있습니다.
2. **보유 상품 수 (NumOfProducts)**: 보유 상품 수가 많을수록(붉은색) SHAP Value가 양(+)의 방향으로 크게 작용하여 **이탈 위험을 가중**시킵니다.
3. **활성 상태 (IsActiveMember)**: 비활성 상태(푸른색, 0)인 고객은 이탈 확률이 높고, 반대로 활성 상태(붉은색, 1)인 고객은 이탈 확률이 강하게 억제(음의 SHAP Value)됩니다. **지속적인 앱/웹 접속 유도가 이탈 방어에 핵심**임을 시사합니다.
4. **특정 국가 (Geography_Germany)**: 독일 고객(붉은색, 1)의 경우 타 국가 대비 이탈 경향성이 두드러지게 나타났습니다.

<br/>

## 💡인사이트 제안

* 모델 분석 결과 다상품을 보유한 우수 고객과 고연령층의 이탈 위험이 예상 외로 높게 나타났으며 앱 비활성 상태일수록 이탈 확률이 급증했습니다.
* 이를 방어하기 위해 다상품 고객을 위한 통합 관리 및 혜택 리뉴얼이 시급하며 시니어 전용 간편 UI 도입 등 맞춤형 서비스 개선이 필요합니다.
* 더불어 휴면 고객을 꺠우는 적극적인 타겟팅 캠페인과 고액 자산가의 자산 이탈을 막기 위한 선제적인 특판 투자 상품 제안 등 핵심 고객층을 향한 세밀한 밀착 관리가 실행 되어야 합니다.

<br/>

## 🎯 결론 및 기대 효과 (Conclusion & Impact)
최종 모델은 은행의 이탈 위험 고객을 매우 안정적이고 정확하게 예측할 수 있음을 증명했습니다. 이를 통해 얻을 수 있는 비즈니스적 기대 효과는 다음과 같습니다.

- **선제적 고객 관리**: 이탈 확률이 높은 고객을 조기에 식별하여 타겟팅된 프로모션 및 CRM 전략 수립 가능.
- **수익성 개선**: 기존 고객 유지 비용이 신규 고객 유치 비용보다 낮다는 점을 고려할 때, 효과적인 이탈 방어를 통한 직접적인 영업 이익 향상.

---
