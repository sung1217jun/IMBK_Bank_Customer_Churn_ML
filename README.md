# IMBK_Bank_Customer_Churn_ML

# 🏦 은행 고객 이탈 예측 (Bank Customer Churn Prediction)

## 📌 프로젝트 개요
본 프로젝트는 은행 고객 데이터를 분석하여 이탈(Churn) 위험이 있는 고객을 사전에 예측하는 머신러닝 모델링 프로젝트입니다. 단순히 모델 성능을 높이는 것을 넘어, 아래 세 가지 핵심 목표를 달성하는 데 주안점을 두었습니다.

1. **데이터에 대한 깊은 이해 (Understanding)**
2. **풀 프로세스 코딩 구현 (Full-process Coding)**
3. **비즈니스 관점의 설명력 확보 (Explainability)**

- **데이터 출처**: [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)
- **개발자**: 박성준

<br/>

## 🛠 기술 스택 (Tech Stack)
- **Language**: `Python`
- **Environment**: `Jupyter Notebook`
- **Libraries**: `Pandas`, `NumPy`, `Scikit-learn`, `CatBoost`, `XGBoost`, `LightGBM` (사용하신 라이브러리에 맞게 수정해주세요)

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

## 🎯 결론 및 기대 효과 (Conclusion & Impact)
최종 모델은 은행의 이탈 위험 고객을 매우 안정적이고 정확하게 예측할 수 있음을 증명했습니다. 이를 통해 얻을 수 있는 비즈니스적 기대 효과는 다음과 같습니다.

- **선제적 고객 관리**: 이탈 확률이 높은 고객을 조기에 식별하여 타겟팅된 프로모션 및 CRM 전략 수립 가능.
- **수익성 개선**: 기존 고객 유지 비용이 신규 고객 유치 비용보다 낮다는 점을 고려할 때, 효과적인 이탈 방어를 통한 직접적인 영업 이익 향상.

---
