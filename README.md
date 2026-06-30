# 은행 고객 이탈 예측 (Bank Customer Churn Prediction)

은행 고객 데이터를 활용해 **고객 이탈(Churn) 가능성을 예측**하고, 이탈 위험 고객을 조기에 식별하기 위한 EDA·머신러닝 모델링 프로젝트입니다. 최종 결과는 Streamlit 대시보드와 Tableau를 통해 고객별 이탈 확률과 핵심 인사이트로 제공됩니다.

> 머신러닝 팀 프로젝트 · 1조 · 2024.01

---

## 📌 프로젝트 개요


| 항목        | 내용                                                                    |
| --------- | --------------------------------------------------------------------- |
| **목표**    | 이탈 위험 고객을 조기에 식별하여 선제적 유지(Retention) 전략 수립                            |
| **기간**    | 2024.01                                                               |
| **데이터**   | Kaggle *"Binary Classification with a Bank Churn Dataset"* (165,034행) |
| **핵심 기술** | Python, XGBoost, CatBoost, LightGBM, ONNX, onnxruntime-web               |
| **결과물**   | 분류 모델 6종 비교 · 브라우저 추론 정적 데모(HF Static Space) · Tableau 시각화                      |


핵심은 단순 정확도가 아니라 **실제 유지 전략으로 연결되는 해석 가능성**이었습니다. 연령대·보유 상품 수처럼 마케팅 전략으로 전환 가능한 변수를 중심으로 인사이트를 정리했습니다.

---

## 📊 데이터셋

- **출처**: [Kaggle - Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1)
- **규모**: 165,034개 데이터 포인트
- **타깃**: `Exited` (1 = 이탈, 0 = 유지)


| 컬럼                | 설명                            |
| ----------------- | ----------------------------- |
| `CreditScore`     | 신용 점수                         |
| `Geography`       | 국가 (France / Spain / Germany) |
| `Gender`          | 성별                            |
| `Age`             | 나이                            |
| `Tenure`          | 거래 기간(년)                      |
| `Balance`         | 계좌 잔액                         |
| `NumOfProducts`   | 가입 상품 수                       |
| `HasCrCard`       | 신용카드 보유 여부                    |
| `IsActiveMember`  | 활성 고객 여부                      |
| `EstimatedSalary` | 추정 연봉                         |
| `**Exited`**      | **이탈 여부 (타깃)**                |


---

## 🛠 기술 스택

- **언어/분석**: Python, Pandas, NumPy, scikit-learn
- **모델링**: LogisticRegression, DecisionTree, RandomForest, LightGBM, XGBoost, CatBoost
- **시각화**: Matplotlib, Seaborn, statsmodels(mosaic)
- **배포(데모)**: ONNX, onnxruntime-web(WASM), Hugging Face Static Space

---

## 📂 디렉터리 구조

```
ML_Team_Project/
├── data/                       # 데이터셋
│   ├── train.csv/              # 학습 데이터 (165,034행)
│   ├── test.csv/               # 테스트 데이터
│   ├── bank_turnover/          # 원본 Churn_Modelling.csv
│   └── datalab.xlsx            # 보조 데이터
├── eda_source/                 # 탐색적 데이터 분석(EDA) 노트북
│   ├── visualization.py        # 시각화 모듈 (mosaic/count/hist)
│   ├── KMeans_Clustering.ipynb # 군집 분석 (KMeans / K-Prototype / Cohort)
│   └── ...                     # 단계별 EDA 노트북
├── model_source/               # 모델링 노트북
│   ├── clf_evaluation.py       # 분류 평가 지표 모듈
│   ├── visualization.py        # 시각화 모듈
│   └── ...                     # 모델 학습/튜닝 노트북
├── model/                      # 학습된 모델 (joblib) — lr/dt/rf/lgbm/xgb/cat (+ _gs: GridSearch)
│   └── cat_web.onnx            # 웹 추론용 CatBoost ONNX (ZipMap 제거, [N,2] 확률 출력)
├── demo/                       # 정적 데모 (HF Static Space) — index.html / app.js / cat_web.onnx
├── img/                        # README/시각화 이미지
└── [CASE STUDY] ... .ipynb     # 통합 케이스 스터디 노트북
```

---

## 🔬 분석 프로세스

### 1. 탐색적 데이터 분석 (EDA)

- 연령대·국가·성별·잔액·상품 수와 이탈률의 관계 탐색
- `visualization.py`의 모자이크/카운트/히스토그램 플롯으로 변수별 분포 비교

### 2. 데이터 불균형 처리

이탈 고객이 소수 클래스인 **클래스 불균형 문제**를 두 방식으로 비교했습니다.


| 방법             | 특징                | 결과                          |
| -------------- | ----------------- | --------------------------- |
| `class_weight` | 소수 클래스에 높은 가중치 부여 | 낮은 Precision, **높은 Recall** |
| SMOTE 오버샘플링    | 소수 클래스 합성 샘플 생성   | 높은 Precision, 낮은 Recall     |


> **신규 고객 유치 비용 > 이탈 방지 비용** 이므로, 이탈 고객을 놓치지 않는 것(=Recall)이 중요합니다.
> 따라서 `**class_weight` 방식을 채택**해 Recall을 우선했습니다.

### 3. 모델링 & 평가

6종 분류 모델을 학습·비교하고, `clf_evaluation.py`로 Accuracy/Precision/Recall/F1/AUC를 산출했습니다.

---

## 📈 모델 성능 (테스트셋)


| 모델                 | Accuracy | Precision | Recall     | F1     | AUC        |
| ------------------ | -------- | --------- | ---------- | ------ | ---------- |
| LogisticRegression | 0.7541   | 0.4525    | 0.7398     | 0.5616 | 0.8186     |
| DecisionTree       | 0.8165   | 0.5489    | 0.7737     | 0.6422 | 0.8831     |
| RandomForest       | 0.8148   | 0.5451    | 0.7845     | 0.6433 | 0.8850     |
| LightGBM           | 0.8157   | 0.5462    | 0.7917     | 0.6464 | 0.8883     |
| XGBoost            | 0.8143   | 0.5436    | 0.7957     | 0.6459 | 0.8885     |
| **CatBoost**       | 0.8125   | 0.5404    | **0.7975** | 0.6442 | **0.8889** |


> **AUC 기준 CatBoost가 최고 성능(0.8889)**, Recall도 가장 높아 **최종 배포 모델로 CatBoost를 선정**했습니다.

---

## 💡 주요 인사이트

**특성 중요도: 가입 상품 수 > 나이 > 활성 고객 여부 > 계좌 잔액**

1. **가입 상품 1개 고객의 이탈률이 높음** → 신규 고객 온보딩 시 추가 상품 교차 판매 필요
2. **40~50대의 이탈률이 높고, 젊은 층은 낮음** → 연령대별 차별화된 접근 전략 필요
3. **고객의 54%가 잔고 0원이며 이들의 이탈률(16%)은 오히려 낮음** → 고잔고 고객 유지 전략이 별도로 필요
4. **생애 초기에 거래를 시작한 고객의 이탈률이 낮음** → 청소년·사회 초년생 대상 조기 유치가 장기 충성 고객으로 연결

---

## 🎯 결론 및 제언

### 신규 고객 온보딩 강화

가입 상품(계좌) 수가 1개인 고객의 이탈률이 높습니다. 신규·저활동 고객의 추가 상품 가입을 유도해 활동성을 높이고, 신규 가입 시 결합 상품으로 유도합니다.

### 연령대별 마케팅 전략


| 고객층            | 전략                                  |
| -------------- | ----------------------------------- |
| 젊은 고객 (20~30대) | 디지털 편의성 강화 / 금융 교육 프로그램 / 일상 연계 리워드 |
| 중년 고객 (40~50대) | 투자·재무 계획 지원 / 고급·맞춤형 상품 / 가족 중심 마케팅 |
| 노년 고객 (60대 이상) | 대면 서비스 강화 / 건강·안전 상품 / 상속·자산 계획 서비스 |


### 초기 고객 유치의 중요성

생애 초기 단계에 은행과 긍정적 관계를 형성한 고객은 장기적으로 충성 고객으로 전환될 가능성이 높습니다. 다양한 연령대·제품 묶음에 대한 맞춤형 마케팅이 이탈 감소의 핵심입니다.

---

## 🖥 결과물

- **🚀 라이브 데모 (정적)**: [ingyoun-bank-churn.static.hf.space](https://ingyoun-bank-churn.static.hf.space/) — 고객 정보를 입력하면 이탈 확률을 즉시 예측
- **발표 자료**: [Google Slides](https://docs.google.com/presentation/d/10ULm9v4lCIxGlRADUyCfLKp7EPNeNGgnIIKTCDfqVFM/edit?usp=sharing)

### 정적 데모 아키텍처 (`demo/`)

콜드스타트를 없애기 위해, 학습된 CatBoost 모델을 **ONNX로 변환**해 **브라우저에서 직접 추론**하는 서버리스 정적 페이지로 배포했습니다. ([Hugging Face Static Space](https://huggingface.co/spaces/ingyoun/bank_churn))

```
입력 폼 → JS에서 get_dummies 전처리(13 features)
        → onnxruntime-web(WASM)로 cat_web.onnx 추론
        → 이탈 확률 출력 (서버·콜드스타트 없음)
```

- **모델 변환**: `cat.joblib` → `cat.onnx`(CatBoost 네이티브 export) → ZipMap 제거로 `[N,2]` float 출력 `cat_web.onnx` 생성. Python `predict_proba`와 출력 일치 검증(오차 `~1e-7`).
- **전처리 재현**: 학습 시점 컬럼 순서대로 `Geography`/`Gender` 원-핫을 JS에서 동일하게 구성.
- **호스팅**: HF Static Space (무료 티어, 콜드스타트 0). 모델은 Git LFS로 관리.

---

## ⚙️ 실행 방법

### 1. 분류 평가 모듈 — `clf_evaluation.py`

분류 모델의 평가 지표 계산 및 시각화 함수를 제공합니다.

```python
from clf_evaluation import print_eval, get_eval_by_threshold, roc_curve_plot

# 평가 지표 출력 (Accuracy / Precision / Recall / F1 / ROC AUC)
print_eval(y_test, pred, pred_proba)

# 임계값별 지표 비교
get_eval_by_threshold(y_test, pred_proba, thresholds=[0.3, 0.4, 0.5])

# ROC 곡선
roc_curve_plot(y_test, pred_proba_c1)
```

### 2. 시각화 모듈 — `visualization.py`

EDA용 시각화 함수 3종을 제공합니다.

```python
from visualization import create_mosaic_plot, draw_countplot, draw_histplot

create_mosaic_plot(df, 'Geography', 'Exited')   # 두 변수 간 모자이크 플롯
draw_countplot(df, 'Gender', 'Exited')          # 카테고리 분포 카운트 플롯
draw_histplot(df, 'Balance', 'Exited', bins=30) # 연속형 변수 히스토그램
```


| 함수                                                        | 매개변수          | 설명                         |
| --------------------------------------------------------- | ------------- | -------------------------- |
| `create_mosaic_plot(df, col1, col2)`                      | 데이터프레임, 두 컬럼명 | 두 범주형 변수 관계를 모자이크 플롯으로 시각화 |
| `draw_countplot(df, x, hue, figsize=(10,8))`              | x축 컬럼, 그룹 컬럼  | 카테고리 값 분포 시각화              |
| `draw_histplot(df, x, hue=None, figsize=(10,8), bins=25)` | x축 컬럼, 그룹 컬럼  | 연속형 데이터 분포 히스토그램           |


---

## 👥 팀

머신러닝 팀 프로젝트 1조 · [buzziru/ML_Team_Project](https://github.com/buzziru/ML_Team_Project)