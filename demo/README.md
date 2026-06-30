---
title: Bank Churn Prediction Demo
emoji: 🏦
colorFrom: indigo
colorTo: red
sdk: static
pinned: false
---

# 은행 이탈 고객 예측 데모 (Static)

고객 정보를 입력하면 **이탈 확률**을 예측하는 정적 데모입니다.
서버 없이 **브라우저에서 ONNX 모델을 직접 추론**하므로 콜드스타트가 없습니다.

## 동작 방식

```
입력 폼 → JS에서 get_dummies 전처리(13 features)
        → onnxruntime-web(WASM)로 cat_web.onnx 추론
        → 이탈 확률 출력
```

- **모델**: CatBoost (학습된 `cat.joblib` → ONNX 변환, ZipMap 제거)
- **입력 피처(13)**: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
  IsActiveMember, EstimatedSalary, Geography(France/Germany/Spain 원-핫), Gender(Female/Male 원-핫)
- **추론**: `onnxruntime-web` (CDN, WASM)

## 파일

| 파일 | 설명 |
|------|------|
| `index.html` | 입력 폼 UI |
| `style.css` | 스타일 |
| `app.js` | 전처리 + ONNX 추론 |
| `cat_web.onnx` | 웹 추론용 모델 (`[N,2]` 확률 출력) |

## 로컬 실행

정적 파일이라 간단한 HTTP 서버만 있으면 됩니다(파일 `file://`로 열면 모델 fetch가 막힘).

```bash
python -m http.server 8000
# http://localhost:8000 접속
```
