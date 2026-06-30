// 은행 이탈 고객 예측 — 브라우저 내 ONNX 추론
// 모델 입력 피처 순서는 학습 시점(get_dummies)과 정확히 일치해야 한다.
const FEATURES = [
  "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
  "HasCrCard", "IsActiveMember", "EstimatedSalary",
  "Geography_France", "Geography_Germany", "Geography_Spain",
  "Gender_Female", "Gender_Male",
];

// onnxruntime-web의 WASM 바이너리를 CDN에서 로드 (정적 호스팅이라 동일 버전 사용)
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

let session = null;
const btn = document.getElementById("predict-btn");

// 모델 로드
(async () => {
  try {
    session = await ort.InferenceSession.create("cat_web.onnx", {
      executionProviders: ["wasm"],
    });
    btn.disabled = false;
    btn.textContent = "이탈 확률 예측하기";
  } catch (e) {
    btn.textContent = "모델 로드 실패";
    console.error(e);
  }
})();

// 입력값 → 13차원 Float32 벡터 (학습 시 pd.get_dummies 전처리 재현)
function buildFeatureVector() {
  const num = (id) => parseFloat(document.getElementById(id).value) || 0;
  const radio = (name) =>
    parseFloat(document.querySelector(`input[name="${name}"]:checked`).value);

  const geography = document.getElementById("Geography").value;
  const gender = document.querySelector('input[name="Gender"]:checked').value;

  const vec = {
    CreditScore: num("CreditScore"),
    Age: num("Age"),
    Tenure: num("Tenure"),
    Balance: num("Balance"),
    NumOfProducts: num("NumOfProducts"),
    HasCrCard: radio("HasCrCard"),
    IsActiveMember: radio("IsActiveMember"),
    EstimatedSalary: num("EstimatedSalary"),
    Geography_France: geography === "France" ? 1 : 0,
    Geography_Germany: geography === "Germany" ? 1 : 0,
    Geography_Spain: geography === "Spain" ? 1 : 0,
    Gender_Female: gender === "Female" ? 1 : 0,
    Gender_Male: gender === "Male" ? 1 : 0,
  };
  return Float32Array.from(FEATURES.map((f) => vec[f]));
}

async function predict() {
  if (!session) return;
  const data = buildFeatureVector();
  const tensor = new ort.Tensor("float32", data, [1, FEATURES.length]);
  const out = await session.run({ features: tensor });
  // probability_tensor: [1, 2] → [P(유지), P(이탈)]
  const proba = out.probability_tensor.data[1];
  render(proba);
}

function render(proba) {
  const pct = (proba * 100).toFixed(1);
  const churn = proba >= 0.5;

  const result = document.getElementById("result");
  const verdict = document.getElementById("verdict");
  const fill = document.getElementById("gauge-fill");

  verdict.textContent = churn ? "⚠️ 이탈 위험 고객" : "✅ 유지 예상 고객";
  verdict.className = "verdict " + (churn ? "churn" : "stay");
  fill.style.width = pct + "%";
  fill.style.background = churn ? "var(--accent)" : "var(--safe)";
  document.getElementById("proba").textContent = `이탈 확률 ${pct}%`;
  result.classList.remove("hidden");
  result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---- UI 핸들러 ----
document.getElementById("churn-form").addEventListener("submit", (e) => {
  e.preventDefault();
  predict();
});

// 슬라이더 값 표시
["Tenure", "NumOfProducts"].forEach((id) => {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "-out");
  el.addEventListener("input", () => (out.textContent = el.value));
});

// number 스테퍼 (− / +)
document.querySelectorAll(".step").forEach((b) => {
  b.addEventListener("click", () => {
    const input = document.getElementById(b.dataset.target);
    const step = parseFloat(b.dataset.step || input.step || 1);
    const dir = parseFloat(b.dataset.dir);
    let v = (parseFloat(input.value) || 0) + dir * step;
    const min = parseFloat(input.min), max = parseFloat(input.max);
    if (!isNaN(min)) v = Math.max(min, v);
    if (!isNaN(max)) v = Math.min(max, v);
    input.value = v;
  });
});
