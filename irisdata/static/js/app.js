const PRESETS = {
  setosa: { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2 },
  versicolor: { sepal_length: 5.9, sepal_width: 2.8, petal_length: 4.3, petal_width: 1.3 },
  virginica: { sepal_length: 6.7, sepal_width: 3.0, petal_length: 5.7, petal_width: 2.1 },
};

const fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"];

function syncOutputs() {
  fields.forEach((name) => {
    const input = document.getElementById(name);
    const out = document.getElementById(`out-${name.replaceAll("_", "-")}`);
    out.value = Number(input.value).toFixed(1) + " cm";
  });
}

function readFeatures() {
  return Object.fromEntries(
    fields.map((name) => [name, Number(document.getElementById(name).value)])
  );
}

function applyPreset(name) {
  const values = PRESETS[name];
  fields.forEach((key) => {
    document.getElementById(key).value = values[key];
  });
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.preset === name);
  });
  syncOutputs();
}

function renderPrediction(data) {
  const winner = document.getElementById("winner");
  const visual = document.getElementById("winner-visual");
  const shortName = data.predicted_class.replace("Iris ", "").toLowerCase();

  winner.hidden = false;
  document.getElementById("winner-name").textContent = data.predicted_class;
  document.getElementById("winner-pct").textContent =
    data.probabilities[data.predicted_class].toFixed(1);
  visual.className = `winner-visual ${shortName}`;
  document.getElementById("status-text").textContent =
    "Model softmax çıktısından en yüksek olasılıklı sınıf seçildi.";

  Object.entries(data.probabilities).forEach(([label, pct]) => {
    document.querySelector(`[data-pct="${label}"]`).textContent = `${pct.toFixed(1)}%`;
    document.querySelector(`[data-bar="${label}"]`).style.width = `${pct}%`;
  });

  document.querySelectorAll(".species-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.species === data.predicted_class);
  });
}

async function predict(event) {
  event.preventDefault();
  const button = document.getElementById("predict-btn");
  button.disabled = true;
  document.getElementById("status-text").textContent = "Model çalışıyor…";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readFeatures()),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || "Tahmin alınamadı");
    }
    renderPrediction(await response.json());
  } catch (error) {
    document.getElementById("status-text").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

fields.forEach((name) => {
  document.getElementById(name).addEventListener("input", () => {
    document.querySelectorAll(".chip").forEach((chip) => chip.classList.remove("active"));
    syncOutputs();
  });
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => applyPreset(chip.dataset.preset));
});

document.getElementById("predict-form").addEventListener("submit", predict);
syncOutputs();
applyPreset("setosa");
