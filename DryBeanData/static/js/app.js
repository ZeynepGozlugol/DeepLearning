const FIELDS = [
  "Area",
  "Perimeter",
  "MajorAxisLength",
  "MinorAxisLength",
  "AspectRation",
  "Eccentricity",
  "ConvexArea",
  "EquivDiameter",
  "Extent",
  "Solidity",
  "roundness",
  "Compactness",
  "ShapeFactor1",
  "ShapeFactor2",
  "ShapeFactor3",
  "ShapeFactor4",
];

const PRESETS = {
  BARBUNYA: {
    Area: 69584, Perimeter: 1047.351, MajorAxisLength: 367.078, MinorAxisLength: 242.05,
    AspectRation: 1.51654, Eccentricity: 0.7518, ConvexArea: 71131, EquivDiameter: 297.653,
    Extent: 0.75337, Solidity: 0.97825, roundness: 0.79714, Compactness: 0.81087,
    ShapeFactor1: 0.005275, ShapeFactor2: 0.001407, ShapeFactor3: 0.65751, ShapeFactor4: 0.99714,
  },
  BOMBAY: {
    Area: 171566, Perimeter: 1554.742, MajorAxisLength: 575.336, MinorAxisLength: 381.643,
    AspectRation: 1.50753, Eccentricity: 0.74832, ConvexArea: 173000, EquivDiameter: 467.381,
    Extent: 0.82804, Solidity: 0.99171, roundness: 0.89192, Compactness: 0.81236,
    ShapeFactor1: 0.003353, ShapeFactor2: 0.000901, ShapeFactor3: 0.65993, ShapeFactor4: 0.99486,
  },
  CALI: {
    Area: 74795, Perimeter: 1034.077, MajorAxisLength: 393.361, MinorAxisLength: 243.115,
    AspectRation: 1.618, Eccentricity: 0.78614, ConvexArea: 75468, EquivDiameter: 308.597,
    Extent: 0.80244, Solidity: 0.99108, roundness: 0.87898, Compactness: 0.78451,
    ShapeFactor1: 0.005259, ShapeFactor2: 0.001229, ShapeFactor3: 0.61546, ShapeFactor4: 0.99582,
  },
  DERMASON: {
    Area: 31890, Perimeter: 661.977, MajorAxisLength: 242.999, MinorAxisLength: 167.301,
    AspectRation: 1.45246, Eccentricity: 0.72525, ConvexArea: 32232, EquivDiameter: 201.503,
    Extent: 0.79358, Solidity: 0.98939, roundness: 0.91449, Compactness: 0.82924,
    ShapeFactor1: 0.00762, ShapeFactor2: 0.002223, ShapeFactor3: 0.68763, ShapeFactor4: 0.99876,
  },
  HOROZ: {
    Area: 53802, Perimeter: 929.21, MajorAxisLength: 374.296, MinorAxisLength: 184.607,
    AspectRation: 2.02753, Eccentricity: 0.86991, ConvexArea: 54493, EquivDiameter: 261.73,
    Extent: 0.60684, Solidity: 0.98732, roundness: 0.78303, Compactness: 0.69926,
    ShapeFactor1: 0.006957, ShapeFactor2: 0.001026, ShapeFactor3: 0.48896, ShapeFactor4: 0.99139,
  },
  SEKER: {
    Area: 39180, Perimeter: 709.775, MajorAxisLength: 243.845, MinorAxisLength: 204.65,
    AspectRation: 1.19152, Eccentricity: 0.54372, ConvexArea: 39457, EquivDiameter: 223.351,
    Extent: 0.75731, Solidity: 0.99298, roundness: 0.97731, Compactness: 0.91596,
    ShapeFactor1: 0.006224, ShapeFactor2: 0.002702, ShapeFactor3: 0.83897, ShapeFactor4: 0.99965,
  },
  SIRA: {
    Area: 44595, Perimeter: 791.968, MajorAxisLength: 296.668, MinorAxisLength: 192.288,
    AspectRation: 1.54283, Eccentricity: 0.76151, ConvexArea: 45055, EquivDiameter: 238.286,
    Extent: 0.78468, Solidity: 0.98979, roundness: 0.89347, Compactness: 0.80321,
    ShapeFactor1: 0.006652, ShapeFactor2: 0.001708, ShapeFactor3: 0.64514, ShapeFactor4: 0.99535,
  },
};

function formatValue(name, value) {
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(4);
  return value.toFixed(6);
}

function fillForm(values) {
  FIELDS.forEach((name) => {
    document.getElementById(name).value = formatValue(name, Number(values[name]));
  });
}

function readFeatures() {
  return Object.fromEntries(
    FIELDS.map((name) => [name, Number(document.getElementById(name).value)])
  );
}

function setActiveChip(name) {
  document.querySelectorAll(".chip[data-preset]").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.preset === name);
  });
}

function applyPreset(name) {
  fillForm(PRESETS[name]);
  setActiveChip(name);
}

function renderPrediction(data) {
  const winner = document.getElementById("winner");
  const visual = document.getElementById("winner-visual");

  winner.hidden = false;
  document.getElementById("winner-name").textContent = data.predicted_class;
  document.getElementById("winner-pct").textContent =
    data.probabilities[data.predicted_class].toFixed(1);
  visual.className = `winner-visual ${data.predicted_class}`;
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

async function loadRandomSample() {
  document.getElementById("status-text").textContent = "Rastgele örnek yükleniyor…";
  try {
    const response = await fetch("/sample");
    if (!response.ok) throw new Error("Örnek yüklenemedi");
    const data = await response.json();
    fillForm(data);
    setActiveChip(data.Class);
  } catch (error) {
    document.getElementById("status-text").textContent = error.message;
  }
}

FIELDS.forEach((name) => {
  document.getElementById(name).addEventListener("input", () => {
    document.querySelectorAll(".chip").forEach((chip) => chip.classList.remove("active"));
  });
});

document.querySelectorAll(".chip[data-preset]").forEach((chip) => {
  chip.addEventListener("click", () => applyPreset(chip.dataset.preset));
});

document.getElementById("random-sample").addEventListener("click", loadRandomSample);
document.getElementById("predict-form").addEventListener("submit", predict);

applyPreset("SEKER");
