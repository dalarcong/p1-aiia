// Results website for the VM (task 2.4).
// Reads data/detections.csv and data/latest.jpg, published by vm/server.py,
// and refreshes the page every REFRESH_MS milliseconds.

const REFRESH_MS = 5000;
const MAX_POINTS = 120;
const STALE_AFTER_MS = 30000;

const PEOPLE_COLOR = "#3987e5";
const INFERENCE_COLOR = "#d95926";
Chart.defaults.color = "#c3c2b7";
Chart.defaults.borderColor = "#2c2c2a";

function timeSeries(canvasId, color, yTitle) {
  return new Chart(document.getElementById(canvasId), {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { data: [], borderColor: color, borderWidth: 2, pointRadius: 0 },
      ],
    },
    options: {
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          title: { display: true, text: "Time" },
          ticks: { maxTicksLimit: 8 },
        },
        y: {
          title: { display: true, text: yTitle },
          beginAtZero: true,
          ticks: { precision: 0 },
        },
      },
    },
  });
}

// Charts are created once and updated in place on every refresh
const timeline = timeSeries("timeline", PEOPLE_COLOR, "People");
const inferenceChart = timeSeries(
  "inference-chart",
  INFERENCE_COLOR,
  "Inference (ms)",
);

const histogram = new Chart(document.getElementById("histogram"), {
  type: "bar",
  data: {
    labels: [],
    datasets: [{ data: [], backgroundColor: PEOPLE_COLOR }],
  },
  options: {
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { title: { display: true, text: "People in the frame" } },
      y: {
        title: { display: true, text: "Number of frames" },
        beginAtZero: true,
        ticks: { precision: 0 },
      },
    },
  },
});

function parseCsv(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  const fields = header.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(fields.map((field, i) => [field, values[i]]));
  });
}

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;

function updateSeries(chart, rows, field) {
  chart.data.labels = rows.map((row) =>
    new Date(row.captured_at).toLocaleTimeString(),
  );
  chart.data.datasets[0].data = rows.map((row) => Number(row[field]));
  chart.update();
}

function setStatus(lastReceivedAt) {
  const status = document.getElementById("status");
  const ageMs = Date.now() - lastReceivedAt.getTime();
  const live = ageMs < STALE_AFTER_MS;
  status.className = `status ${live ? "live" : "stale"}`;
  status.textContent = live
    ? "Live"
    : `No new frames for ${Math.round(ageMs / 1000)} s`;
}

function render(rows) {
  const counts = rows.map((row) => Number(row.n_people));
  const recent = rows.slice(-MAX_POINTS);

  updateSeries(timeline, recent, "n_people");
  updateSeries(inferenceChart, recent, "inference_ms");

  // Frequency of each person count, from 0 to the maximum seen in this run
  const frequency = new Array(Math.max(...counts) + 1).fill(0);
  counts.forEach((count) => frequency[count]++);
  histogram.data.labels = frequency.map((_, count) => count);
  histogram.data.datasets[0].data = frequency;
  histogram.update();

  const last = rows[rows.length - 1];
  const receivedAt = new Date(last.received_at);
  setText("camera", last.camera_id);
  setText("current", last.n_people);
  setText("mean", mean(counts).toFixed(1));
  setText("max", Math.max(...counts));
  setText("frames", rows.length);
  setText(
    "inference",
    `${mean(rows.map((row) => Number(row.inference_ms))).toFixed(0)} ms`,
  );
  setText(
    "size",
    `${(mean(rows.map((row) => Number(row.frame_bytes))) / 1024).toFixed(0)} KB`,
  );
  setText("updated", receivedAt.toLocaleTimeString());
  setStatus(receivedAt);
}

async function refresh() {
  // The timestamp in the URL stops the browser from reusing a cached copy
  const nocache = `?t=${Date.now()}`;
  try {
    const response = await fetch(`data/detections.csv${nocache}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const rows = parseCsv(await response.text());
    if (rows.length > 0) render(rows);
  } catch (error) {
    // No data yet (e.g. the server has just started): keep the page as it is
    console.warn("Could not load detections.csv:", error);
  }
  document.getElementById("latest").src = `data/latest.jpg${nocache}`;
}

refresh();
setInterval(refresh, REFRESH_MS);
