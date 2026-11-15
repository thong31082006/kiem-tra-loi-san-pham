"use strict";

const byId = (id) => document.getElementById(id);
const modelNames = { svm: "HOG + SVM", random_forest: "HOG + Random Forest", mobilenet: "MobileNetV3" };
const labelNames = { normal: "Bình thường", defect: "Có dấu hiệu lỗi", uncertain: "Chưa đủ tin cậy" };
const defectNames = {
  defect_low_fill: "Thiếu mức nước", defect_no_cap: "Thiếu nắp",
  defect_loose_cap: "Nắp lỏng / hỏng", defect_debris: "Dị vật",
  defect_label_damage: "Nhãn hỏng / lệch / thiếu",
};
const state = {
  status: null, models: [], currentFile: null, previewUrl: null,
  prediction: null, predicting: false, predictController: null, fileVersion: 0,
  stream: null, activeTab: "inspect", pollTimer: null,
  samples: [], sampleTotal: 0, sampleLimit: 60, sampleRequest: 0,
  reference: null, comparison: [], comparing: false, analysisVersion: 0,
  batch: { id: null, entries: [], running: false, stopRequested: false, model: null },
  reviewItem: null, reviewSaving: false, reviewDrafts: {}, reviewTarget: null,
  presentation: false, presentationReturnTab: "inspect", historyRequest: 0,
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function feedback(id, message, kind = "") {
  const node = byId(id);
  node.textContent = message;
  node.dataset.kind = kind;
}

async function request(url, options = {}) {
  let response;
  try {
    response = await fetch(url, { cache: "no-store", ...options });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    const unavailable = new Error("Không kết nối được với chương trình. Kiểm tra cửa sổ đang chạy máy chủ và thử lại.");
    unavailable.network = true;
    throw unavailable;
  }
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    let detail = data?.detail || data?.message;
    if (Array.isArray(detail)) detail = detail.map((item) => item.msg || String(item)).join("; ");
    if (typeof detail !== "string") detail = "";
    const error = new Error(detail || `Yêu cầu chưa thực hiện được (HTTP ${response.status}).`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function percentage(value) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
}

function countText(value) {
  if (typeof value === "number") return value.toLocaleString("vi-VN");
  if (value && typeof value === "object") {
    const total = value.total ?? value.count ?? value.image_count;
    if (typeof total === "number") return total.toLocaleString("vi-VN");
  }
  return "—";
}

function friendlyModel(model) {
  if (typeof model === "object" && model) return model.name || modelNames[model.id] || model.id;
  return state.models.find((item) => item.id === model)?.name || modelNames[model] || model || "—";
}

function currentModel() {
  return state.models.find((model) => model.id === byId("model-select").value);
}

function trainingRunning() {
  return ["running", "training", "preparing", "queued"].includes(state.status?.training?.state);
}

function updatePredictButton() {
  const ready = Boolean(currentModel()?.ready);
  const busy = state.predicting || state.batch.running;
  byId("predict-button").disabled = !state.currentFile || !ready || busy || trainingRunning();
  byId("predict-button").textContent = state.predicting ? "Đang kiểm tra..." : "Kiểm tra ảnh";
  byId("compare-button").disabled = !state.currentFile || busy || trainingRunning() || state.models.length !== 3 || !state.models.every((model) => model.ready);
  byId("compare-button").textContent = state.comparing ? "Đang so sánh 3 mô hình..." : "So sánh cả 3 mô hình";
  byId("model-select").disabled = busy;
  byId("panel-inspect").setAttribute("aria-busy", String(busy));
  byId("train-models").disabled = busy || trainingRunning() || !state.status;
  byId("clear-image").disabled = !state.currentFile;
  const hint = byId("model-hint");
  if (!state.status) hint.textContent = "Chưa đọc được trạng thái mô hình.";
  else if (trainingRunning()) hint.textContent = "Đang huấn luyện. Có thể kiểm tra ảnh khi hoàn tất.";
  else if (state.batch.running) hint.textContent = "Đang xử lý lô ảnh. Tạm dừng lô trước khi kiểm tra ảnh riêng.";
  else if (!ready) hint.textContent = "Mô hình chưa được huấn luyện. Xem mục Mô hình để bắt đầu.";
  else hint.textContent = "Mô hình đã sẵn sàng. Chỉ cần một ảnh đầu vào.";
  updateBatchControls();
}

function applyModelList(models) {
  if (!Array.isArray(models)) return;
  state.models = models;
  const select = byId("model-select");
  const previous = select.value;
  select.replaceChildren();
  const ordered = [...models].sort((a, b) => ["svm", "random_forest", "mobilenet"].indexOf(a.id) - ["svm", "random_forest", "mobilenet"].indexOf(b.id));
  for (const model of ordered) {
    const option = element("option", model.name || friendlyModel(model.id));
    option.value = model.id;
    select.append(option);
  }
  if (models.some((model) => model.id === previous)) select.value = previous;
  else if (models.some((model) => model.id === "svm")) select.value = "svm";
  if (!select.value && models.length) select.value = models[0].id;
  const batchSelect = byId("batch-model");
  const batchModel = state.batch.model || batchSelect.value || "svm";
  batchSelect.replaceChildren();
  for (const model of ordered) {
    const option = element("option", model.name || friendlyModel(model.id));
    option.value = model.id;
    batchSelect.append(option);
  }
  batchSelect.value = batchModel;
  updatePredictButton();
}

function renderTraining() {
  const training = state.status?.training || {};
  const ready = state.models.some((model) => model.ready);
  const running = trainingRunning();
  byId("train-models").disabled = running || !state.status || state.batch.running || state.predicting;
  byId("train-models").textContent = running ? "Đang huấn luyện..." : (ready ? "Huấn luyện lại" : "Huấn luyện mô hình");
  byId("training-title").textContent = running ? "Đang huấn luyện các mô hình" : training.state === "failed" ? "Huấn luyện chưa hoàn tất" : ready ? "Mô hình đã sẵn sàng" : "Chưa có mô hình sẵn sàng";
  byId("training-message").textContent = training.message || (ready ? "Có thể chọn mô hình ở mục Kiểm tra để phân loại ảnh." : "Bấm Huấn luyện mô hình để tạo các mô hình từ dữ liệu đã chuẩn bị.");
  const progress = byId("training-progress");
  progress.hidden = !running;
  if (typeof training.progress === "number") {
    progress.value = Math.max(0, Math.min(100, training.progress <= 1 ? training.progress * 100 : training.progress));
  } else {
    progress.removeAttribute("value");
  }
}

function renderDataset(dataset = {}, scope = "") {
  dataset = dataset || {};
  const facts = byId("dataset-facts");
  facts.replaceChildren();
  const items = [["Tổng ảnh", countText(dataset.image_count ?? dataset.total)], ["Train", countText(dataset.train)], ["Validation", countText(dataset.val)], ["Test", countText(dataset.test)]];
  for (const [name, value] of items) {
    const block = element("div");
    block.append(element("dt", name), element("dd", value));
    facts.append(block);
  }
  const source = typeof dataset.source === "object" ? dataset.source.name || dataset.source.title || dataset.source.url : dataset.source;
  if (source) {
    const block = element("div");
    block.append(element("dt", "Nguồn dữ liệu"), element("dd", source));
    facts.append(block);
  }
  const scopeText = scope || dataset.scope || "Kết quả đánh giá chỉ áp dụng cho dữ liệu và điều kiện chụp đã sử dụng.";
  byId("dataset-scope").textContent = scopeText;
  byId("inspection-scope").textContent = scopeText;
}

function renderModels(payload) {
  const models = Array.isArray(payload.models) ? payload.models : state.models;
  const body = byId("models-body");
  body.replaceChildren();
  if (!models.length) {
    const row = element("tr");
    const cell = element("td", "Chưa có thông tin mô hình.", "empty-cell");
    cell.colSpan = 6;
    row.append(cell);
    body.append(row);
  }
  for (const model of models) {
    const row = element("tr");
    row.append(element("td", friendlyModel(model)));
    const status = element("td");
    status.append(element("span", model.ready ? "Sẵn sàng" : "Chưa huấn luyện", `model-status${model.ready ? " is-ready" : ""}`));
    row.append(status);
    const metrics = model.metrics?.test || model.metrics || {};
    for (const metric of ["accuracy", "precision", "recall", "f1"]) row.append(element("td", percentage(metrics[metric] ?? metrics[`${metric}_macro`])));
    body.append(row);
  }
  const report = payload.report || {};
  const note = report.metrics_note || report.evaluation_note || report.note;
  byId("metrics-note").textContent = typeof note === "string" ? note : "Chỉ số lấy từ tập kiểm tra của lần huấn luyện đã lưu. Điểm mô hình của một ảnh không bảo đảm ảnh được phân loại đúng.";
  const pipeline = payload.end_to_end || {};
  const pipelineBody = byId("pipeline-body");
  pipelineBody.replaceChildren();
  byId("pipeline-results").hidden = !pipeline.models;
  if (pipeline.models) {
    for (const [modelId, metrics] of Object.entries(pipeline.models)) {
      const row = element("tr");
      row.append(element("td", friendlyModel(modelId)));
      for (const key of ["accuracy", "coverage", "recall", "f1"]) row.append(element("td", percentage(metrics[key])));
      row.append(element("td", metrics.uncertain_count ?? "—"));
      pipelineBody.append(row);
    }
    byId("pipeline-note").textContent = pipeline.note || "Đánh giá trên cùng các khối ảnh của bộ nguồn; chưa chứng minh khả năng sử dụng ở mọi góc chụp.";
  }
  renderDataset(payload.dataset || state.status?.dataset, payload.scope || report.scope || state.status?.scope);
  renderEvaluation(models);
  renderTypeStatus(payload.defect_types);
}

function renderTypeStatus(info = {}) {
  const classes = (info.classes || []).filter((item) => item.enabled);
  const disabled = (info.classes || []).filter((item) => !item.enabled);
  const training = info.training?.state === "running";
  byId("type-capabilities").textContent = training ? `Đang huấn luyện: ${info.training.message}` : info.ready
    ? `Đã bật: ${classes.map((item) => item.name).join("; ")}. ${disabled.length ? `Chưa đạt điều kiện gọi tên: ${disabled.map((item) => item.name).join("; ")}. ` : ""}Phạm vi: mẫu chai Ron 88 của bộ dữ liệu.`
    : "Bộ phân tích loại lỗi chưa sẵn sàng. Kết quả bình thường / có lỗi vẫn dùng ba bộ phân loại hiện có.";
  byId("type-model-section").hidden = !info.ready;
  if (!info.ready) return;
  byId("type-model-description").textContent = `${info.model_name} · ${info.scope}`;
  const body = byId("type-metrics-body");
  body.replaceChildren();
  for (const item of info.classes || []) {
    const metrics = info.test?.accepted?.per_class?.[item.code] || {};
    const row = element("tr");
    row.append(element("td", item.name), element("td", item.enabled ? "Đã bật" : "Chưa đạt Validation"),
      element("td", info.counts?.train?.[item.code] ?? "—"), element("td", metrics.support ?? "—"));
    for (const key of ["precision", "recall", "f1"]) row.append(element("td", percentage(metrics[key])));
    body.append(row);
  }
}

function renderEvaluation(models) {
  const chart = byId("f1-chart");
  const grid = byId("confusion-grid");
  chart.replaceChildren();
  grid.replaceChildren();
  byId("evaluation-visuals").hidden = true;
  for (const model of models) {
    const metrics = model.metrics?.test || model.metrics || {};
    if (typeof metrics.f1 === "number" && Number.isFinite(metrics.f1)) {
      byId("evaluation-visuals").hidden = false;
      const row = element("div", null, "f1-row");
      const track = element("div", null, "f1-track");
      const bar = element("span");
      bar.style.width = `${Math.max(0, Math.min(1, metrics.f1)) * 100}%`;
      track.append(bar);
      row.append(element("span", friendlyModel(model)), track, element("strong", percentage(metrics.f1)));
      chart.append(row);
    }
    const matrix = metrics.confusion_matrix;
    if (!Array.isArray(matrix) || matrix.length !== 2 || matrix.some((row) => !Array.isArray(row) || row.length !== 2)) continue;
    const order = metrics.class_order || ["normal", "defect"];
    const card = element("article", null, "confusion-card");
    card.append(element("h4", friendlyModel(model)));
    const table = element("table", null, "confusion-table");
    const caption = element("caption", "Hàng: nhãn thật · Cột: dự đoán");
    const head = element("thead");
    const header = element("tr");
    header.append(element("th", "Thật ↓ / Đoán →"));
    for (const key of order) header.append(element("th", key === "normal" ? "Bình thường" : "Có lỗi"));
    head.append(header);
    const body = element("tbody");
    matrix.forEach((values, index) => {
      const row = element("tr");
      const heading = element("th", order[index] === "normal" ? "Bình thường" : "Có lỗi");
      heading.scope = "row";
      row.append(heading);
      values.forEach((value, column) => row.append(element("td", value, index === column ? "matrix-correct" : "matrix-error")));
      body.append(row);
    });
    table.append(caption, head, body);
    card.append(table);
    const normalIndex = order.indexOf("normal"), defectIndex = order.indexOf("defect");
    if (normalIndex >= 0 && defectIndex >= 0) card.append(element("p", `Báo nhầm lỗi: ${matrix[normalIndex][defectIndex]} · Bỏ sót lỗi: ${matrix[defectIndex][normalIndex]}`, "field-hint"));
    grid.append(card);
  }
}

async function loadStatus() {
  clearTimeout(state.pollTimer);
  try {
    const payload = await request("/api/status");
    state.status = payload;
    applyModelList(payload.models || []);
    renderTraining();
    renderDataset(payload.dataset || {}, payload.scope);
    renderTypeStatus(payload.defect_types);
    const ready = state.models.some((model) => model.ready);
    byId("connection-state").textContent = trainingRunning() ? "Đang huấn luyện" : ready ? "Mô hình sẵn sàng" : "Chưa huấn luyện";
    byId("connection-state").dataset.state = ready && !trainingRunning() ? "ready" : "pending";
    if (state.activeTab === "models") await loadModels(false);
    const allReady = state.models.length > 0 && state.models.every((model) => model.ready);
    if (trainingRunning() || !allReady || payload.defect_types?.training?.state === "running") state.pollTimer = setTimeout(loadStatus, 3000);
  } catch (error) {
    state.status = null;
    byId("connection-state").textContent = "Mất kết nối";
    byId("connection-state").dataset.state = "error";
    byId("train-models").disabled = true;
    updatePredictButton();
    byId("model-hint").textContent = error.message;
    state.pollTimer = setTimeout(loadStatus, 5000);
  }
}

async function loadModels(showLoading = true) {
  if (showLoading) feedback("models-feedback", "Đang đọc kết quả đánh giá...");
  try {
    const payload = await request("/api/models");
    if (Array.isArray(payload.models)) applyModelList(payload.models);
    renderModels(payload);
    feedback("models-feedback", "");
  } catch (error) {
    feedback("models-feedback", error.message, "error");
  }
}

function resetPrediction(keepComparison = false) {
  state.prediction = null;
  byId("result-content").hidden = true;
  byId("result-placeholder").hidden = false;
  byId("result-label").textContent = "";
  byId("reference-match").hidden = true;
  byId("type-analysis-panel").hidden = true;
  if (!keepComparison) {
    state.comparison = [];
    state.analysisVersion += 1;
    byId("comparison-section").hidden = true;
    byId("analysis-section").hidden = true;
    byId("crop-preview").removeAttribute("src");
    byId("hog-preview").removeAttribute("src");
  }
  feedback("action-feedback", "");
}

function setFile(file, reference = null) {
  const allowed = /\.(jpe?g|png|webp|bmp)$/i.test(file.name);
  if (!file.type.startsWith("image/") && !allowed) {
    feedback("action-feedback", "Hãy chọn tệp ảnh JPG, PNG, WebP hoặc BMP.", "error");
    return false;
  }
  if (state.predictController) state.predictController.abort();
  state.fileVersion += 1;
  state.predicting = false;
  state.comparing = false;
  state.currentFile = file;
  state.reference = reference;
  renderReference();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  const image = byId("image-preview");
  image.src = state.previewUrl;
  image.hidden = false;
  byId("drop-placeholder").hidden = true;
  byId("drop-zone").classList.add("has-image");
  byId("file-caption").textContent = `${file.name} · ${file.size < 1024 * 1024 ? `${Math.round(file.size / 1024)} KB` : `${(file.size / (1024 * 1024)).toFixed(1)} MB`}`;
  resetPrediction();
  updatePredictButton();
  return true;
}

function clearImage() {
  if (state.predictController) state.predictController.abort();
  state.fileVersion += 1;
  state.currentFile = null;
  state.predicting = false;
  state.comparing = false;
  state.reference = null;
  renderReference();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  byId("image-file").value = "";
  byId("image-preview").removeAttribute("src");
  byId("image-preview").hidden = true;
  byId("drop-placeholder").hidden = false;
  byId("drop-zone").classList.remove("has-image");
  byId("file-caption").textContent = "Chưa chọn ảnh.";
  resetPrediction();
  updatePredictButton();
}

function renderPrediction(result) {
  state.prediction = result;
  const label = labelNames[result.label] ? result.label : "uncertain";
  const labelNode = byId("result-label");
  labelNode.textContent = labelNames[label];
  labelNode.dataset.label = label;
  byId("result-note").textContent = label === "uncertain" ? "Điểm dự đoán chưa đủ để kết luận. Thử chụp rõ hơn hoặc đổi góc nhìn." : label === "defect" ? "Mô hình nhận thấy dấu hiệu thuộc nhóm ảnh có lỗi." : "Ảnh được xếp vào nhóm bình thường trong phạm vi mô hình đã học.";
  byId("result-confidence").textContent = percentage(result.confidence);
  byId("result-model").textContent = result.model_name || friendlyModel(result.model);
  byId("result-latency").textContent = result.latency_ms !== null && result.latency_ms !== undefined && Number.isFinite(Number(result.latency_ms)) ? `${Number(result.latency_ms).toFixed(0)} ms` : "—";
  const probabilities = result.probabilities;
  const hasScores = probabilities && Number.isFinite(Number(probabilities.normal)) && Number.isFinite(Number(probabilities.defect));
  byId("score-distribution").hidden = !hasScores;
  if (hasScores) {
    for (const key of ["normal", "defect"]) {
      const value = Math.max(0, Math.min(1, Number(probabilities[key])));
      byId(`score-${key}`).style.width = `${value * 100}%`;
      byId(`score-${key}-text`).textContent = percentage(value);
    }
  }
  byId("saved-note").textContent = result.id !== undefined && result.id !== null ? "Đã lưu lần kiểm tra này trong Lịch sử." : "";
  byId("result-placeholder").hidden = true;
  byId("result-content").hidden = false;
  renderTypeAnalysis(result);
  const match = referenceMatch(result);
  byId("reference-match").hidden = !match;
  byId("reference-match").textContent = match;
  byId("reference-match").dataset.match = result.label === state.reference?.label ? "yes" : "no";
  byId("review-result").disabled = !result.id;
  byId("result-review-summary").textContent = reviewSummary(result);
}

function renderTypeAnalysis(result) {
  const analysis = result.defect_analysis;
  byId("type-analysis-panel").hidden = !analysis;
  if (!analysis) return;
  const review = analysis.agreement === "review";
  byId("type-analysis-panel").dataset.status = review ? "review" : analysis.status;
  byId("type-analysis-name").textContent = analysis.name || "Chưa xác định được loại lỗi";
  let note;
  if (analysis.status === "typed") {
    note = `Điểm cho loại lỗi: ${percentage(analysis.score)}. ` + (review
      ? "Kết quả bổ sung khác hoặc chưa khớp kết luận chất lượng bên trên; cần chụp rõ hơn và kiểm tra lại."
      : "Đây là loại lỗi chính do bộ AI bổ sung dự đoán từ nội dung ảnh, không lấy từ nhãn thư viện.");
  } else if (analysis.status === "normal") {
    note = review ? "Bộ phân tích loại lỗi chưa xác nhận được lỗi phù hợp với kết quả bên trên. Hai bộ phân tích cần được đối chiếu; chưa đủ cơ sở gọi tên lỗi."
      : "Bộ AI bổ sung không nhận ra các lỗi đã học trong ảnh này. Móp chai và nước đục chưa được kiểm tra.";
  } else {
    note = analysis.status === "unavailable" ? "Vẫn có thể dùng kết quả phân loại chất lượng. Bộ phân tích loại lỗi hiện chưa sẵn sàng."
      : "Điểm chưa đạt ngưỡng gọi tên loại lỗi đã chọn bằng Validation. Thử chụp thêm góc khác; app chưa xác định được lỗi cụ thể.";
  }
  if (analysis.status === "normal" && review) byId("type-analysis-name").textContent = "Chưa thống nhất được loại lỗi";
  byId("type-analysis-note").textContent = note;
  byId("type-analysis-model").textContent = analysis.model_name ? `Bộ phân tích: ${analysis.model_name}. Chưa khoanh vị trí lỗi; nắp lỏng/hỏng chỉ là dấu hiệu ngoại quan.` : "";
}

function defectDescription(sample) {
  if (sample.label === "normal") return "Bình thường";
  return sample.defect_types?.length ? sample.defect_types.map((key) => defectNames[key] || key).join("; ") : "Có lỗi (chưa có mô tả loại lỗi)";
}

function renderReference() {
  const sample = state.reference;
  byId("reference-panel").hidden = !sample;
  if (!sample) return;
  byId("reference-label").textContent = defectDescription(sample);
  byId("reference-split").textContent = `Tập ${sample.split === "val" ? "Validation" : sample.split === "train" ? "Train" : "Test"} · Đây là nhãn tham chiếu, chưa phải loại lỗi do AI xác định.`;
}

function referenceMatch(result) {
  if (!state.reference || result.error) return "";
  if (result.label === "uncertain") return "Chưa có kết luận để đối chiếu nhãn tham chiếu.";
  return result.label === state.reference.label ? "Khớp nhãn bình thường / có lỗi của ảnh thư viện." : "Khác nhãn bình thường / có lỗi của ảnh thư viện.";
}

async function renderAnalysis(analysis) {
  const version = ++state.analysisVersion;
  byId("analysis-section").hidden = true;
  if (!analysis || !state.currentFile) return;
  const picture = new Image();
  picture.src = state.previewUrl;
  try { await picture.decode(); } catch { return; }
  if (version !== state.analysisVersion) return;
  const canvas = byId("bottle-overlay");
  const scale = Math.min(1, 1200 / picture.naturalWidth);
  canvas.width = Math.round(picture.naturalWidth * scale);
  canvas.height = Math.round(picture.naturalHeight * scale);
  const context = canvas.getContext("2d");
  context.drawImage(picture, 0, 0, canvas.width, canvas.height);
  const [x1, y1, x2, y2] = analysis.crop_bbox;
  const sx = canvas.width / analysis.image_size[0], sy = canvas.height / analysis.image_size[1];
  context.strokeStyle = "#36e2b0";
  context.lineWidth = Math.max(2, canvas.width / 300);
  context.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
  byId("crop-preview").src = analysis.crop;
  byId("hog-preview").src = analysis.hog;
  byId("analysis-section").hidden = false;
}

function renderComparison(results) {
  state.comparison = results;
  const grid = byId("comparison-cards");
  grid.replaceChildren();
  const valid = results.filter((result) => !result.error);
  for (const result of results) {
    const card = element("article", null, "comparison-card");
    card.append(element("h3", result.model_name || friendlyModel(result.model)));
    const badge = element("p", result.error ? "Không có kết quả" : labelNames[result.label], "result-badge");
    badge.dataset.label = result.error ? "uncertain" : result.label;
    card.append(badge);
    if (result.error) card.append(element("p", result.error, "field-hint"));
    else {
      const score = element("p", percentage(result.confidence), "comparison-score");
      score.append(element("span", "Điểm mô hình"));
      card.append(score, element("p", `${Math.round(result.latency_ms)} ms${result.cold_start ? " · Lần đầu nạp mô hình" : ""}`, "field-hint"));
      const match = referenceMatch(result);
      if (match) card.append(element("p", match, "field-hint"));
      if (result.defect_analysis) {
        const analysis = result.defect_analysis;
        card.append(element("p", analysis.status === "typed" ? `AI bổ sung: ${analysis.name}${analysis.agreement === "review" ? " · Cần đối chiếu" : ""}` : "AI bổ sung: chưa gọi tên lỗi cụ thể", "field-hint"));
      }
      const button = element("button", "Xem chi tiết", "text-button");
      button.type = "button";
      button.addEventListener("click", () => {
        byId("model-select").value = result.model;
        renderPrediction(result);
        byId("result-heading").scrollIntoView({ behavior: "smooth", block: "center" });
      });
      card.append(button);
    }
    grid.append(card);
  }
  const same = valid.length === 3 && new Set(valid.map((item) => item.label)).size === 1;
  byId("comparison-summary").textContent = same ? `Cả 3 mô hình cùng trả về: ${labelNames[valid[0].label]}.` : `Có ${valid.length}/3 mô hình trả kết quả. Các kết quả cần được xem riêng; không tự động lấy đa số làm kết luận.`;
  byId("comparison-section").hidden = false;
}

function exportComparison() {
  if (!state.comparison.length) return;
  const rows = [["Tên ảnh", "Mô hình", "Kết quả", "Điểm mô hình", "Thời gian (ms)", "Nạp lần đầu", "Nhãn tham chiếu thư viện", "Ghi chú", "AI phân tích loại lỗi", "Bộ phân tích loại lỗi", "Đối chiếu hai bộ phân tích"]];
  for (const item of state.comparison) rows.push([state.currentFile.name, item.model_name || friendlyModel(item.model), item.error ? "Không có kết quả" : labelNames[item.label], item.confidence ?? "", item.latency_ms ?? "", item.cold_start ? "Có" : "Không", state.reference ? defectDescription(state.reference) : "", item.error || "Điểm mô hình chưa hiệu chuẩn", item.defect_analysis?.name || "", item.defect_analysis?.model_name || "", item.defect_analysis?.agreement || ""]);
  const csv = rows.map((row) => row.map((value) => {
    let text = String(value);
    if (text.trimStart().match(/^[=+\-@]/)) text = "'" + text;
    return `"${text.replaceAll('"', '""')}"`;
  }).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" }));
  const link = element("a");
  link.href = url;
  link.download = "so_sanh_3_mo_hinh.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function predict(compare = false) {
  if (byId(compare ? "compare-button" : "predict-button").disabled || !state.currentFile) return;
  resetPrediction();
  state.predicting = true;
  state.comparing = compare;
  const version = state.fileVersion;
  const controller = new AbortController();
  state.predictController = controller;
  const form = new FormData();
  form.append("file", state.currentFile, state.currentFile.name);
  if (!compare) form.append("model", byId("model-select").value);
  feedback("action-feedback", compare ? "Đang chạy lần lượt SVM, Random Forest và MobileNet trên cùng ảnh..." : "Đang tự tìm vùng chai và phân loại ảnh...");
  updatePredictButton();
  try {
    const result = await request(compare ? "/api/compare" : "/api/predict", { method: "POST", body: form, signal: controller.signal });
    if (version !== state.fileVersion) return;
    if (compare) {
      renderComparison(result.results);
      const selected = result.results.find((item) => item.model === byId("model-select").value && !item.error) || result.results.find((item) => !item.error);
      byId("model-select").value = selected.model;
      renderPrediction(selected);
    } else renderPrediction(result);
    await renderAnalysis(result.analysis);
    if (version !== state.fileVersion) return;
    feedback("action-feedback", compare ? "Đã so sánh. Kết quả của từng mô hình thành công được lưu trong Lịch sử." : "Đã kiểm tra ảnh.", "success");
  } catch (error) {
    if (error.name !== "AbortError" && version === state.fileVersion) {
      resetPrediction();
      feedback("action-feedback", error.message, "error");
    }
  } finally {
    if (version === state.fileVersion) {
      state.predicting = false;
      state.comparing = false;
      state.predictController = null;
      updatePredictButton();
    }
  }
}

async function trainModels() {
  if (state.batch.running || state.predicting) return;
  byId("train-models").disabled = true;
  feedback("models-feedback", "Đang bắt đầu huấn luyện ba mô hình...");
  try {
    const payload = await request("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ epochs: 8, batch_size: 8 }) });
    if (state.status) state.status.training = payload.training || payload;
    renderTraining();
    updatePredictButton();
    feedback("models-feedback", payload.message || "Đã bắt đầu huấn luyện. Có thể theo dõi trạng thái tại đây.");
    await loadStatus();
  } catch (error) {
    feedback("models-feedback", error.message, "error");
    byId("train-models").disabled = !state.status || trainingRunning();
  }
}

function activateTab(name, focusTab = false) {
  state.activeTab = name;
  for (const tab of document.querySelectorAll("[data-tab]")) {
    const selected = tab.dataset.tab === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    byId(`panel-${tab.dataset.tab}`).hidden = !selected;
    if (selected && focusTab) tab.focus();
  }
  if (name === "models") loadModels();
  if (name === "samples") loadSamples();
  if (name === "history") loadHistory();
}

async function loadSamples() {
  const version = ++state.sampleRequest;
  const split = byId("sample-split").value;
  feedback("samples-feedback", "Đang đọc thư viện ảnh...");
  byId("load-more-samples").disabled = true;
  try {
    const label = byId("sample-label").value;
    const payload = await request(`/api/samples?split=${encodeURIComponent(split)}&label=${encodeURIComponent(label)}&limit=${state.sampleLimit}`);
    if (version !== state.sampleRequest) return;
    state.samples = Array.isArray(payload.samples) ? payload.samples : [];
    state.sampleTotal = Number(payload.total ?? state.samples.length);
    renderSamples();
    feedback("samples-feedback", state.samples.length ? "" : "Tập dữ liệu này chưa có ảnh.");
  } catch (error) {
    if (version !== state.sampleRequest) return;
    state.samples = [];
    state.sampleTotal = 0;
    renderSamples();
    feedback("samples-feedback", error.message, "error");
  } finally {
    if (version === state.sampleRequest) byId("load-more-samples").disabled = false;
  }
}

function renderSamples() {
  const grid = byId("sample-grid");
  grid.replaceChildren();
  const selectedLabel = byId("sample-label").value;
  const samples = state.samples.filter((sample) => selectedLabel === "all" || sample.label === selectedLabel);
  byId("samples-count").textContent = `${samples.length} ảnh đang hiển thị · ${state.sampleTotal} ảnh trong tập`;
  for (const sample of samples) {
    const button = element("button", null, "sample-item");
    button.type = "button";
    const fileName = sample.filename || sample.name || String(sample.id);
    button.setAttribute("aria-label", `Kiểm tra ảnh ${fileName}, nhãn nguồn ${labelNames[sample.label] || sample.label || "chưa rõ"}`);
    const image = element("img");
    image.src = sample.image_url;
    image.alt = `Chai nhựa trong ảnh ${fileName}`;
    image.loading = "lazy";
    const text = element("span", null, "sample-item-text");
    text.append(element("span", fileName, "sample-item-name"), element("span", `Nhãn nguồn: ${defectDescription(sample)}`, "sample-item-label"));
    button.append(image, text);
    button.addEventListener("click", () => selectSample(sample, button));
    grid.append(button);
  }
  byId("load-more-samples").hidden = state.samples.length >= state.sampleTotal;
}

async function selectSample(sample, button) {
  button.disabled = true;
  feedback("samples-feedback", "Đang mở ảnh đã chọn...");
  try {
    const response = await fetch(sample.image_url);
    if (!response.ok) throw new Error("Không đọc được ảnh trong thư viện.");
    const blob = await response.blob();
    const fileName = sample.filename || sample.name || `${String(sample.id).replace(/[^a-zA-Z0-9_.-]/g, "_")}.jpg`;
    const file = new File([blob], fileName, { type: blob.type || "image/jpeg" });
    if (!setFile(file, sample)) return;
    feedback("samples-feedback", "");
    activateTab("inspect");
    feedback("action-feedback", "Đã chọn ảnh thư viện. Bấm Kiểm tra ảnh để chạy mô hình.");
    byId("predict-button").focus();
  } catch (error) {
    feedback("samples-feedback", error.message || "Không mở được ảnh đã chọn.", "error");
  } finally {
    button.disabled = false;
  }
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
}

async function loadHistory() {
  const version = ++state.historyRequest;
  feedback("history-feedback", "Đang đọc lịch sử...");
  try {
    const payload = await request(`/api/history?limit=100&review=${encodeURIComponent(byId("history-review-filter").value)}`);
    if (version !== state.historyRequest) return;
    renderHistory(Array.isArray(payload.items) ? payload.items : []);
    feedback("history-feedback", "");
  } catch (error) {
    if (version !== state.historyRequest) return;
    feedback("history-feedback", error.message, "error");
  }
}

function renderHistory(items) {
  const body = byId("history-body");
  body.replaceChildren();
  if (!items.length) {
    const cell = element("td", "Chưa có lần kiểm tra nào được lưu.", "empty-cell");
    cell.colSpan = 6;
    const row = element("tr");
    row.append(cell);
    body.append(row);
  }
  for (const item of items) {
    const row = element("tr");
    const imageCell = element("td");
    if (item.image_url) {
      const image = element("img", null, "history-thumbnail");
      image.src = item.image_url;
      image.alt = `Ảnh kiểm tra ${item.id}`;
      image.loading = "lazy";
      imageCell.append(image);
    } else imageCell.textContent = "—";
    row.append(imageCell, element("td", formatDate(item.created_at)), element("td", item.model_name || friendlyModel(item.model)));
    const resultCell = element("td", labelNames[item.label] || "Chưa đủ tin cậy", "history-result");
    if (item.defect_analysis?.status === "typed") resultCell.append(element("small", `${item.defect_analysis.name}${item.defect_analysis.agreement === "review" ? " · Cần đối chiếu" : ""}`, "history-defect-type"));
    resultCell.dataset.label = labelNames[item.label] ? item.label : "uncertain";
    resultCell.append(element("small", reviewSummary(item), "history-review-summary"));
    row.append(resultCell, element("td", percentage(item.confidence)));
    const actionCell = element("td");
    const review = element("button", "Đánh giá", "text-button");
    review.type = "button";
    review.setAttribute("aria-label", `Đánh giá ${item.filename || item.id}`);
    review.addEventListener("click", () => openReview(item));
    const remove = element("button", "Xóa", "delete-button");
    remove.type = "button";
    remove.setAttribute("aria-label", `Xóa lần kiểm tra ${item.id}`);
    remove.addEventListener("click", () => deleteHistoryItem(item, remove));
    actionCell.append(review, remove);
    row.append(actionCell);
    body.append(row);
  }
}

async function deleteHistoryItem(item, button) {
  if (!window.confirm("Xóa bản ghi kiểm tra này khỏi lịch sử?")) return;
  button.disabled = true;
  try {
    await request(`/api/history/${encodeURIComponent(item.id)}`, { method: "DELETE" });
    await loadHistory();
  } catch (error) {
    feedback("history-feedback", error.message, "error");
    button.disabled = false;
  }
}

function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  byId("camera-video").srcObject = null;
  byId("capture-image").disabled = true;
}

async function openCamera() {
  const dialog = byId("camera-dialog");
  dialog.showModal();
  feedback("camera-feedback", "Đang mở camera...");
  if (!navigator.mediaDevices?.getUserMedia) {
    feedback("camera-feedback", "Trình duyệt chưa cho phép dùng camera tại địa chỉ này. Hãy mở bằng localhost hoặc chọn ảnh từ máy.", "error");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
    if (!dialog.open) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    state.stream = stream;
    const video = byId("camera-video");
    video.srcObject = stream;
    await video.play();
    byId("capture-image").disabled = false;
    feedback("camera-feedback", "");
  } catch (error) {
    stopCamera();
    const message = ["NotAllowedError", "PermissionDeniedError"].includes(error.name) ? "Chưa được cấp quyền camera. Cho phép camera trong trình duyệt rồi thử lại, hoặc chọn ảnh từ máy." : error.name === "NotFoundError" ? "Không tìm thấy camera. Bạn có thể chọn ảnh từ máy." : "Không mở được camera. Kiểm tra camera có đang được ứng dụng khác sử dụng rồi thử lại.";
    feedback("camera-feedback", message, "error");
  }
}

function captureImage() {
  const video = byId("camera-video");
  if (!video.videoWidth || !video.videoHeight) {
    feedback("camera-feedback", "Camera chưa có hình. Chờ một chút rồi chụp lại.");
    return;
  }
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    if (!blob) {
      feedback("camera-feedback", "Chưa tạo được ảnh. Hãy chụp lại.", "error");
      return;
    }
    setFile(new File([blob], `chai_${Date.now()}.jpg`, { type: "image/jpeg" }));
    byId("camera-dialog").close();
    feedback("action-feedback", "Đã chụp ảnh. Bấm Kiểm tra ảnh để chạy mô hình.");
  }, "image/jpeg", 0.94);
}

function bindEvents() {
  bindWorkflowEvents();
  const tabs = [...document.querySelectorAll("[data-tab]")];
  for (const tab of tabs) {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      const index = tabs.indexOf(tab);
      let target;
      if (event.key === "ArrowRight") target = tabs[(index + 1) % tabs.length];
      if (event.key === "ArrowLeft") target = tabs[(index - 1 + tabs.length) % tabs.length];
      if (event.key === "Home") target = tabs[0];
      if (event.key === "End") target = tabs[tabs.length - 1];
      if (target) {
        event.preventDefault();
        activateTab(target.dataset.tab, true);
      }
    });
  }
  const input = byId("image-file");
  byId("choose-file").addEventListener("click", () => input.click());
  input.addEventListener("change", () => { if (input.files[0]) setFile(input.files[0]); input.value = ""; });
  byId("image-preview").addEventListener("error", () => {
    clearImage();
    feedback("action-feedback", "Không đọc được ảnh này. Hãy chọn một tệp ảnh hợp lệ khác.", "error");
  });
  const dropZone = byId("drop-zone");
  dropZone.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); input.click(); } });
  for (const eventName of ["dragenter", "dragover"]) dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add("is-dragging"); });
  for (const eventName of ["dragleave", "drop"]) dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("is-dragging"); });
  dropZone.addEventListener("drop", (event) => { if (event.dataTransfer.files[0]) setFile(event.dataTransfer.files[0]); });
  byId("clear-image").addEventListener("click", clearImage);
  byId("predict-button").addEventListener("click", () => predict(false));
  byId("compare-button").addEventListener("click", () => predict(true));
  byId("export-comparison").addEventListener("click", exportComparison);
  byId("model-select").addEventListener("change", () => {
    if (state.predictController) state.predictController.abort();
    state.fileVersion += 1;
    state.predicting = false;
    resetPrediction(true);
    updatePredictButton();
  });
  byId("browse-samples").addEventListener("click", () => activateTab("samples"));
  byId("sample-split").addEventListener("change", () => { state.sampleLimit = 60; loadSamples(); });
  byId("sample-label").addEventListener("change", () => { state.sampleLimit = 60; loadSamples(); });
  byId("load-more-samples").addEventListener("click", () => { state.sampleLimit += 60; loadSamples(); });
  byId("refresh-models").addEventListener("click", async () => { await loadStatus(); await loadModels(); });
  byId("train-models").addEventListener("click", trainModels);
  byId("refresh-history").addEventListener("click", loadHistory);
  byId("open-camera").addEventListener("click", openCamera);
  byId("close-camera").addEventListener("click", () => byId("camera-dialog").close());
  byId("camera-dialog").addEventListener("close", stopCamera);
  byId("capture-image").addEventListener("click", captureImage);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) loadStatus(); });
  window.addEventListener("pagehide", () => { stopCamera(); if (state.previewUrl) URL.revokeObjectURL(state.previewUrl); });
}

bindEvents();
loadStatus();
