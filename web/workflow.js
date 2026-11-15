"use strict";

const reviewVerdicts = { correct: "Đúng", incorrect: "Sai", unsure: "Chưa rõ" };

function reviewSummary(item) {
  const reviews = item?.reviews || {};
  return Object.entries(reviews).map(([target, review]) => `${target === "quality" ? "Chất lượng" : "Loại lỗi"}: ${reviewVerdicts[review.verdict] || "Chưa rõ"}`).join(" · ") || "Chưa có đánh giá.";
}

function batchCounts(entries) {
  const counts = { total: entries.length, done: 0, error: 0, pending: 0, running: 0, normal: 0, defect: 0, uncertain: 0 };
  for (const item of entries) {
    if (Object.hasOwn(counts, item.status)) counts[item.status] += 1;
    if (item.status === "done" && Object.hasOwn(labelNames, item.result?.label)) counts[item.result.label] += 1;
  }
  return counts;
}

function batchFileError(file) {
  if (!/\.(jpe?g|png|webp|bmp)$/i.test(file.name) && !file.type.startsWith("image/")) return "Định dạng tệp không được hỗ trợ.";
  if (!file.size || file.size > 15 * 1024 * 1024) return "Ảnh cần có dung lượng từ 1 byte đến 15 MB.";
  return "";
}

function setBatchFiles(files) {
  if (state.batch.running) return;
  if (!files.length) return;
  if (files.length > 50 || files.reduce((total, file) => total + file.size, 0) > 150 * 1024 * 1024) {
    feedback("batch-feedback", "Chọn tối đa 50 ảnh, tổng dung lượng không quá 150 MB.", "error");
    return;
  }
  for (const item of state.batch.entries) if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  const id = [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  state.batch = { id, running: false, stopRequested: false, model: null, entries: files.map((file, position) => {
    const error = batchFileError(file);
    return { file, position, name: file.name, status: error ? "error" : "pending", error, result: null, previewUrl: null };
  }) };
  feedback("batch-feedback", "Đã chọn lô ảnh. Chọn mô hình rồi bắt đầu kiểm tra.");
  renderBatch();
}

function updateBatchControls() {
  const batch = state.batch;
  const counts = batchCounts(batch.entries);
  const model = state.models.find((item) => item.id === (batch.model || byId("batch-model").value));
  const blocked = state.predicting || trainingRunning() || !model?.ready;
  byId("start-batch").disabled = batch.running || blocked || !counts.pending;
  byId("start-batch").textContent = batch.model && counts.pending ? "Tiếp tục kiểm tra" : "Bắt đầu kiểm tra";
  byId("pause-batch").disabled = !batch.running || batch.stopRequested;
  byId("pause-batch").textContent = batch.stopRequested && batch.running ? "Đang đợi ảnh hiện tại..." : "Tạm dừng";
  byId("choose-batch").disabled = batch.running;
  byId("batch-model").disabled = batch.running || Boolean(batch.model);
  byId("retry-batch").disabled = batch.running || state.predicting || !batch.entries.some((entry) => entry.status === "error" && !batchFileError(entry.file));
  byId("export-batch").disabled = batch.running || !counts.done && !counts.error;
  byId("presentation-toggle").disabled = batch.running;
}

function renderBatch() {
  const batch = state.batch;
  const counts = batchCounts(batch.entries);
  const processed = counts.done + counts.error;
  byId("batch-progress").max = Math.max(1, counts.total);
  byId("batch-progress").value = processed;
  byId("batch-progress-text").textContent = counts.total ? `${processed}/${counts.total} ảnh đã xử lý · ${counts.pending} đang chờ${batch.running ? " · Đang chạy" : counts.pending && batch.model ? " · Đã tạm dừng" : ""}` : "Chưa chọn lô ảnh.";
  const summary = byId("batch-summary");
  summary.replaceChildren();
  for (const [key, name] of [["normal", "Bình thường"], ["defect", "Có dấu hiệu lỗi"], ["uncertain", "Chưa kết luận"], ["error", "Không xử lý được"]]) {
    const card = element("div", null, "batch-stat");
    card.dataset.label = key;
    card.append(element("strong", counts[key]), element("span", name));
    summary.append(card);
  }
  const body = byId("batch-body");
  body.replaceChildren();
  if (!counts.total) {
    const cell = element("td", "Chọn ảnh để tạo lô kiểm tra.", "empty-cell"); cell.colSpan = 7;
    const row = element("tr"); row.append(cell); body.append(row);
  }
  for (const entry of batch.entries) {
    const row = element("tr");
    row.dataset.position = entry.position;
    const preview = element("td");
    if (entry.result?.image_url) {
      const image = element("img", null, "history-thumbnail");
      image.src = entry.result.image_url; image.alt = `Ảnh ${entry.name}`; image.loading = "lazy";
      preview.append(image);
    } else preview.append(element("span", entry.position + 1, "batch-position"));
    row.append(preview, element("td", entry.name, "batch-filename"));
    const status = element("td", entry.status === "done" ? labelNames[entry.result.label] : ({ pending: "Đang chờ", running: "Đang kiểm tra...", error: "Không xử lý được" })[entry.status], "batch-result");
    status.dataset.label = entry.result?.label || entry.status;
    if (entry.error) status.append(element("small", entry.error, "batch-row-note"));
    if (entry.result?.reviews) status.append(element("small", reviewSummary(entry.result), "batch-row-note"));
    const analysis = entry.result?.defect_analysis;
    const typeText = analysis?.status === "typed" ? analysis.name + (analysis.agreement === "review" ? " · Cần đối chiếu" : "") : entry.result ? "Chưa gọi tên lỗi cụ thể" : "—";
    row.append(status, element("td", typeText), element("td", entry.result ? percentage(entry.result.confidence) : "—"), element("td", entry.result ? `${Math.round(entry.result.latency_ms)} ms` : "—"));
    const actions = element("td", null, "batch-actions");
    if (entry.result) {
      const view = element("button", "Chi tiết", "text-button"); view.type = "button";
      view.addEventListener("click", () => openBatchItem(entry));
      const review = element("button", "Đánh giá", "text-button"); review.type = "button";
      review.addEventListener("click", () => openReview(entry.result));
      actions.append(view, review);
    }
    row.append(actions); body.append(row);
  }
  updateBatchControls();
}

async function runBatch() {
  if (byId("start-batch").disabled) return;
  const batch = state.batch;
  batch.model ||= byId("batch-model").value;
  batch.running = true; batch.stopRequested = false;
  updatePredictButton(); renderTraining();
  feedback("batch-feedback", "Đang xử lý lần lượt các ảnh. Có thể tạm dừng sau ảnh hiện tại.");
  try {
    for (const entry of batch.entries) {
      if (batch.stopRequested) break;
      if (entry.status !== "pending") continue;
      entry.status = "running"; entry.error = ""; renderBatch();
      const form = new FormData();
      form.append("file", entry.file, entry.name); form.append("model", batch.model);
      form.append("batch_id", batch.id); form.append("position", String(entry.position));
      try {
        entry.result = await request("/api/batch/predict", { method: "POST", body: form });
        entry.status = "done";
      } catch (error) {
        entry.status = "error"; entry.error = error.message;
        if (error.network || error.status === 409) batch.stopRequested = true;
      }
      renderBatch();
    }
  } finally {
    batch.running = false;
    renderBatch(); updatePredictButton(); renderTraining();
    const counts = batchCounts(batch.entries);
    feedback("batch-feedback", counts.pending ? `Đã tạm dừng. Còn ${counts.pending} ảnh đang chờ; có thể tiếp tục hoặc thử lại ảnh lỗi.` : `Hoàn tất: ${counts.done} ảnh có kết quả, ${counts.error} ảnh không xử lý được.`, counts.error ? "" : "success");
  }
}

function retryBatchErrors() {
  if (state.batch.running) return;
  for (const entry of state.batch.entries) {
    if (entry.status === "error") {
      entry.error = batchFileError(entry.file);
      if (!entry.error) entry.status = "pending";
    }
  }
  feedback("batch-feedback", "Ảnh lỗi hợp lệ đã chuyển về hàng chờ. Bấm Tiếp tục để thử lại; kết quả đã lưu không bị chạy lại.");
  renderBatch();
}

async function openBatchItem(entry) {
  if (!entry.result || !setFile(entry.file)) return;
  const version = state.fileVersion;
  byId("model-select").value = entry.result.model;
  renderPrediction(entry.result);
  activateTab("inspect");
  await renderAnalysis(entry.result.analysis);
  if (version !== state.fileVersion) return;
  updatePredictButton();
  feedback("action-feedback", "Đang xem kết quả đã lưu của ảnh trong lô.");
}

function downloadWorkflowCsv(rows, filename) {
  const csv = rows.map((row) => row.map((value) => {
    let text = String(value ?? "");
    if (/^[=+\-@]/.test(text.trimStart())) text = "'" + text;
    return `"${text.replaceAll('"', '""')}"`;
  }).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" }));
  const link = element("a"); link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportBatch() {
  const batch = state.batch;
  const rows = [["Mã lô", "Thứ tự", "Tên ảnh", "Mô hình", "Phiên mô hình", "Kết quả", "Điểm mô hình (0..1)", "Thời gian (ms)", "Loại lỗi AI bổ sung", "Đối chiếu hai bộ phân tích", "Đánh giá người dùng", "Lỗi xử lý", "Nhãn chất lượng sửa (người dùng)", "Ghi chú chất lượng", "Loại lỗi sửa (người dùng)", "Ghi chú loại lỗi"]];
  for (const entry of batch.entries) {
    const item = entry.result;
    rows.push([batch.id, entry.position + 1, entry.name, item?.model_name || friendlyModel(batch.model || byId("batch-model").value), item?.run_id || "", item ? labelNames[item.label] : entry.status === "error" ? "Không xử lý được" : "Chưa xử lý", item?.confidence, item?.latency_ms, item?.defect_analysis?.name, item?.defect_analysis?.agreement, item ? reviewSummary(item) : "", entry.error, item?.reviews?.quality?.corrected_label, item?.reviews?.quality?.note, item?.reviews?.defect_type?.corrected_type, item?.reviews?.defect_type?.note]);
  }
  downloadWorkflowCsv(rows, "ket_qua_lo_anh.csv");
}

function typeReviewNames() {
  return { normal: "Không thấy lỗi", ...defectNames, defect_dent: "Móp / biến dạng chai (AI chưa hỗ trợ)", defect_cloudy: "Nước đục / đổi màu (AI chưa hỗ trợ)", other: "Khác (ghi rõ trong ghi chú)", unknown: "Chưa xác định" };
}

function openReview(item) {
  if (!item?.id || state.reviewSaving) return;
  state.reviewItem = item; state.reviewDrafts = {}; state.reviewTarget = "quality";
  byId("review-filename").textContent = item.filename || "Ảnh kiểm tra";
  byId("review-image").src = item.image_url;
  byId("review-context-text").textContent = `${item.model_name || friendlyModel(item.model)} · ${formatDate(item.created_at)}`;
  byId("review-target").value = "quality";
  byId("review-target").querySelector('option[value="defect_type"]').disabled = !item.defect_analysis || item.defect_analysis.status === "unavailable";
  feedback("review-feedback", "");
  populateReviewTarget();
  byId("review-dialog").showModal();
}

function readReviewDraft() {
  return { verdict: document.querySelector('input[name="review-verdict"]:checked')?.value || "",
           correction: byId("review-corrected").value, note: byId("review-note").value };
}

function populateReviewTarget() {
  const target = byId("review-target").value;
  const item = state.reviewItem;
  const saved = item.reviews?.[target];
  const draft = state.reviewDrafts[target] || { verdict: saved?.verdict || "", correction: saved?.corrected_label || saved?.corrected_type || "", note: saved?.note || "" };
  byId("review-current-prediction").textContent = `AI đã trả về: ${target === "quality" ? labelNames[item.label] : item.defect_analysis?.name || "Chưa xác định"}`;
  for (const radio of document.querySelectorAll('input[name="review-verdict"]')) radio.checked = radio.value === draft.verdict;
  const select = byId("review-corrected"); select.replaceChildren();
  const options = { "": "Chưa biết / Không ghi nhãn sửa", ...(target === "quality" ? labelNames : typeReviewNames()) };
  for (const [key, value] of Object.entries(options)) {
    const option = element("option", value); option.value = key; select.append(option);
  }
  select.value = draft.correction;
  byId("review-note").value = draft.note;
  byId("review-saved-at").textContent = saved ? `Lần lưu gần nhất: ${formatDate(saved.updated_at)}` : "Chưa có đánh giá cho nội dung này.";
  updateReviewCorrection();
}

function updateReviewCorrection() {
  byId("review-correction").hidden = document.querySelector('input[name="review-verdict"]:checked')?.value !== "incorrect";
}

async function saveReview(event) {
  event.preventDefault();
  if (!state.reviewItem || state.reviewSaving) return;
  const item = state.reviewItem;
  const target = byId("review-target").value;
  const draft = readReviewDraft();
  if (!draft.verdict) { feedback("review-feedback", "Chọn Đúng, Sai hoặc Chưa rõ trước khi lưu.", "error"); return; }
  const body = { target, verdict: draft.verdict, note: draft.note };
  if (draft.verdict === "incorrect" && draft.correction) body[target === "quality" ? "corrected_label" : "corrected_type"] = draft.correction;
  state.reviewSaving = true; byId("save-review").disabled = true; byId("close-review").disabled = true;
  byId("review-target").disabled = true;
  byId("review-note").disabled = true; byId("review-corrected").disabled = true;
  for (const radio of document.querySelectorAll('input[name="review-verdict"]')) radio.disabled = true;
  feedback("review-feedback", "Đang lưu đánh giá...");
  try {
    const review = await request(`/api/history/${encodeURIComponent(item.id)}/review`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const sources = [item, state.prediction, ...state.comparison, ...state.batch.entries.map((entry) => entry.result)].filter(Boolean);
    for (const source of sources) if (source.id === item.id) source.reviews = { ...(source.reviews || {}), [target]: review };
    state.reviewDrafts[target] = draft;
    byId("review-saved-at").textContent = `Đã lưu: ${formatDate(review.updated_at)}`;
    if (state.prediction?.id === item.id) byId("result-review-summary").textContent = reviewSummary(state.prediction);
    if (state.batch.entries.length) renderBatch();
    if (state.activeTab === "history") await loadHistory();
    feedback("review-feedback", "Đã lưu phản hồi. Dự đoán AI gốc và dữ liệu huấn luyện được giữ nguyên.", "success");
  } catch (error) {
    feedback("review-feedback", error.message, "error");
  } finally {
    state.reviewSaving = false; byId("save-review").disabled = false; byId("close-review").disabled = false; byId("review-target").disabled = false;
    byId("review-note").disabled = false; byId("review-corrected").disabled = false;
    for (const radio of document.querySelectorAll('input[name="review-verdict"]')) radio.disabled = false;
  }
}

function setPresentation(enabled) {
  if (enabled && state.batch.running) return;
  if (enabled && !state.presentation) state.presentationReturnTab = state.activeTab;
  state.presentation = enabled;
  document.body.classList.toggle("presentation-mode", enabled);
  byId("presentation-bar").hidden = !enabled;
  byId("presentation-toggle").setAttribute("aria-pressed", String(enabled));
  if (enabled) {
    activateTab("inspect");
    byId("presentation-exit").focus();
    window.scrollTo({ top: 0, behavior: "smooth" });
  } else {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    activateTab(state.presentationReturnTab || "inspect");
    byId("presentation-toggle").focus();
  }
}

async function togglePresentationFullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
    else throw new Error("unsupported");
    feedback("presentation-feedback", "");
  } catch {
    feedback("presentation-feedback", "Trình duyệt chưa cho phép toàn màn hình. Chế độ trình chiếu phóng lớn vẫn sử dụng được.");
  }
}

function bindWorkflowEvents() {
  byId("choose-batch").addEventListener("click", () => byId("batch-files").click());
  byId("batch-files").addEventListener("change", (event) => { setBatchFiles([...event.target.files]); event.target.value = ""; });
  byId("batch-model").addEventListener("change", updateBatchControls);
  byId("start-batch").addEventListener("click", runBatch);
  byId("pause-batch").addEventListener("click", () => { state.batch.stopRequested = true; updateBatchControls(); });
  byId("retry-batch").addEventListener("click", retryBatchErrors);
  byId("export-batch").addEventListener("click", exportBatch);
  byId("review-result").addEventListener("click", () => openReview(state.prediction));
  byId("review-form").addEventListener("submit", saveReview);
  byId("review-target").addEventListener("change", () => {
    state.reviewDrafts[state.reviewTarget] = readReviewDraft();
    state.reviewTarget = byId("review-target").value;
    populateReviewTarget(); feedback("review-feedback", "");
  });
  for (const radio of document.querySelectorAll('input[name="review-verdict"]')) radio.addEventListener("change", updateReviewCorrection);
  byId("close-review").addEventListener("click", () => byId("review-dialog").close());
  byId("review-dialog").addEventListener("cancel", (event) => { if (state.reviewSaving) event.preventDefault(); });
  byId("history-review-filter").addEventListener("change", loadHistory);
  byId("presentation-toggle").addEventListener("click", () => setPresentation(!state.presentation));
  byId("presentation-exit").addEventListener("click", () => setPresentation(false));
  byId("presentation-fullscreen").addEventListener("click", togglePresentationFullscreen);
  document.addEventListener("fullscreenchange", () => { byId("presentation-fullscreen").textContent = document.fullscreenElement ? "Thoát toàn màn hình" : "Toàn màn hình"; });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.presentation && !document.querySelector("dialog[open]")) setPresentation(false);
  });
  window.addEventListener("pagehide", () => { for (const entry of state.batch.entries) if (entry.previewUrl) URL.revokeObjectURL(entry.previewUrl); });
  renderBatch();
}
