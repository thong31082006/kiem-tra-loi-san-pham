'use strict';
const $ = id => document.getElementById(id);
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
let selectedFile = null, previewUrl = null, busy = false, training = false, ready = false, polling = null;
const SAMPLE_PAGE_SIZE = 12;
const splitNames = {test: 'Test', val: 'Validation', train: 'Train'};
const labelNames = {normal: 'Bình thường', defect: 'Có lỗi'};
const defectNames = {defect_low_fill: 'Thiếu mức nước', defect_no_cap: 'Thiếu nắp',
  defect_loose_cap: 'Nắp lỏng / hỏng', defect_debris: 'Dị vật nhìn thấy', defect_label_damage: 'Nhãn hỏng / lệch / thiếu'};
let libraryOffset = 0, libraryTotal = 0, libraryLoading = false, libraryRequest = 0, sampleButtons = [];
let libraryCategories = [], groupButtons = [], typePolicies = {};
function buttons() {
  $('inspect').disabled = busy || training || !ready || !selectedFile;
  $('train').disabled = busy || training;
  $('image-file').disabled = busy;
  $('open-library').disabled = busy;
  $('library-split').disabled = busy;
  $('library-label').disabled = busy;
  $('reload-library').disabled = busy || libraryLoading;
  $('library-prev').disabled = busy || libraryLoading || libraryOffset === 0;
  $('library-next').disabled = busy || libraryLoading || libraryOffset + SAMPLE_PAGE_SIZE >= libraryTotal;
  for (const button of sampleButtons) button.disabled = busy || libraryLoading;
  for (const button of groupButtons) button.disabled = busy || libraryLoading || button.dataset.empty === 'true';
}
function referenceName(sample) {
  const quality = labelNames[sample.label] || sample.label;
  const types = sample.label === 'defect' ? (sample.defect_types || []).map(code => defectNames[code] || code) : [];
  return types.length ? `${quality} (${types.join(', ')})` : quality;
}
function renderGroups() {
  $('library-groups').replaceChildren(); groupButtons = [];
  const selected = $('library-label').value || 'all';
  for (const category of libraryCategories) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'sample-group';
    button.setAttribute('aria-pressed', String(category.code === selected));
    button.dataset.empty = String(category.count === 0);
    if (category.preview_url) {
      const image = document.createElement('img'); image.src = category.preview_url; image.alt = ''; image.loading = 'lazy';
      button.append(image);
    } else {
      const empty = document.createElement('span'); empty.className = 'group-empty'; empty.textContent = 'Chưa có ảnh'; button.append(empty);
    }
    const name = document.createElement('strong'); name.textContent = category.name;
    const count = document.createElement('span'); count.className = 'muted'; count.textContent = `${category.count} ảnh`;
    button.append(name, count);
    if (typePolicies[category.code]?.enabled === false) {
      const warning = document.createElement('span'); warning.className = 'group-warning';
      warning.textContent = 'SVM chưa đủ tin cậy để gọi tên lỗi này'; button.append(warning);
    }
    button.addEventListener('click', () => {
      if (busy || libraryLoading || !category.count) return;
      $('library-label').value = category.code; libraryOffset = 0; return loadLibrary();
    });
    groupButtons.push(button); $('library-groups').append(button);
  }
  const name = selected === 'all' ? 'Tất cả' : selected === 'defect' ? 'Tất cả ảnh lỗi'
    : selected === 'normal' ? 'Bình thường' : defectNames[selected] || selected;
  $('library-results-heading').textContent = `Danh sách ảnh · ${name}`;
}
function resetResult() {
  document.querySelector('.result').removeAttribute('data-state');
  $('verdict').textContent = 'Chờ kiểm tra';
  $('defect').textContent = 'Bấm Kiểm tra ảnh để xem kết quả.';
  $('result-note').textContent = '';
  $('feedback').textContent = '';
}
function selectFile(file, reference = null) {
  resetResult();
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  $('preview').hidden = true; $('placeholder').hidden = false;
  $('sample-reference').hidden = true;
  $('sample-reference').textContent = '';
  if (file && file.size > 15 * 1024 * 1024) {
    $('feedback').textContent = 'Chọn ảnh không quá 15 MB.';
  } else if (file) {
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    $('preview').src = previewUrl;
    $('preview').hidden = false; $('placeholder').hidden = true;
  }
  $('filename').textContent = selectedFile ? selectedFile.name : 'Chưa chọn ảnh.';
  if (selectedFile && reference) {
    $('sample-reference').textContent = `Nhãn dữ liệu: ${referenceName(reference)} · ${splitNames[reference.split] || reference.split}. Đây là nhãn tham chiếu, chưa phải dự đoán SVM.`;
    $('sample-reference').hidden = false;
  }
  buttons();
  return !!selectedFile;
}
$('image-file').addEventListener('change', () => {
  selectFile($('image-file').files[0]);
});
async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Yêu cầu chưa xử lý được. Hãy thử lại.');
  return data;
}
async function loadLibrary() {
  const ticket = ++libraryRequest;
  libraryLoading = true;
  $('library-feedback').textContent = 'Đang tải thư viện ảnh…';
  $('library-page').textContent = '';
  $('library-grid').replaceChildren(); sampleButtons = [];
  $('library-groups').replaceChildren(); groupButtons = []; libraryCategories = [];
  buttons();
  try {
    const split = encodeURIComponent($('library-split').value || 'test');
    const label = encodeURIComponent($('library-label').value || 'all');
    const data = await request(`/api/samples?split=${split}&label=${label}&limit=${SAMPLE_PAGE_SIZE}&offset=${libraryOffset}`);
    if (ticket !== libraryRequest) return;
    libraryTotal = data.total;
    libraryCategories = data.categories || [];
    renderGroups();
    for (const sample of data.samples) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'sample-card';
      const image = document.createElement('img');
      image.src = sample.image_url; image.alt = ''; image.loading = 'lazy';
      const name = document.createElement('span'); name.className = 'sample-name'; name.textContent = sample.filename;
      const label = document.createElement('span'); label.className = 'sample-label';
      label.textContent = `Nhãn dữ liệu: ${referenceName(sample)} · ${splitNames[sample.split] || sample.split}`;
      button.append(image, name, label);
      button.addEventListener('click', () => chooseSample(sample));
      sampleButtons.push(button); $('library-grid').append(button);
    }
    $('library-feedback').textContent = data.samples.length ? '' : 'Không có ảnh trong bộ lọc này. Bạn có thể chọn bộ lọc khác hoặc chọn ảnh từ máy.';
    $('library-page').textContent = data.samples.length ? `${libraryOffset + 1}–${libraryOffset + data.samples.length} / ${libraryTotal} ảnh` : '0 ảnh';
  } catch (error) {
    if (ticket === libraryRequest) {
      libraryTotal = 0;
      $('library-feedback').textContent = `Chưa tải được thư viện: ${error.message}`;
    }
  } finally {
    if (ticket === libraryRequest) { libraryLoading = false; buttons(); }
  }
}
async function chooseSample(sample) {
  if (busy || libraryLoading) return;
  busy = true; buttons();
  const ticket = ++libraryRequest;
  $('library-feedback').textContent = 'Đang mở ảnh đã chọn…';
  try {
    const response = await fetch(sample.image_url);
    if (!response.ok) throw new Error('Ảnh này hiện không mở được. Hãy tải lại thư viện hoặc chọn ảnh khác.');
    const blob = await response.blob();
    if (ticket !== libraryRequest || !$('library-dialog').open) return;
    const file = new File([blob], sample.filename, {type: blob.type || 'image/jpeg'});
    if (selectFile(file, sample)) {
      $('image-file').value = '';
      $('library-dialog').close();
    } else {
      $('library-feedback').textContent = $('feedback').textContent;
    }
  } catch (error) {
    if (ticket === libraryRequest) $('library-feedback').textContent = error.message;
  } finally { busy = false; buttons(); }
}
$('open-library').addEventListener('click', () => {
  if (busy) return;
  $('library-dialog').showModal();
  return loadLibrary();
});
$('close-library').addEventListener('click', () => $('library-dialog').close());
$('library-dialog').addEventListener('close', () => { libraryRequest++; libraryLoading = false; buttons(); });
for (const id of ['library-split', 'library-label']) {
  $(id).addEventListener('change', () => { libraryOffset = 0; return loadLibrary(); });
}
$('reload-library').addEventListener('click', loadLibrary);
$('library-prev').addEventListener('click', () => {
  if (busy || libraryLoading || libraryOffset === 0) return;
  libraryOffset = Math.max(0, libraryOffset - SAMPLE_PAGE_SIZE); return loadLibrary();
});
$('library-next').addEventListener('click', () => {
  if (busy || libraryLoading || libraryOffset + SAMPLE_PAGE_SIZE >= libraryTotal) return;
  libraryOffset += SAMPLE_PAGE_SIZE; return loadLibrary();
});
$('inspect').addEventListener('click', async () => {
  if (!selectedFile || busy || training || !ready) return;
  busy = true; buttons(); resetResult(); $('verdict').textContent = 'Đang kiểm tra…';
  const form = new FormData(); form.append('file', selectedFile); form.append('model', 'svm');
  try {
    const result = await request('/api/predict', {method: 'POST', body: form});
    document.querySelector('.result').dataset.state = result.label;
    $('verdict').textContent = {normal: 'Đạt', defect: 'Không đạt', uncertain: 'Chưa kết luận'}[result.label] || 'Chưa kết luận';
    const kind = result.defect_analysis || {};
    if (result.label === 'normal') {
      $('defect').textContent = 'Không phát hiện lỗi trong các nhóm đã học.';
    } else if (result.label === 'defect') {
      $('defect').textContent = kind.status === 'typed' ? `Lỗi: ${kind.name}` : 'Phát hiện dấu hiệu lỗi; chưa đủ tin cậy để gọi tên loại lỗi.';
    } else {
      $('defect').textContent = 'Ảnh chưa đủ tin cậy. Hãy chụp rõ toàn bộ chai và kiểm tra lại.';
    }
    $('result-note').textContent = 'Kết quả chỉ áp dụng cho các nhóm lỗi đã học trong phạm vi thử nghiệm.';
  } catch (error) {
    document.querySelector('.result').dataset.state = 'error';
    $('verdict').textContent = 'Chưa kết luận'; $('defect').textContent = error.message;
  } finally { busy = false; buttons(); }
});
function row(cells) {
  const tr = document.createElement('tr');
  for (const value of cells) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
  return tr;
}
async function refresh() {
  try {
    const data = await request('/api/models');
    ready = !!data.ready; training = !!data.training?.running;
    typePolicies = Object.fromEntries((data.defect_types?.classes || []).map(item => [item.code, item]));
    if ($('library-dialog').open && !libraryLoading) renderGroups();
    $('model-state').textContent = training ? data.training.message : ready ? 'HOG + SVM đã sẵn sàng.' : (data.error || 'Chưa có mô hình. Bấm Huấn luyện và test lại SVM.');
    if (data.training?.state === 'failed') $('model-state').textContent = data.training.message;
    const d = data.dataset || {};
    $('split-counts').textContent = `Train: ${d.train ?? '—'} ảnh · Validation: ${d.val ?? '—'} ảnh · Test: ${d.test ?? '—'} ảnh.`;
    const metrics = $('metrics'); metrics.replaceChildren();
    for (const [label, scores] of [['Vùng chai, trước bước từ chối', data.report?.models?.svm?.test], ['Toàn luồng, gồm chưa kết luận', data.end_to_end?.models?.svm]]) {
      if (scores) metrics.append(row([label, ...['accuracy', 'precision', 'recall', 'f1'].map(k => percent(scores[k]))]));
    }
    const pipeline = data.end_to_end?.models?.svm;
    $('coverage').textContent = pipeline ? `Test toàn luồng: đúng ${pipeline.correct_count}/${pipeline.total} ảnh; ${pipeline.uncertain_count} ảnh chưa kết luận; tỷ lệ có kết luận ${percent(pipeline.coverage)}.` : 'Chưa có kết quả Test toàn luồng.';
    const typeRows = $('type-metrics'); typeRows.replaceChildren();
    for (const item of data.defect_types?.classes || []) {
      const scores = data.defect_types?.test?.accepted?.per_class?.[item.code] || {};
      typeRows.append(row([item.name, item.enabled ? 'Được bật theo Validation' : 'Chưa đủ tin cậy', percent(scores.precision), percent(scores.recall)]));
    }
    if (polling) clearTimeout(polling);
    if (training) polling = setTimeout(refresh, 3000);
    $('train').textContent = training ? 'Đang huấn luyện và test…' : 'Huấn luyện và test lại SVM';
  } catch (error) {
    $('model-state').textContent = error.message; ready = false;
    if (training) polling = setTimeout(refresh, 5000);
  }
  buttons();
}
$('train').addEventListener('click', async () => {
  if (busy || training) return;
  training = true; buttons();
  try { await request('/api/train', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'}); await refresh(); }
  catch (error) { training = false; $('model-state').textContent = error.message; buttons(); }
});
$('model-details').addEventListener('toggle', () => { if ($('model-details').open) refresh(); });
refresh();
