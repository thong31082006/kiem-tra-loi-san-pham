// Exercise result behavior without a browser or the user's actual history DB.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const {File} = require('node:buffer');

function setup(prediction, library = {}) {
  const elements = new Map();
  const element = () => ({
    textContent: '', value: '', disabled: false, hidden: false, open: false, dataset: {}, attributes: {}, files: [], handlers: {}, children: [],
    addEventListener(event, fn) {this.handlers[event] = fn;},
    replaceChildren(...nodes) {this.children = nodes;}, append(...nodes) {this.children.push(...nodes);},
    removeAttribute() {this.dataset = {};},
    setAttribute(name, value) {this.attributes[name] = value;},
    showModal() {this.open = true;}, close() {this.open = false; this.handlers.close?.();},
  });
  const get = id => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  const requests = [];
  get('preview').hidden = true;
  get('sample-reference').hidden = true;
  get('library-split').value = 'test';
  get('library-label').value = 'all';
  const context = {
    document: {getElementById: get, querySelector: () => get('result'), createElement: element},
    URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}},
    FormData: class {constructor() {this.values = [];} append(...value) {this.values.push(value);}},
    File,
    setTimeout: () => 1, clearTimeout() {},
    fetch: async (url, options) => {
      requests.push({url, options});
      if (url.startsWith('/api/samples?')) {
        const data = library.load ? await library.load(url) : {samples: library.samples || [], total: library.samples?.length || 0};
        return {ok: true, json: async () => ({categories: library.categories || [], ...data})};
      }
      if (url.startsWith('/api/samples/')) return library.image || {ok: true, blob: async () => new Blob(['pixels'], {type: 'image/jpeg'})};
      return {ok: true, json: async () => url === '/api/models'
        ? {ready: true, training: {running: false}, dataset: {train: 729, val: 230, test: 231}, defect_types: {classes: library.policies || []}}
        : prediction};
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'web/svm.js'), 'utf8'), context);
  return {get, requests};
}
const settle = () => new Promise(resolve => setImmediate(resolve));

for (const [label, type, verdict, text] of [
  ['normal', {status: 'normal'}, 'Đạt', 'Không phát hiện lỗi'],
  ['defect', {status: 'typed', name: 'Thiếu nắp'}, 'Không đạt', 'Thiếu nắp'],
  ['defect', {status: 'uncertain', name: 'Nhãn hỏng'}, 'Không đạt', 'chưa đủ tin cậy để gọi tên'],
  ['uncertain', {status: 'typed', name: 'Thiếu nắp'}, 'Chưa kết luận', 'chụp rõ toàn bộ chai'],
]) {
  test(`quality ${label}, type ${type.status}`, async () => {
    const ui = setup({label, defect_analysis: type});
    await settle();
    ui.get('image-file').files = [{name: 'anh.png', size: 1000}];
    ui.get('image-file').handlers.change();
    assert.equal(ui.get('inspect').disabled, false);
    await ui.get('inspect').handlers.click();
    assert.equal(ui.get('verdict').textContent, verdict);
    assert.ok(ui.get('defect').textContent.includes(text));
    const submitted = ui.requests.find(r => r.url === '/api/predict').options.body.values;
    assert.equal(submitted.find(([key]) => key === 'model')[1], 'svm');
    // Changing the selected file must clear the old verdict immediately.
    ui.get('image-file').handlers.change();
    assert.equal(ui.get('verdict').textContent, 'Chờ kiểm tra');
  });
}

test('oversized file cannot reuse an old prediction or submit', async () => {
  const ui = setup({label: 'normal'});
  await settle();
  ui.get('image-file').files = [{name: 'too-big.png', size: 16 * 1024 * 1024}];
  ui.get('image-file').handlers.change();
  await ui.get('inspect').handlers.click();
  assert.equal(ui.get('inspect').disabled, true);
  assert.ok(ui.get('feedback').textContent.includes('15 MB'));
  assert.equal(ui.requests.filter(r => r.url === '/api/predict').length, 0);
});

const sample = (id = 'test-1', split = 'test') => ({id, split, label: 'defect', defect_types: ['defect_no_cap'], filename: `${id}.jpg`, image_url: `/api/samples/${id}/image`});

test('library selection uses pixels with SVM and keeps reference label separate', async () => {
  const ui = setup({label: 'normal', defect_analysis: {status: 'normal'}}, {samples: [sample()]});
  await settle();
  await ui.get('open-library').handlers.click();
  assert.equal(ui.get('library-dialog').open, true);
  assert.ok(ui.requests.some(r => r.url.includes('split=test&label=all&limit=12&offset=0')));
  assert.equal(ui.get('library-grid').children.length, 1);
  await ui.get('library-grid').children[0].handlers.click();
  assert.equal(ui.get('library-dialog').open, false);
  assert.equal(ui.get('filename').textContent, 'test-1.jpg');
  assert.ok(ui.get('sample-reference').textContent.includes('Có lỗi'));
  assert.ok(ui.get('sample-reference').textContent.includes('Thiếu nắp'));
  assert.equal(ui.get('verdict').textContent, 'Chờ kiểm tra');
  assert.equal(ui.requests.filter(r => r.url === '/api/predict').length, 0);
  await ui.get('inspect').handlers.click();
  assert.equal(ui.get('verdict').textContent, 'Đạt'); // Reference truth cannot replace SVM prediction.
  const submitted = ui.requests.find(r => r.url === '/api/predict').options.body.values;
  assert.deepEqual(Array.from(submitted, ([key]) => key), ['file', 'model']);
  assert.equal(submitted[0][1].name, 'test-1.jpg');
  ui.get('image-file').files = [{name: 'local.png', size: 1000}];
  ui.get('image-file').handlers.change();
  assert.equal(ui.get('sample-reference').hidden, true);
  assert.equal(ui.get('verdict').textContent, 'Chờ kiểm tra');
});

test('library supports pagination and resets page when filters change', async () => {
  const samples = Array.from({length: 13}, (_, i) => sample(`test-${i}`));
  const ui = setup({}, {load: async url => {
    const offset = Number(new URL(url, 'http://test').searchParams.get('offset'));
    return {total: 13, samples: samples.slice(offset, offset + 12)};
  }});
  await settle(); await ui.get('open-library').handlers.click();
  assert.equal(ui.get('library-next').disabled, false);
  await ui.get('library-next').handlers.click();
  assert.equal(ui.get('library-grid').children.length, 1);
  assert.equal(ui.get('library-next').disabled, true);
  assert.equal(ui.get('library-prev').disabled, false);
  ui.get('library-label').value = 'normal';
  await ui.get('library-label').handlers.change();
  assert.ok(ui.requests.at(-1).url.includes('label=normal&limit=12&offset=0'));
  assert.equal(ui.get('library-prev').disabled, true);
});

test('late library response cannot overwrite newer filter results', async () => {
  let completeOldRequest;
  const oldRequest = new Promise(resolve => {completeOldRequest = resolve;});
  const ui = setup({}, {load: url => url.includes('split=test') ? oldRequest
    : Promise.resolve({samples: [sample('train-1', 'train')], total: 1})});
  await settle();
  const firstLoad = ui.get('open-library').handlers.click();
  ui.get('library-split').value = 'train';
  await ui.get('library-split').handlers.change();
  completeOldRequest({samples: [sample('test-stale')], total: 1});
  await firstLoad;
  assert.equal(ui.get('library-grid').children[0].children[1].textContent, 'train-1.jpg');
});

test('empty library and missing sample image leave selection safe to retry', async () => {
  const library = {samples: []};
  const ui = setup({}, library);
  await settle(); await ui.get('open-library').handlers.click();
  assert.ok(ui.get('library-feedback').textContent.includes('Không có ảnh'));
  assert.equal(ui.get('library-next').disabled, true);
  library.samples = [sample()]; library.image = {ok: false};
  await ui.get('reload-library').handlers.click();
  await ui.get('library-grid').children[0].handlers.click();
  assert.equal(ui.get('library-dialog').open, true);
  assert.equal(ui.get('inspect').disabled, true);
  assert.ok(ui.get('library-feedback').textContent.includes('không mở được'));
});

test('closing library cancels a pending sample selection', async () => {
  let finishImage;
  const image = new Promise(resolve => {finishImage = resolve;});
  const ui = setup({}, {samples: [sample()], image: {ok: true, blob: () => image}});
  await settle(); await ui.get('open-library').handlers.click();
  const selecting = ui.get('library-grid').children[0].handlers.click();
  ui.get('close-library').handlers.click();
  finishImage(new Blob(['pixels'])); await selecting;
  assert.equal(ui.get('inspect').disabled, true);
  assert.equal(ui.get('preview').hidden, true); // No selection was committed after close.
  assert.equal(ui.requests.filter(r => r.url === '/api/predict').length, 0);
});

test('normal and five fault groups have examples, counts and a selectable type filter', async () => {
  const codes = ['normal', 'defect_low_fill', 'defect_no_cap', 'defect_loose_cap', 'defect_debris', 'defect_label_damage'];
  const categories = codes.map((code, i) => ({code, name: code, count: i + 1, preview_url: `/api/samples/${code}/image`}));
  const ui = setup({}, {samples: [sample()], categories, policies: [{code: 'defect_label_damage', enabled: false}]});
  await settle(); await ui.get('open-library').handlers.click();
  const groups = ui.get('library-groups').children;
  assert.equal(groups.length, 6);
  for (let i = 0; i < groups.length; i++) {
    assert.equal(groups[i].children[0].src, categories[i].preview_url);
    assert.equal(groups[i].children[2].textContent, `${i + 1} ảnh`);
  }
  assert.ok(groups[5].children[3].textContent.includes('SVM chưa đủ tin cậy'));
  await groups[2].handlers.click();
  assert.ok(ui.requests.at(-1).url.includes('label=defect_no_cap&limit=12&offset=0'));
  assert.equal(ui.get('library-groups').children[2].attributes['aria-pressed'], 'true');
  assert.ok(ui.get('library-results-heading').textContent.includes('Thiếu nắp'));
  assert.equal(ui.requests.filter(r => r.url === '/api/predict').length, 0);
});

test('a group with no source photos shows an empty state and cannot select a cover', async () => {
  const ui = setup({}, {categories: [{code: 'defect_label_damage', name: 'Nhãn hỏng', count: 0, preview_url: null}]});
  await settle(); await ui.get('open-library').handlers.click();
  const group = ui.get('library-groups').children[0];
  assert.equal(group.disabled, true);
  assert.equal(group.children[0].textContent, 'Chưa có ảnh');
  const before = ui.requests.length;
  await group.handlers.click();
  assert.equal(ui.requests.length, before);
});
