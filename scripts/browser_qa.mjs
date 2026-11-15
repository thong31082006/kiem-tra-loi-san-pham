// Use the installed Chrome and Node's built-in DevTools WebSocket; no browser download.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const workspace = path.dirname(appRoot);
const artifactDir = path.join(workspace, 'artifacts');
const base = process.env.BOTTLE_QA_URL || 'http://127.0.0.1:8080';
const chrome = [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)'], process.env.LOCALAPPDATA]
  .filter(Boolean).map(root => path.join(root, 'Google/Chrome/Application/chrome.exe')).find(file => fs.existsSync(file));
assert(chrome, 'Chrome is required for this local browser check');
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'bottle-browser-qa-'));
const browser = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { windowsHide: true, stdio: 'ignore' });
const errors = [];
const pending = new Map();
let socket, nextId = 0;
let createdHistory = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function until(test, description, timeout = 20000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    const result = await test();
    if (result) return result;
    await sleep(150);
  }
  throw new Error(`Timed out: ${description}`);
}

function command(method, params = {}) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`DevTools timeout: ${method}`)); }, 30000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const response = await command('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}

async function click(id) {
  return evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
}

async function screenshot(name) {
  const result = await command('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  fs.mkdirSync(artifactDir, { recursive: true });
  fs.writeFileSync(path.join(artifactDir, name), Buffer.from(result.data, 'base64'));
}

try {
  const activePort = path.join(profile, 'DevToolsActivePort');
  await until(() => fs.existsSync(activePort), 'Chrome starts');
  const port = fs.readFileSync(activePort, 'utf8').split('\n')[0];
  const pages = await fetch(`http://127.0.0.1:${port}/json/list`).then(response => response.json());
  const page = pages.find(item => item.type === 'page');
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const task = pending.get(message.id);
      clearTimeout(task.timer); pending.delete(message.id);
      if (message.error) task.reject(new Error(JSON.stringify(message.error))); else task.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
  });
  await command('Page.enable');
  await command('Runtime.enable');
  await command('Emulation.setDeviceMetricsOverride', { width: 1360, height: 1000, deviceScaleFactor: 1, mobile: false });
  await command('Page.navigate', { url: base });
  await until(() => evaluate(`document.readyState === 'complete' && document.getElementById('connection-state').textContent === 'Mô hình sẵn sàng'`), 'trained models connect');
  assert.equal(await evaluate(`document.querySelectorAll('#model-select option').length`), 3);
  assert.equal(await evaluate(`document.body.textContent.includes('Ảnh chuẩn')`), false);

  await click('tab-models');
  await until(() => evaluate(`document.querySelectorAll('#models-body tr').length === 3 && !document.getElementById('models-body').textContent.includes('—')`), 'actual model metrics');
  await screenshot('web_mo_hinh.png');

  await click('tab-samples');
  await evaluate(`document.getElementById('sample-label').value='defect';document.getElementById('sample-label').dispatchEvent(new Event('change'));`);
  await until(() => evaluate(`document.querySelectorAll('.sample-item').length > 0 && document.getElementById('samples-feedback').textContent === ''`), 'defect sample filter');
  assert.equal(await evaluate(`[...document.querySelectorAll('.sample-item-label')].every(node => node.textContent.includes('Có lỗi'))`), true);
  await evaluate(`document.querySelector('.sample-item').click()`);
  await until(() => evaluate(`state.currentFile && document.getElementById('image-preview').naturalWidth > 0 && !document.getElementById('predict-button').disabled`), 'sample becomes uploaded image');
  await screenshot('web_anh_da_chon.png');

  for (const model of ['mobilenet', 'svm', 'random_forest']) {
    await evaluate(`document.getElementById('model-select').value=${JSON.stringify(model)};document.getElementById('model-select').dispatchEvent(new Event('change'));`);
    await click('predict-button');
    await until(() => evaluate(`!state.predicting && (state.prediction !== null || document.getElementById('action-feedback').dataset.kind==='error')`), `real ${model} prediction`, 45000);
    const prediction = await evaluate(`state.prediction`);
    assert(prediction, await evaluate(`document.getElementById('action-feedback').textContent`));
    assert.equal(prediction.model, model);
    assert(['normal', 'defect', 'uncertain'].includes(prediction.label));
    assert(Number.isFinite(prediction.confidence));
    assert(prediction.localization?.bbox?.length === 4, 'real inference locates the bottle');
    createdHistory.push(prediction.id);
    if (model === 'mobilenet') await screenshot('web_kiem_tra_chai_nhua.png');
  }

  await click('tab-history');
  await until(() => evaluate(`document.querySelectorAll('#history-body img').length >= 3`), 'saved history appears');
  await screenshot('web_lich_su.png');
  await click('tab-inspect');
  await click('clear-image');
  assert.equal(await evaluate(`document.getElementById('result-content').hidden && document.getElementById('predict-button').disabled`), true);

  await command('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await sleep(300);
  assert.equal(await evaluate(`document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1`), true, 'mobile page does not overflow horizontally');
  await screenshot('web_dien_thoai.png');
  assert.equal(errors.length, 0, `Browser JavaScript exceptions: ${JSON.stringify(errors)}`);
  const result = { status: 'pass', models: 3, real_predictions: 3, sample_filter: true,
    one_image_input: true, saved_history: true, mobile_width: 390, js_errors: errors,
    note: 'Camera hardware was not accessed. Classification quality is measured separately in the evaluation report.' };
  fs.writeFileSync(path.join(artifactDir, 'browser_qa.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
} finally {
  for (const id of createdHistory) {
    try { await fetch(`${base}/api/history/${id}`, { method: 'DELETE' }); } catch {}
  }
  if (socket?.readyState === WebSocket.OPEN) {
    try { await command('Browser.close'); } catch {}
    socket.close();
  }
  if (browser.exitCode === null) browser.kill();
  // Only remove the unique temporary Chrome profile created above.
  const resolved = path.resolve(profile);
  if (resolved.startsWith(path.join(path.resolve(os.tmpdir()), 'bottle-browser-qa-'))) {
    for (let retry = 0; retry < 6; retry++) {
      try { fs.rmSync(resolved, { recursive: true, force: true }); break; } catch { await sleep(300); }
    }
  }
}
