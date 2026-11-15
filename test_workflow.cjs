"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "web/workflow.js"), "utf8");

function fixture(request) {
  const controls = new Map();
  const state = { batch: { id: "a".repeat(32), running: false, stopRequested: false, model: null,
    entries: [0, 1, 2].map((position) => ({ position, name: `image${position}.jpg`, file: { name: `image${position}.jpg`, type: "image/jpeg", size: 200 }, status: "pending", result: null, error: "" })) },
    activeTab: "history", presentation: false, presentationReturnTab: "inspect" };
  class FakeFormData {
    constructor() { this.items = new Map(); }
    append(key, value) { this.items.set(key, value); }
    get(key) { return this.items.get(key); }
  }
  const classes = new Set();
  const context = vm.createContext({ state, FormData: FakeFormData,
    labelNames: { normal: "Normal", defect: "Defect", uncertain: "Uncertain" },
    request: (...args) => request(...args),
    byId: (id) => {
      if (!controls.has(id)) controls.set(id, { disabled: false, value: "svm", hidden: true, focus() {}, setAttribute(key, value) { this[key] = value; } });
      return controls.get(id);
    }, updatePredictButton() {}, renderTraining() {}, feedback() {},
    activateTab(name) { state.activeTab = name; },
    document: { body: { classList: { toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); } } } },
    window: { scrollTo() {} },
  });
  vm.runInContext(source, context);
  context.renderBatch = () => {};
  return { context, state, controls, classes };
}

test("pause finishes current image; resume skips completed images and pins model/positions", async () => {
  const calls = [];
  let setup;
  setup = fixture(async (_url, options) => {
    const position = Number(options.body.get("position"));
    calls.push({ position, model: options.body.get("model"), id: options.body.get("batch_id") });
    if (calls.length === 1) setup.state.batch.stopRequested = true;
    return { id: `saved-${position}`, label: "normal" };
  });
  await setup.context.runBatch();
  assert.equal(setup.state.batch.running, false);
  assert.deepEqual(setup.state.batch.entries.map((entry) => entry.status), ["done", "pending", "pending"]);
  setup.controls.get("batch-model").value = "mobilenet";
  await setup.context.runBatch();
  assert.deepEqual(calls.map((call) => call.position), [0, 1, 2]);
  assert.ok(calls.every((call) => call.model === "svm" && call.id === "a".repeat(32)));
  assert.ok(setup.state.batch.entries.every((entry) => entry.status === "done"));
});

test("invalid image does not block other images; retry only sends failed positions", async () => {
  const calls = [];
  let fail = true;
  const setup = fixture(async (_url, options) => {
    const position = Number(options.body.get("position")); calls.push(position);
    if (position === 1 && fail) { const error = new Error("Invalid image"); error.status = 422; throw error; }
    return { id: `saved-${position}`, label: "normal" };
  });
  await setup.context.runBatch();
  assert.deepEqual(setup.state.batch.entries.map((entry) => entry.status), ["done", "error", "done"]);
  fail = false;
  setup.context.retryBatchErrors();
  await setup.context.runBatch();
  assert.deepEqual(calls, [0, 1, 2, 1]);
  assert.ok(setup.state.batch.entries.every((entry) => entry.status === "done"));
});

test("network interruption pauses the queue instead of failing every remaining image", async () => {
  let calls = 0;
  const setup = fixture(async () => { calls++; const error = new Error("Offline"); error.network = true; throw error; });
  await setup.context.runBatch();
  assert.equal(calls, 1);
  assert.deepEqual(setup.state.batch.entries.map((entry) => entry.status), ["error", "pending", "pending"]);
  assert.equal(setup.state.batch.running, false);
});

test("presentation focuses inspection, restores prior tab, and cannot hide a running batch", () => {
  const setup = fixture(() => {});
  setup.context.setPresentation(true);
  assert.equal(setup.state.activeTab, "inspect");
  assert.equal(setup.controls.get("presentation-bar").hidden, false);
  assert.ok(setup.classes.has("presentation-mode"));
  setup.context.setPresentation(false);
  assert.equal(setup.state.activeTab, "history");
  assert.ok(!setup.classes.has("presentation-mode"));
  setup.state.batch.running = true;
  setup.context.setPresentation(true);
  assert.equal(setup.state.presentation, false);
});

test("new workflow script loads before app bindings; all literal DOM ids exist", () => {
  const html = fs.readFileSync(path.join(__dirname, "web/index.html"), "utf8");
  const app = fs.readFileSync(path.join(__dirname, "web/app.js"), "utf8");
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  for (const match of (source + app).matchAll(/byId\("([^"]+)"\)/g)) assert.ok(ids.includes(match[1]), `Missing DOM id ${match[1]}`);
  assert.ok(html.indexOf('src="/static/workflow.js') < html.indexOf('src="/static/app.js'));
});
