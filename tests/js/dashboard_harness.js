/**
 * Minimal, dependency-free harness that runs the real dashboard script (vecta/ui/static/js/app.js)
 * against a tiny fake DOM built from index.html and scripted API responses, then prints a JSON
 * snapshot of what the user would see. Used by tests/test_dashboard_js.py.
 *
 * Usage: node dashboard_harness.js <scenario>
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..', '..');
const HTML = fs.readFileSync(path.join(ROOT, 'vecta', 'ui', 'static', 'index.html'), 'utf8');
const APP_JS = fs.readFileSync(path.join(ROOT, 'vecta', 'ui', 'static', 'js', 'app.js'), 'utf8');

// ---------------------------------------------------------------- fake DOM
class ClassList {
  constructor(el) { this.el = el; }
  _get() { return new Set(String(this.el.className).split(/\s+/).filter(Boolean)); }
  _set(s) { this.el.className = [...s].join(' '); }
  add(...c) { const s = this._get(); c.forEach(x => s.add(x)); this._set(s); }
  remove(...c) { const s = this._get(); c.forEach(x => s.delete(x)); this._set(s); }
  toggle(c, force) {
    const s = this._get();
    const on = force === undefined ? !s.has(c) : !!force;
    if (on) s.add(c); else s.delete(c);
    this._set(s);
    return on;
  }
  contains(c) { return this._get().has(c); }
}

class El {
  constructor(tag, attrs = {}) {
    this.tagName = tag.toUpperCase();
    this.id = attrs.id || '';
    this.className = attrs.class || '';
    this.dataset = {};
    this.style = {};
    this.children = [];
    this._text = '';
    this._html = '';
    this.hidden = 'hidden' in attrs;
    this.disabled = false;
    this.value = attrs.value || '';
    this.title = '';
    this.listeners = {};
    this.classList = new ClassList(this);
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set innerHTML(v) { this._html = String(v); this._text = String(v).replace(/<[^>]*>/g, ''); this.children = []; }
  get innerHTML() { return this._html; }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  click() { (this.listeners.click || []).forEach(fn => fn({ target: this })); }
  getContext() { return {}; }
}

function parseAttrs(src) {
  const attrs = {};
  const re = /([\w-]+)(?:="([^"]*)")?/g;
  let m;
  while ((m = re.exec(src))) attrs[m[1]] = m[2] === undefined ? '' : m[2];
  return attrs;
}

function buildDocument() {
  const byId = {};
  const presets = [];
  const tagRe = /<(\w+)(\s[^>]*)>/g;
  let m;
  while ((m = tagRe.exec(HTML))) {
    const attrs = parseAttrs(m[2]);
    const el = new El(m[1], attrs);
    if (attrs['data-preset']) el.dataset.preset = attrs['data-preset'];
    if (attrs.id) byId[attrs.id] = el;
    if ((attrs.class || '').split(/\s+/).includes('preset-btn')) presets.push(el);
  }
  const ready = [];
  const document = {
    getElementById: id => byId[id] || null,
    querySelectorAll: sel => (sel === '.preset-btn' ? presets : []),
    createElement: tag => new El(tag),
    addEventListener: (type, fn) => { if (type === 'DOMContentLoaded') ready.push(fn); }
  };
  return { document, byId, ready };
}

// ---------------------------------------------------------------- API fixtures
const WINNER = 'HNSW(M=16, efC=64, efS=32)';
const profileOk = (warnings = []) => ({
  status: 200,
  body: {
    workload_profile: { intrinsic_dimension: 20.5, dimension: 128, clustering_tendency_hopkins: 0.65, norm_variance: 0.0 },
    warnings, estimated_peak_memory_mb: 40.0, vector_sample_2d: []
  }
});
const tuneOk = (memTarget = 128, memPred = 9.6) => ({
  status: 200,
  body: {
    recommended_config: { id_string: WINNER, family: 'HNSW' },
    recommended_prediction: {
      predicted_recall: 0.95, recall_uncertainty: 0.01, predicted_latency_p95: 0.08,
      latency_uncertainty_p95: 0.02, predicted_memory_mb: memPred, predicted_build_time_s: 0.5
    },
    rationale: 'Vecta selected ' + WINNER,
    all_points: [{ id: WINNER, family: 'HNSW', recall: 0.95, latency_p95: 0.08, memory_mb: memPred, is_pareto: true, is_feasible: true, is_winner: true }],
    total_candidates_evaluated: 281, pareto_optimal_count: 50, tuning_duration_ms: 3000,
    sla_satisfied: memPred <= memTarget, feasible_candidate_count: 10, violated_constraints: memPred <= memTarget ? [] : ['memory_mb'],
    constraint_report: [
      { constraint: 'recall', predicted: 0.95, target: 0.9, comparison: '>=', met: true },
      { constraint: 'latency_p95_ms', predicted: 0.08, target: 1.5, comparison: '<=', met: true },
      { constraint: 'memory_mb', predicted: memPred, target: memTarget, comparison: '<=', met: memPred <= memTarget }
    ],
    deployed_config: { id_string: WINNER }, config_adjustments: []
  }
});
const row = (id, recall) => ({
  config: { id_string: id }, recall_at_k: recall, latency_p95_ms: 0.1, latency_p99_ms: 0.12,
  throughput_qps: 9000, memory_footprint_mb: 7.5, index_serialized_mb: 7.3, index_build_time_s: 0.05
});
const benchOk = () => ({
  status: 200,
  body: {
    summary_verdict: 'VERDICT-TEXT', sla_check: { all_met: true },
    results: {
      ground_truth_flat: row('Flat-Exact', 1.0), faiss_default_ivf: row('IVF', 0.25),
      faiss_default_hnsw: row('HNSW-default', 0.55), vecta_auto_tuned: row(WINNER, 0.95)
    }
  }
});
const err = (status, detail) => ({ status, body: { detail } });
const system = () => ({ status: 200, body: { hardware: { logical_cpus: 8, available_ram_gb: 4.0 }, engine: { name: 'FAISS', version: 'x', compile_options: '' } } });

// ---------------------------------------------------------------- runner
async function flush(rounds = 40) {
  for (let i = 0; i < rounds; i++) await new Promise(r => setImmediate(r));
}

async function main() {
  const scenario = process.argv[2];
  const { document, byId, ready } = buildDocument();
  const routes = { '/api/system': [system()] };
  const set = (url, ...responses) => { routes[url] = responses; };
  const calls = [];

  const fetch = async (url, opts) => {
    calls.push(url);
    const queue = routes[url];
    if (!queue || !queue.length) throw new Error('no scripted response for ' + url);
    const res = queue.length > 1 ? queue.shift() : queue[0];
    return {
      ok: res.status >= 200 && res.status < 300, status: res.status, statusText: 'x',
      json: async () => res.body
    };
  };
  class Chart { constructor() {} destroy() {} }
  const ctx = vm.createContext({
    document, window: {}, fetch, Chart, performance: { now: () => Date.now() },
    setInterval: () => 0, console: { log() {}, warn() {}, error() {} }
  });

  // Initial automatic run on page load: success unless the scenario says otherwise
  set('/api/profile', profileOk(scenario === 'warnings' ? ['num_vectors=50000 exceeds the interactive limit; profiled and tuned a synthetic dataset of 30000 vectors instead.'] : []));
  set('/api/tune', scenario === 'memory_label' ? tuneOk(50, 60) : tuneOk());
  set('/api/benchmark', benchOk());

  vm.runInContext(APP_JS, ctx);
  ready.forEach(fn => fn());
  await flush();

  if (scenario === 'profile_fails_after_success') {
    set('/api/profile', err(400, 'Invalid request: dimension: Input should be less than or equal to 4096'));
    byId.triggerAutoTuneBtn.click();
    await flush();
  } else if (scenario === 'tune_fails_after_profile') {
    set('/api/profile', profileOk());
    set('/api/tune', err(500, 'Index rebuild failed (RuntimeError: boom); state unchanged: no index is currently active for this dataset.'));
    byId.triggerAutoTuneBtn.click();
    await flush();
  }

  const text = id => byId[id].textContent;
  console.log(JSON.stringify({
    calls,
    bannerHidden: byId.statusBanner.hidden,
    bannerClass: byId.statusBanner.className,
    bannerText: text('statusBannerText'),
    winnerTitle: text('winnerConfigTitle'),
    verdict: text('verdictText'),
    arenaRows: byId.arenaTableBody.children.length,
    arenaFirstCell: byId.arenaTableBody.children.length ? byId.arenaTableBody.children[0].children[0].textContent : null,
    memoryCellTitle: byId.arenaTableBody.children.length > 1 ? byId.arenaTableBody.children[1].children[5].title : null,
    searchDisabled: byId.executeSearchBtn.disabled,
    benchmarkDisabled: byId.runArenaBtn.disabled,
    runDisabled: byId.triggerAutoTuneBtn.disabled,
    memorySub: text('winnerMemorySub'),
    memorySubClass: byId.winnerMemorySub.className
  }));
}

main().catch(e => { console.error(e && e.stack || e); process.exit(1); });
