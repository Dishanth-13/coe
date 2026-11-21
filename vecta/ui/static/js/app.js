/**
 * VECTA Vector Search Optimization Platform - Interactive Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    workload: {
      num_vectors: 15000,
      dimension: 128,
      metric: "L2",
      target_recall: 0.95,
      max_latency_ms: 1.5,
      memory_budget_mb: 128.0,
      expected_qps: 600.0,
      target_k: 10
    },
    busy: false,
    hasActiveIndex: false,
    deploymentId: null,          // server deployment this page's results describe
    staleDeploymentNotified: false,
    paretoChart: null,
    arenaBarChart: null
  };

  // DOM Elements
  const el = {
    sliderVectors: document.getElementById('sliderVectors'),
    valVectors: document.getElementById('valVectors'),
    sliderDim: document.getElementById('sliderDim'),
    valDim: document.getElementById('valDim'),
    sliderRecall: document.getElementById('sliderRecall'),
    valRecall: document.getElementById('valRecall'),
    sliderLatency: document.getElementById('sliderLatency'),
    valLatency: document.getElementById('valLatency'),
    sliderMemory: document.getElementById('sliderMemory'),
    valMemory: document.getElementById('valMemory'),
    sliderK: document.getElementById('sliderK'),
    valK: document.getElementById('valK'),

    presetButtons: document.querySelectorAll('.preset-btn'),
    triggerAutoTuneBtn: document.getElementById('triggerAutoTuneBtn'),
    deployWinnerBtn: document.getElementById('deployWinnerBtn'),
    runArenaBtn: document.getElementById('runArenaBtn'),
    executeSearchBtn: document.getElementById('executeSearchBtn'),

    // Status banner
    statusBanner: document.getElementById('statusBanner'),
    statusBannerText: document.getElementById('statusBannerText'),
    statusBannerClose: document.getElementById('statusBannerClose'),

    // Telemetry & Metrics
    telemetryEngine: document.getElementById('telemetryEngine'),
    telemetryCpu: document.getElementById('telemetryCpu'),
    telemetryRam: document.getElementById('telemetryRam'),
    mIntrinsicDim: document.getElementById('mIntrinsicDim'),
    mIntrinsicSub: document.getElementById('mIntrinsicSub'),
    mHopkins: document.getElementById('mHopkins'),
    mNormVar: document.getElementById('mNormVar'),
    mProfTime: document.getElementById('mProfTime'),

    // Recommendation Elements
    candidateCountBadge: document.getElementById('candidateCountBadge'),
    paretoCountBadge: document.getElementById('paretoCountBadge'),
    tuningDurationBadge: document.getElementById('tuningDurationBadge'),
    paretoEmpty: document.getElementById('paretoEmpty'),
    winnerConfigTitle: document.getElementById('winnerConfigTitle'),
    winnerFamilyTag: document.getElementById('winnerFamilyTag'),
    winnerSlaStatus: document.getElementById('winnerSlaStatus'),
    recommendationDesc: document.getElementById('recommendationDesc'),
    winnerRecall: document.getElementById('winnerRecall'),
    winnerRecallCi: document.getElementById('winnerRecallCi'),
    winnerLatency: document.getElementById('winnerLatency'),
    winnerLatencyCi: document.getElementById('winnerLatencyCi'),
    winnerMemory: document.getElementById('winnerMemory'),
    winnerMemorySub: document.getElementById('winnerMemorySub'),
    winnerBuildTime: document.getElementById('winnerBuildTime'),
    winnerRationaleText: document.getElementById('winnerRationaleText'),

    // Arena Table
    arenaVerdictBanner: document.getElementById('arenaVerdictBanner'),
    verdictText: document.getElementById('verdictText'),
    arenaTableBody: document.getElementById('arenaTableBody'),
    arenaEmpty: document.getElementById('arenaEmpty'),

    // Search Playground
    qLatencyVal: document.getElementById('qLatencyVal'),
    qGtMatchVal: document.getElementById('qGtMatchVal'),
    qTotalServed: document.getElementById('qTotalServed'),
    neighborsList: document.getElementById('neighborsList')
  };

  const RUN_BTN_HTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run Optimization';

  // Initialize System Telemetry & Event Bindings
  initTelemetry();
  bindEvents();

  // Run initial profile and tuning
  runWorkloadProfileAndTune();

  // Polling Telemetry every 5 seconds
  setInterval(initTelemetry, 5000);

  // -------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------

  /** fetch() wrapper: throws an Error carrying the server's `detail` on non-2xx responses. */
  async function apiFetch(url, options) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (err) {
      throw new Error(`Cannot reach the Vecta server (${url}): ${err.message}`);
    }
    let body = null;
    try {
      body = await res.json();
    } catch (_) {
      body = null;
    }
    if (!res.ok) {
      let detail = res.statusText;
      if (body && body.detail !== undefined) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      }
      throw new Error(`${url} failed (HTTP ${res.status}): ${detail}`);
    }
    if (body === null) {
      throw new Error(`${url} returned an empty or non-JSON response.`);
    }
    return body;
  }

  function showStatus(message, kind) {
    el.statusBanner.hidden = false;
    el.statusBanner.className = `status-banner ${kind || ''}`;
    el.statusBannerText.textContent = message;
  }

  function clearStatus() {
    el.statusBanner.hidden = true;
    el.statusBannerText.textContent = '';
  }

  function setBusy(busy) {
    state.busy = busy;
    el.triggerAutoTuneBtn.disabled = busy;
    el.runArenaBtn.disabled = busy || !state.hasActiveIndex;
    el.deployWinnerBtn.disabled = busy || !state.hasActiveIndex;
    el.executeSearchBtn.disabled = busy || !state.hasActiveIndex;
    el.presetButtons.forEach(b => { b.disabled = busy; });
  }

  function fmt(value, digits, suffix) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(digits) + (suffix || '');
  }

  function setChartOverlay(overlay, message, isError) {
    if (message) {
      overlay.hidden = false;
      overlay.textContent = message;
      overlay.classList.toggle('error', !!isError);
    } else {
      overlay.hidden = true;
    }
  }

  function setVerdict(message, kind) {
    el.arenaVerdictBanner.className = `arena-verdict-banner ${kind || ''}`;
    el.verdictText.textContent = message;
  }

  function renderTableMessage(message, isError) {
    el.arenaTableBody.innerHTML = '';
    const tr = document.createElement('tr');
    tr.classList.add('empty-row');
    const td = document.createElement('td');
    td.colSpan = 7;
    td.textContent = message;
    if (isError) td.style.color = 'var(--accent-red)';
    tr.appendChild(td);
    el.arenaTableBody.appendChild(tr);
  }

  function familyLabel(family) {
    switch (family) {
      case 'HNSW': return 'Graph-based index (HNSW)';
      case 'IVF_FLAT': return 'Inverted file index (IVF-Flat)';
      case 'IVF_PQ': return 'Inverted file + product quantization (IVF-PQ)';
      case 'FLAT': return 'Exact brute-force index';
      default: return family || '—';
    }
  }

  // -------------------------------------------------------------
  // Event Bindings
  // -------------------------------------------------------------
  function bindEvents() {
    // Slider Sync
    el.sliderVectors.addEventListener('input', (e) => {
      state.workload.num_vectors = parseInt(e.target.value, 10);
      el.valVectors.textContent = state.workload.num_vectors.toLocaleString();
    });
    el.sliderDim.addEventListener('input', (e) => {
      state.workload.dimension = parseInt(e.target.value, 10);
      el.valDim.textContent = state.workload.dimension;
    });
    el.sliderRecall.addEventListener('input', (e) => {
      state.workload.target_recall = parseFloat(e.target.value);
      el.valRecall.textContent = (state.workload.target_recall * 100).toFixed(1) + '%';
    });
    el.sliderLatency.addEventListener('input', (e) => {
      state.workload.max_latency_ms = parseFloat(e.target.value);
      el.valLatency.textContent = state.workload.max_latency_ms.toFixed(2) + ' ms';
    });
    el.sliderMemory.addEventListener('input', (e) => {
      state.workload.memory_budget_mb = parseFloat(e.target.value);
      el.valMemory.textContent = state.workload.memory_budget_mb.toFixed(0) + ' MB';
    });
    el.sliderK.addEventListener('input', (e) => {
      state.workload.target_k = parseInt(e.target.value, 10);
      el.valK.textContent = state.workload.target_k;
    });

    // Preset Buttons
    el.presetButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        if (state.busy) return;
        el.presetButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        applyPreset(btn.dataset.preset);
      });
    });

    // Action Triggers
    el.triggerAutoTuneBtn.addEventListener('click', () => runWorkloadProfileAndTune());
    el.deployWinnerBtn.addEventListener('click', () => runArenaBenchmark());
    el.runArenaBtn.addEventListener('click', () => runArenaBenchmark());
    el.executeSearchBtn.addEventListener('click', () => executeLiveSearch());
    el.statusBannerClose.addEventListener('click', () => clearStatus());
  }

  // -------------------------------------------------------------
  // Preset Configuration
  // -------------------------------------------------------------
  function applyPreset(presetKey) {
    const presets = {
      rag_semantic_search: { vectors: 20000, dim: 128, recall: 0.98, latency: 2.0, memory: 180, k: 10 },
      recommendation_high_qps: { vectors: 25000, dim: 64, recall: 0.90, latency: 0.6, memory: 90, k: 20 },
      edge_memory_constrained: { vectors: 15000, dim: 128, recall: 0.88, latency: 2.5, memory: 35, k: 10 },
      sift_benchmark: { vectors: 15000, dim: 128, recall: 0.95, latency: 1.2, memory: 140, k: 10 }
    };
    const p = presets[presetKey] || presets.rag_semantic_search;

    state.workload.num_vectors = p.vectors;
    state.workload.dimension = p.dim;
    state.workload.target_recall = p.recall;
    state.workload.max_latency_ms = p.latency;
    state.workload.memory_budget_mb = p.memory;
    state.workload.target_k = p.k;

    el.sliderVectors.value = p.vectors;
    el.valVectors.textContent = p.vectors.toLocaleString();
    el.sliderDim.value = p.dim;
    el.valDim.textContent = p.dim;
    el.sliderRecall.value = p.recall;
    el.valRecall.textContent = (p.recall * 100).toFixed(1) + '%';
    el.sliderLatency.value = p.latency;
    el.valLatency.textContent = p.latency.toFixed(2) + ' ms';
    el.sliderMemory.value = p.memory;
    el.valMemory.textContent = p.memory + ' MB';
    el.sliderK.value = p.k;
    el.valK.textContent = p.k;

    runWorkloadProfileAndTune();
  }

  // -------------------------------------------------------------
  // Telemetry & Profiling
  // -------------------------------------------------------------
  async function initTelemetry() {
    // Snapshot what this page believes is deployed; ignore polls that overlap our own tuning run
    const knownDeployment = state.deploymentId;
    const busyAtStart = state.busy;
    try {
      const data = await apiFetch('/api/system');
      if (!busyAtStart && !state.busy && knownDeployment !== null && knownDeployment === state.deploymentId) {
        checkDeployment(data.active_index);
      }
      if (data.hardware) {
        el.telemetryCpu.textContent = `${data.hardware.logical_cpus} Threads`;
        el.telemetryRam.textContent = `${data.hardware.available_ram_gb} GB`;
      }
      if (data.engine) {
        const opts = data.engine.compile_options ? ` ${data.engine.compile_options}` : '';
        el.telemetryEngine.textContent = `${data.engine.name} ${data.engine.version}${opts}`;
      }
      [el.telemetryEngine, el.telemetryCpu, el.telemetryRam].forEach(n => n.classList.remove('offline'));
    } catch (err) {
      console.warn('Telemetry polling error:', err);
      [el.telemetryEngine, el.telemetryCpu, el.telemetryRam].forEach(n => {
        n.textContent = 'offline';
        n.classList.add('offline');
      });
    }
  }

  /**
   * The server keeps ONE shared deployment. If another session (another tab, or a direct API call)
   * re-tunes or re-profiles, the recommendation and results shown here no longer describe the
   * serving index: say so, and disable actions that would silently use the other deployment.
   */
  function checkDeployment(activeIndex) {
    const liveId = activeIndex ? activeIndex.deployment_id : null;
    if (liveId === state.deploymentId || state.staleDeploymentNotified) return;
    state.staleDeploymentNotified = true;
    state.hasActiveIndex = false;
    setBusy(false);
    const now = activeIndex ? `it now serves ${activeIndex.config.id_string}` : 'no index is deployed now';
    showStatus(`Another session changed the server's deployment (${now}). The recommendation and benchmark ` +
               'results on this page are out of date. Click Run Optimization to refresh.', 'warning');
  }

  async function runWorkloadProfileAndTune() {
    if (state.busy) return;
    setBusy(true);
    el.triggerAutoTuneBtn.innerHTML = 'Optimizing...';

    // The server only replaces its dataset (and drops the old index) when /api/profile succeeds.
    // Until then the previous recommendation, benchmark and index remain valid and untouched.
    let profileCommitted = false;
    let warnings = [];

    try {
      // 1. Profile Workload
      showStatus('Profiling workload…', 'loading');
      const t0 = performance.now();
      const profData = await apiFetch('/api/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.workload)
      });
      profileCommitted = true;
      state.hasActiveIndex = false;  // the server invalidated the previous index
      setBusy(true);
      const wp = profData.workload_profile;
      updateProfileUI(wp, performance.now() - t0);
      warnings = profData.warnings || [];

      // 2. Run Auto-Tuning (also deploys the winner as the active index)
      showStatus('Running auto-tuning optimizer…', 'loading');
      setTuningPending();
      const tuneData = await apiFetch('/api/tune', { method: 'POST' });
      updateTuningUI(tuneData);
      state.hasActiveIndex = true;
      state.deploymentId = tuneData.deployment_id !== undefined ? tuneData.deployment_id : null;
      state.staleDeploymentNotified = false;

      // 3. Run Benchmark automatically
      showStatus('Benchmarking deployed configuration against FAISS defaults…', 'loading');
      await benchmarkInner();
      if (warnings.length) {
        // Keep server warnings (e.g. dataset size capped) visible after the run completes
        showStatus(`Warning: ${warnings.join(' ')}`, 'warning');
      } else {
        clearStatus();
      }
    } catch (err) {
      console.error('Tuning error:', err);
      const prefix = warnings.length ? `Warning: ${warnings.join(' ')} ` : '';
      if (!profileCommitted) {
        // Profile was rejected: the server kept its previous dataset and index, so keep the UI as is
        const msg = /[.!?]$/.test(err.message) ? err.message : `${err.message}.`;
        showStatus(`${msg} The previous workload, recommendation and index are unchanged.`, 'error');
      } else {
        showStatus(prefix + err.message, 'error');
        if (!state.hasActiveIndex) {
          setTuningError(err.message);
        }
      }
    } finally {
      el.triggerAutoTuneBtn.innerHTML = RUN_BTN_HTML;
      setBusy(false);
    }
  }

  function updateProfileUI(wp, elapsedMs) {
    el.mIntrinsicDim.textContent = fmt(wp.intrinsic_dimension, 2);
    el.mIntrinsicSub.textContent = wp.intrinsic_dimension > 0
      ? `Ambient / intrinsic: ${(wp.dimension / wp.intrinsic_dimension).toFixed(1)}x`
      : 'Two-NN estimate';
    el.mHopkins.textContent = fmt(wp.clustering_tendency_hopkins, 3);
    el.mNormVar.textContent = fmt(wp.norm_variance, 4);
    el.mProfTime.textContent = fmt(elapsedMs, 1, ' ms');
  }

  function setTuningPending() {
    el.tuningDurationBadge.textContent = 'Tuning…';
    setChartOverlay(el.paretoEmpty, 'Evaluating candidate configurations…');
  }

  function setTuningError(message) {
    el.tuningDurationBadge.textContent = 'Tuning failed';
    el.winnerSlaStatus.hidden = true;
    el.winnerConfigTitle.textContent = 'No recommendation';
    el.winnerFamilyTag.textContent = '—';
    [el.winnerRecall, el.winnerLatency, el.winnerMemory, el.winnerBuildTime].forEach(n => { n.textContent = '—'; });
    [el.winnerRecallCi, el.winnerLatencyCi, el.winnerMemorySub].forEach(n => { n.textContent = ''; });
    el.winnerRationaleText.textContent = message;
    setChartOverlay(el.paretoEmpty, 'No candidates to display.', true);
    setVerdict('Benchmark not run: no active configuration.', 'error');
    renderTableMessage('No benchmark results.', false);
    setChartOverlay(el.arenaEmpty, 'No benchmark results.');
  }

  function updateTuningUI(tuneData) {
    el.candidateCountBadge.textContent = `${tuneData.total_candidates_evaluated} Configurations`;
    el.paretoCountBadge.textContent = `${tuneData.pareto_optimal_count} Pareto Optimal`;
    el.tuningDurationBadge.textContent = `Tuned in ${fmt(tuneData.tuning_duration_ms, 0)} ms`;

    const winner = tuneData.recommended_config;
    const pred = tuneData.recommended_prediction;

    el.winnerConfigTitle.textContent = winner.id_string;
    el.winnerFamilyTag.textContent = familyLabel(winner.family);

    // When the shortlist was benchmarked, the decision rests on MEASURED values; predictions stay visible
    const measured = tuneData.selection_basis === 'measured' ? tuneData.recommended_measured : null;
    const basis = measured ? 'Measured' : 'Predicted';
    if (el.recommendationDesc) {
      el.recommendationDesc.textContent = measured
        ? `Shortlisted from ${tuneData.total_candidates_evaluated} cost-model predictions; chosen from ${tuneData.benchmarked_count} real FAISS benchmarks on this dataset.`
        : 'Configuration chosen by the optimizer from cost-model predictions (not benchmarked).';
    }

    el.winnerSlaStatus.hidden = false;
    if (tuneData.sla_satisfied) {
      el.winnerSlaStatus.className = 'tag-pill winner-sla green';
      el.winnerSlaStatus.textContent = measured
        ? `Measured: meets all targets (${tuneData.feasible_candidate_count} of ${tuneData.benchmarked_count} benchmarked do)`
        : `Predicted to meet all targets (${tuneData.feasible_candidate_count} feasible candidates)`;
    } else {
      const missed = (tuneData.violated_constraints || []).join(', ') || 'one or more targets';
      el.winnerSlaStatus.className = 'tag-pill winner-sla amber';
      el.winnerSlaStatus.textContent = `${basis}: no config meets all targets — best-effort choice (misses: ${missed})`;
    }

    el.winnerRecall.textContent = fmt(pred.predicted_recall * 100, 1, '%');
    el.winnerLatency.textContent = fmt(pred.predicted_latency_p95, 2, ' ms');
    if (measured) {
      el.winnerRecallCi.textContent = `Measured: ${fmt(measured.recall_at_k * 100, 1, '%')}`;
      el.winnerLatencyCi.textContent = `Measured: ${fmt(measured.latency_p95_ms, 3, ' ms')}`;
    } else if (pred.recall_uncertainty !== null && pred.recall_uncertainty !== undefined) {
      el.winnerRecallCi.textContent = `± ${fmt(pred.recall_uncertainty * 100, 1)}% (tree std)`;
      el.winnerLatencyCi.textContent = `± ${fmt(pred.latency_uncertainty_p95, 2)} ms (tree std)`;
    } else {
      el.winnerRecallCi.textContent = 'Not benchmarked';
      el.winnerLatencyCi.textContent = 'Not benchmarked';
    }

    el.winnerMemory.textContent = fmt(pred.predicted_memory_mb, 1, ' MB');
    // Use the budget the server actually tuned against (sliders may have moved since)
    const memRow = (tuneData.constraint_report || []).find(r => r.constraint === 'memory_mb');
    const budget = memRow ? memRow.target : state.workload.memory_budget_mb;
    const withinBudget = memRow ? memRow.met : pred.predicted_memory_mb <= budget;
    el.winnerMemorySub.textContent = measured
      ? `Measured index ${fmt(measured.serialized_index_mb, 1, ' MB')} ${withinBudget ? 'within' : 'exceeds'} ${budget.toFixed(0)} MB budget`
      : (withinBudget ? `Within ${budget.toFixed(0)} MB budget (predicted)` : `Exceeds ${budget.toFixed(0)} MB budget (predicted)`);
    el.winnerMemorySub.classList.toggle('warn', !withinBudget);

    el.winnerBuildTime.textContent = fmt(pred.predicted_build_time_s, 2, ' s');

    let rationale = tuneData.rationale;
    if (tuneData.config_adjustments && tuneData.config_adjustments.length) {
      const deployed = tuneData.deployed_config ? tuneData.deployed_config.id_string : '';
      rationale += ` Deployed as ${deployed} after FAISS compatibility adjustments: ${tuneData.config_adjustments.join('; ')}.`;
    }
    el.winnerRationaleText.textContent = rationale;

    // Render Pareto Frontier Scatter Plot
    renderParetoChart(tuneData.all_points || [], winner.id_string);
  }

  // -------------------------------------------------------------
  // Empirical Arena Benchmarking
  // -------------------------------------------------------------
  async function runArenaBenchmark() {
    if (state.busy) return;
    setBusy(true);
    showStatus('Benchmarking deployed configuration against FAISS defaults…', 'loading');
    try {
      await benchmarkInner();
      clearStatus();
    } catch (err) {
      console.error('Benchmark error:', err);
      showStatus(err.message, 'error');
    } finally {
      setBusy(false);
    }
  }

  /** Runs /api/benchmark and renders results. Throws on failure (after rendering an error state). */
  async function benchmarkInner() {
    el.runArenaBtn.textContent = 'Benchmarking...';
    el.deployWinnerBtn.textContent = 'Benchmarking Active Index...';
    setVerdict('Benchmark running…', 'neutral');
    renderTableMessage('Benchmark running…', false);
    setChartOverlay(el.arenaEmpty, 'Benchmark running…');

    try {
      const data = await apiFetch('/api/benchmark', { method: 'POST' });

      const allMet = data.sla_check ? data.sla_check.all_met : null;
      setVerdict(data.summary_verdict, allMet === false ? 'warning' : 'neutral');
      renderArenaTable(data.results);
      renderArenaBarChart(data.results);
    } catch (err) {
      setVerdict(`Benchmark failed: ${err.message}`, 'error');
      renderTableMessage('Benchmark failed — no results.', true);
      setChartOverlay(el.arenaEmpty, 'Benchmark failed.', true);
      throw err;
    } finally {
      el.runArenaBtn.textContent = 'Re-run Benchmark';
      el.deployWinnerBtn.textContent = 'Benchmark Deployed Configuration';
    }
  }

  function renderArenaTable(results) {
    const tbody = el.arenaTableBody;
    tbody.innerHTML = '';

    const items = [
      { key: 'ground_truth_flat', label: 'Exact Brute-Force (Flat)', data: results.ground_truth_flat, isWinner: false },
      { key: 'faiss_default_ivf', label: 'FAISS Default IVF (nlist=100, nprobe=1)', data: results.faiss_default_ivf, isWinner: false },
      { key: 'faiss_default_hnsw', label: 'FAISS Default HNSW (M=32, efS=16)', data: results.faiss_default_hnsw, isWinner: false },
      { key: 'vecta_auto_tuned', label: `VECTA Auto-Tuned (${results.vecta_auto_tuned.config.id_string})`, data: results.vecta_auto_tuned, isWinner: true }
    ];

    items.forEach(item => {
      if (!item.data) return;
      const tr = document.createElement('tr');
      if (item.isWinner) tr.classList.add('winner-row');

      const d = item.data;
      const rec = fmt(d.recall_at_k * 100, 1, '%');
      const p95 = fmt(d.latency_p95_ms, 3, ' ms');
      const p99 = fmt(d.latency_p99_ms, 3, ' ms');
      const qps = (d.throughput_qps !== undefined && d.throughput_qps !== null)
        ? Math.round(d.throughput_qps).toLocaleString() : '—';
      const mem = fmt(d.memory_footprint_mb, 1, ' MB');
      const bld = fmt(d.index_build_time_s, 3, ' s');

      const recColor = d.recall_at_k > 0.9 ? '#10b981' : (d.recall_at_k > 0.5 ? '#f59e0b' : '#ef4444');

      const cells = [item.label, rec, p95, p99, qps, mem, bld];
      cells.forEach((text, i) => {
        const td = document.createElement('td');
        if (i === 5) {
          // Estimated, not measured: show the measured serialized size alongside on hover
          td.title = `Estimated index memory (not measured RAM). Measured serialized index size: ${fmt(d.index_serialized_mb, 2, ' MB')}`;
        }
        if (i === 1) {
          const span = document.createElement('span');
          span.style.color = recColor;
          span.style.fontWeight = '600';
          span.textContent = text;
          td.appendChild(span);
        } else {
          td.textContent = text;
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  // -------------------------------------------------------------
  // Live Query Testing
  // -------------------------------------------------------------
  async function executeLiveSearch() {
    if (state.busy) return;
    setBusy(true);
    el.executeSearchBtn.textContent = 'Executing Query...';

    try {
      const data = await apiFetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ k: state.workload.target_k, use_random_query: true })
      });

      el.qLatencyVal.textContent = fmt(data.query_latency_ms, 3, ' ms');
      if (data.exact_matches_in_top_k === null || data.exact_matches_in_top_k === undefined) {
        el.qGtMatchVal.textContent = 'n/a';
      } else {
        el.qGtMatchVal.textContent = `${data.exact_matches_in_top_k} / ${data.k !== undefined ? data.k : state.workload.target_k} (${data.accuracy_pct}%)`;
      }
      el.qTotalServed.textContent = data.total_queries_served.toLocaleString();

      // Render Neighbors
      el.neighborsList.innerHTML = '';
      const dists = data.distances[0];
      const idxs = data.indices[0];
      const scoreLabel = data.metric === 'L2' ? 'L2² dist' : (data.metric === 'COSINE' ? 'Cosine sim' : 'Inner product');

      const shown = idxs.slice(0, 8).filter(id => id >= 0);
      if (shown.length === 0) {
        const empty = document.createElement('div');
        empty.classList.add('empty-state');
        empty.textContent = 'The index returned no neighbors for this query.';
        el.neighborsList.appendChild(empty);
      }
      shown.forEach((id, i) => {
        const row = document.createElement('div');
        row.classList.add('neighbor-row');
        const idSpan = document.createElement('span');
        idSpan.className = 'neighbor-id';
        idSpan.textContent = `Vector #${id}`;
        const distSpan = document.createElement('span');
        distSpan.className = 'neighbor-dist';
        distSpan.textContent = `${scoreLabel}: ${dists[i].toFixed(4)}`;
        row.appendChild(idSpan);
        row.appendChild(distSpan);
        el.neighborsList.appendChild(row);
      });

    } catch (err) {
      console.error('Search error:', err);
      el.neighborsList.innerHTML = '';
      const msg = document.createElement('div');
      msg.className = 'empty-state error';
      msg.textContent = err.message;
      el.neighborsList.appendChild(msg);
      showStatus(err.message, 'error');
    } finally {
      el.executeSearchBtn.textContent = 'Execute Test Query';
      setBusy(false);
    }
  }

  // -------------------------------------------------------------
  // Chart.js: Pareto Frontier Scatter Plot
  // -------------------------------------------------------------
  function renderParetoChart(points, winnerId) {
    if (state.paretoChart) {
      state.paretoChart.destroy();
      state.paretoChart = null;
    }
    if (!points || points.length === 0) {
      setChartOverlay(el.paretoEmpty, 'The optimizer returned no candidate configurations.');
      return;
    }
    if (typeof Chart === 'undefined') {
      setChartOverlay(el.paretoEmpty, 'Chart library failed to load (no network access to cdn.jsdelivr.net?).', true);
      return;
    }
    setChartOverlay(el.paretoEmpty, null);

    const ctx = document.getElementById('paretoChart').getContext('2d');

    const infeasiblePts = [];
    const feasiblePts = [];
    const paretoPts = [];
    let winnerPt = null;

    points.forEach(p => {
      const pt = { x: p.latency_p95, y: p.recall * 100, raw: p };
      if (p.is_winner || p.id === winnerId) {
        winnerPt = pt;
      } else if (p.is_pareto) {
        paretoPts.push(pt);
      } else if (p.is_feasible) {
        feasiblePts.push(pt);
      } else {
        infeasiblePts.push(pt);
      }
    });

    const datasets = [
      {
        label: 'Predicted to violate a target',
        data: infeasiblePts,
        backgroundColor: 'rgba(100, 116, 139, 0.4)',
        borderColor: 'rgba(100, 116, 139, 0.7)',
        pointRadius: 4,
        pointHoverRadius: 6
      },
      {
        label: 'Predicted feasible',
        data: feasiblePts,
        backgroundColor: 'rgba(16, 185, 129, 0.55)',
        borderColor: '#10b981',
        pointRadius: 4,
        pointHoverRadius: 6
      },
      {
        label: 'Pareto Optimal Candidates',
        data: paretoPts,
        backgroundColor: 'rgba(59, 130, 246, 0.85)',
        borderColor: '#3b82f6',
        pointRadius: 5,
        pointHoverRadius: 8,
        showLine: false
      }
    ];

    if (winnerPt) {
      datasets.push({
        label: 'Selected Configuration',
        data: [winnerPt],
        backgroundColor: '#f59e0b',
        borderColor: '#ffffff',
        borderWidth: 2,
        pointRadius: 9,
        pointHoverRadius: 12,
        pointStyle: 'triangle'
      });
    }

    state.paretoChart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Predicted Query Latency p95 (ms)',
              color: '#94a3b8',
              font: { family: 'Inter', size: 11, weight: 500 }
            },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            title: {
              display: true,
              text: 'Predicted Recall (%)',
              color: '#94a3b8',
              font: { family: 'Inter', size: 11, weight: 500 }
            },
            min: 0,
            max: 102,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f172a',
            titleColor: '#38bdf8',
            bodyColor: '#f8fafc',
            borderColor: '#1e293b',
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label: (ctx) => {
                const raw = ctx.raw.raw;
                return [
                  `Index: ${raw.id}`,
                  `Recall: ${(raw.recall * 100).toFixed(1)}%`,
                  `Latency p95: ${raw.latency_p95.toFixed(2)} ms`,
                  `Predicted memory: ${raw.memory_mb.toFixed(1)} MB`,
                  raw.is_feasible ? 'Predicted to meet all targets' : 'Predicted to violate a target'
                ].concat(raw.measured_recall !== null && raw.measured_recall !== undefined
                  ? [`Measured: recall ${(raw.measured_recall * 100).toFixed(1)}%, p95 ${raw.measured_latency_p95.toFixed(3)} ms`]
                  : []);
              }
            }
          }
        }
      }
    });
  }

  // -------------------------------------------------------------
  // Chart.js: Arena Comparative Bar Chart
  // -------------------------------------------------------------
  function renderArenaBarChart(results) {
    if (state.arenaBarChart) {
      state.arenaBarChart.destroy();
      state.arenaBarChart = null;
    }
    if (typeof Chart === 'undefined') {
      setChartOverlay(el.arenaEmpty, 'Chart library failed to load (no network access to cdn.jsdelivr.net?).', true);
      return;
    }
    setChartOverlay(el.arenaEmpty, null);

    const ctx = document.getElementById('arenaBarChart').getContext('2d');

    const labels = ['Exact Flat', 'FAISS Default IVF', 'FAISS Default HNSW', 'VECTA Auto-Tuned'];
    const recallData = [
      results.ground_truth_flat.recall_at_k * 100,
      results.faiss_default_ivf.recall_at_k * 100,
      results.faiss_default_hnsw.recall_at_k * 100,
      results.vecta_auto_tuned.recall_at_k * 100
    ];
    const latencyData = [
      results.ground_truth_flat.latency_p95_ms,
      results.faiss_default_ivf.latency_p95_ms,
      results.faiss_default_hnsw.latency_p95_ms,
      results.vecta_auto_tuned.latency_p95_ms
    ];

    state.arenaBarChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Recall@k (%)',
            data: recallData,
            backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#3b82f6'],
            borderRadius: 4,
            yAxisID: 'y'
          },
          {
            label: 'p95 Latency (ms)',
            data: latencyData,
            backgroundColor: 'rgba(148, 163, 184, 0.6)',
            borderRadius: 4,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          },
          y: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Recall (%)', color: '#94a3b8', font: { size: 10 } },
            min: 0,
            max: 105,
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          y1: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Latency (ms)', color: '#94a3b8', font: { size: 10 } },
            grid: { drawOnChartArea: false },
            ticks: { color: '#94a3b8' }
          }
        },
        plugins: {
          legend: {
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          }
        }
      }
    });
  }
});
