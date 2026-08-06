(() => {
  const state = {
    paths: {
      oracle_dir: "",
      agnostic_dir: "",
      agnostic_kimi_dir: "",
      tsm_dir: "",
      event_graph_dirs: "",
      questions_jsonl: "",
    },
    samples: [],
    sample: null,
    activeEventId: null,
    activeAgnosticId: null,
    activeAgnosticKimiId: null,
    activeEventGraphKey: null,
    scrolledEventId: null,
    scrolledAgnosticId: null,
    scrolledAgnosticKimiId: null,
    scrolledEventGraphKey: null,
    videoPath: null,
    duration: 0,
    sourceFps: 1,
    displayFps: 1,
    nFrames: 0,
    frameIndex: 0,
    playing: false,
    playTimer: null,
    frameReq: 0,
    frameWidth: 720,
    frameCache: new Map(),
    warmTimer: null,
    prefetching: new Set(),
    fetchInFlight: 0,
    fetchWaiters: [],
    warmReady: false,
  };

  /** Same retrieved/seed colors for every EventGraph run (matches CSS vars). */
  const EG_EVENT_COLORS = { fill: "#c084fc", seed: "#f472b6" };

  function egRunHeaderHtml(run, fallbackIndex = 0) {
    const title = escapeHtml(run.label || `EG ${fallbackIndex + 1}`);
    return `<h3>${title}</h3>`;
  }

  const el = {
    oracleDir: document.getElementById("oracle-dir"),
    agnosticDir: document.getElementById("agnostic-dir"),
    agnosticKimiDir: document.getElementById("agnostic-kimi-dir"),
    tsmDir: document.getElementById("tsm-dir"),
    eventGraphDir: document.getElementById("event-graph-dir"),
    questionsJsonl: document.getElementById("questions-jsonl"),
    loadBtn: document.getElementById("load-btn"),
    resetBtn: document.getElementById("reset-btn"),
    sampleSelect: document.getElementById("sample-select"),
    videoName: document.getElementById("video-name"),
    status: document.getElementById("status"),
    query: document.getElementById("query"),
    answer: document.getElementById("answer"),
    predAnswers: document.getElementById("pred-answers"),
    metrics: document.getElementById("metrics"),
    providerMetrics: document.getElementById("provider-metrics"),
    frame: document.getElementById("frame"),
    scrubber: document.getElementById("scrubber"),
    scrubberTime: document.getElementById("scrubber-time"),
    scrubberMeta: document.getElementById("scrubber-meta"),
    displayFps: document.getElementById("display-fps"),
    btnPrevFrame: document.getElementById("btn-prev-frame"),
    btnNextFrame: document.getElementById("btn-next-frame"),
    btnPlay: document.getElementById("btn-play"),
    overlay: document.getElementById("overlay"),
    overlayTime: document.getElementById("overlay-time"),
    overlayDesc: document.getElementById("overlay-desc"),
    tracks: document.getElementById("tracks"),
    eventList: document.getElementById("event-list"),
    eventGraphColumns: document.getElementById("event-graph-columns"),
    eventGraphCount: document.getElementById("event-graph-count"),
    agnosticList: document.getElementById("agnostic-list"),
    agnosticKimiList: document.getElementById("agnostic-kimi-list"),
    agnosticCount: document.getElementById("agnostic-count"),
    agnosticKimiCount: document.getElementById("agnostic-kimi-count"),
    agnosticMeta: document.getElementById("agnostic-meta"),
    agnosticKimiMeta: document.getElementById("agnostic-kimi-meta"),
    oracleCount: document.getElementById("oracle-count"),
    graph: document.getElementById("graph"),
    resolved: document.getElementById("resolved"),
  };

  function eventGraphRuns(detail) {
    if (detail.event_graph_runs?.length) return detail.event_graph_runs;
    if (detail.event_graph_events?.length || detail.has_event_graph) {
      return [
        {
          label: "EG",
          events: detail.event_graph_events || [],
          pred_answer: detail.pred_answer,
          qa_error: detail.qa_error,
          has_event_graph: detail.has_event_graph,
        },
      ];
    }
    return [];
  }

  function qs() {
    const p = new URLSearchParams({
      oracle_dir: state.paths.oracle_dir,
      agnostic_dir: state.paths.agnostic_dir,
      agnostic_kimi_dir: state.paths.agnostic_kimi_dir,
      tsm_dir: state.paths.tsm_dir,
      event_graph_dirs: state.paths.event_graph_dirs,
      questions_jsonl: state.paths.questions_jsonl,
    });
    return p.toString();
  }

  function setStatus(msg, isError = false) {
    el.status.textContent = msg || "";
    el.status.classList.toggle("error", Boolean(isError));
  }

  function mmss(sec) {
    if (!Number.isFinite(sec)) return "–";
    const s = Math.max(0, sec);
    const m = Math.floor(s / 60);
    const r = s - m * 60;
    if (Math.abs(r - Math.round(r)) < 1e-6) {
      return `${m}:${String(Math.round(r)).padStart(2, "0")}`;
    }
    return `${m}:${r.toFixed(1).padStart(4, "0")}`;
  }

  function overlaps(a0, a1, b0, b1) {
    return a0 < b1 && b0 < a1;
  }

  async function fetchJSON(url) {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }

  function readPathInputs() {
    state.paths = {
      oracle_dir: el.oracleDir.value.trim(),
      agnostic_dir: el.agnosticDir.value.trim(),
      agnostic_kimi_dir: el.agnosticKimiDir.value.trim(),
      tsm_dir: el.tsmDir.value.trim(),
      event_graph_dirs: el.eventGraphDir.value.trim(),
      questions_jsonl: el.questionsJsonl.value.trim(),
    };
  }

  function writePathInputs(paths) {
    el.oracleDir.value = paths.oracle_dir || "";
    el.agnosticDir.value = paths.agnostic_dir || "";
    el.agnosticKimiDir.value = paths.agnostic_kimi_dir || "";
    el.tsmDir.value = paths.tsm_dir || "";
    el.eventGraphDir.value =
      paths.event_graph_dirs || paths.event_graph_dir || "";
    el.questionsJsonl.value = paths.questions_jsonl || "";
    state.paths = {
      oracle_dir: el.oracleDir.value.trim(),
      agnostic_dir: el.agnosticDir.value.trim(),
      agnostic_kimi_dir: el.agnosticKimiDir.value.trim(),
      tsm_dir: el.tsmDir.value.trim(),
      event_graph_dirs: el.eventGraphDir.value.trim(),
      questions_jsonl: el.questionsJsonl.value.trim(),
    };
  }

  async function loadDefaults() {
    const data = await fetchJSON("/api/defaults");
    writePathInputs(data.paths);
    return data;
  }

  async function loadSamples() {
    readPathInputs();
    setStatus("Loading samples…");
    const data = await fetchJSON(`/api/samples?${qs()}`);
    state.samples = data.samples || [];
    // Keep inputs in sync with what the server actually resolved.
    writePathInputs({
      oracle_dir: data.resolved.oracle_dir,
      agnostic_dir: data.resolved.agnostic_dir,
      agnostic_kimi_dir: data.resolved.agnostic_kimi_dir,
      tsm_dir: data.resolved.tsm_dir,
      event_graph_dirs: data.resolved.event_graph_dirs,
      questions_jsonl: data.resolved.questions_jsonl,
    });
    el.resolved.textContent = "";

    el.sampleSelect.innerHTML = "";
    if (!state.samples.length) {
      el.sampleSelect.innerHTML = `<option value="">No samples found</option>`;
      setStatus("No oracle JSON files in that directory.", true);
      return;
    }
    for (const s of state.samples) {
      const opt = document.createElement("option");
      opt.value = s.sample_id;
      const vid = s.video_id || "unknown";
      const egRuns = s.n_event_graph_runs || (s.has_event_graph ? 1 : 0);
      opt.textContent =
        `${s.sample_id} · ${vid}.mp4 · ${s.n_oracle_events} oracle / ` +
        `${s.n_agnostic_events || 0} gemini / ${s.n_agnostic_kimi_events || 0} kimi / ${s.n_tsm_events} TSM` +
        (egRuns ? ` / ${egRuns} EG` : "");
      el.sampleSelect.appendChild(opt);
    }
    setStatus(`${state.samples.length} samples loaded.`);
    await loadSample(state.samples[0].sample_id);
  }

  async function loadSample(sampleId) {
    if (!sampleId) return;
    setStatus(`Loading sample ${sampleId}…`);
    const detail = await fetchJSON(`/api/sample/${encodeURIComponent(sampleId)}?${qs()}`);
    state.sample = detail;
    state.activeEventId = null;
    state.activeAgnosticId = null;
    state.activeAgnosticKimiId = null;
    state.activeEventGraphKey = null;
    el.sampleSelect.value = detail.sample_id;
    const videoFile =
      (detail.video_path && detail.video_path.split(/[/\\]/).pop()) ||
      (detail.video_id ? `${detail.video_id}.mp4` : "");
    el.videoName.textContent = videoFile
      ? `video: ${videoFile}${detail.video_id ? ` · id ${detail.video_id}` : ""}`
      : "video: (none)";
    el.query.textContent = detail.query || "(no query)";
    if (detail.answer) {
      el.answer.textContent = detail.answer;
      el.answer.classList.remove("muted");
    } else {
      el.answer.textContent = "(answer not found in questions JSONL)";
      el.answer.classList.add("muted");
    }
    renderPredAnswers(detail);
    renderMetrics(detail);
    renderProviderMetrics(detail);
    renderGraph(detail);
    renderAgnosticEvents(detail, "gemini");
    renderAgnosticEvents(detail, "kimi");
    renderEventGraphRuns(detail);
    renderEvents(detail);
    renderTracks(detail);
    setupFrameViewer(detail).catch((err) =>
      setStatus(String(err.message || err), true)
    );
    const runs = eventGraphRuns(detail);
    const egN = runs.reduce((n, run) => n + (run.events?.length || 0), 0);
    setStatus(
      detail.video_exists
        ? `Sample ${detail.sample_id} · ${videoFile || detail.video_id} · ${detail.oracle_events.length} oracle · ${detail.agnostic_events.length} gemini · ${(detail.agnostic_kimi_events || []).length} kimi · ${detail.tsm_events.length} TSM · ${runs.length} EG run(s) / ${egN} spans`
        : `Sample ${detail.sample_id} loaded, but video file is missing: ${detail.video_path || "(none)"}`,
      !detail.video_exists
    );
  }

  function renderPredAnswers(detail) {
    el.predAnswers.innerHTML = "";
    const runs = eventGraphRuns(detail);
    if (!runs.length) {
      el.predAnswers.innerHTML =
        `<div class="muted" style="padding:4px 0">No EventGraph dirs loaded. Add one path per line, then Load.</div>`;
      return;
    }
    runs.forEach((run, index) => {
      const col = document.createElement("div");
      col.className = "pred-col";
      const box = document.createElement("div");
      box.className = "query-box pred-box";
      let text = "(no prediction)";
      if (run.pred_answer) {
        text = run.pred_answer;
        box.classList.remove("muted");
      } else if (run.qa_error) {
        text = `(error) ${run.qa_error}`;
        box.classList.add("muted");
      } else if (run.has_event_graph) {
        text = "(file found, no answer field)";
        box.classList.add("muted");
      } else {
        text = "(no JSON for this sample)";
        box.classList.add("muted");
      }
      box.textContent = text;
      col.innerHTML = egRunHeaderHtml(run, index);
      col.appendChild(box);
      el.predAnswers.appendChild(col);
    });
  }

  function fmtF1(f1) {
    return f1 == null || Number.isNaN(f1) ? "–" : Number(f1).toFixed(2);
  }

  function metricCards(comparison, label) {
    if (!comparison || !comparison.summary) {
      return (
        `<div class="metrics-section">${label}</div>` +
        `<div class="metric"><div class="label">Metrics</div>` +
        `<div class="value muted">n/a</div>` +
        `<div class="hint">TSM vs agnostic — computed on load when both exist.</div></div>`
      );
    }
    const s = comparison.summary;
    const f1 = comparison.boundary_f1 || {};
    const fmtMean = (value) =>
      value == null || Number.isNaN(value) ? "–" : Number(value).toFixed(2);
    const cards = [
      [
        "Overseg",
        fmtMean(s.mean_over_segmentation),
        "Mean extra TSM segments per agnostic event: max(0, n_tsm - 1). Lower is better.",
      ],
      [
        "Underseg",
        fmtMean(s.mean_under_segmentation),
        "Mean extra agnostic events per TSM segment: max(0, n_ref - 1). Lower is better.",
      ],
      [
        "Contam.",
        `${((s.mean_contamination_ratio || 0) * 100).toFixed(0)}%`,
        "Fraction of TSM cover outside the agnostic event (lower is better).",
      ],
      [
        "Start err",
        `${(s.mean_start_boundary_error_sec || 0).toFixed(1)}s`,
        "Mean |agnostic start − covering TSM start| in seconds.",
      ],
      [
        "F1@2s",
        fmtF1(f1["2s"]?.f1),
        "Primary boundary F1: one-to-one match within 2s (tight; preferred).",
      ],
      [
        "F1@5s",
        fmtF1(f1["5s"]?.f1),
        "Boundary F1 with 5s tolerance (medium).",
      ],
      [
        "F1@10s",
        fmtF1(f1["10s"]?.f1),
        "Loose diagnostic boundary F1 with 10s tolerance.",
      ],
    ]
      .map(
        ([lbl, value, hint]) =>
          `<div class="metric">` +
          `<div class="label">${lbl}</div>` +
          `<div class="value">${value}</div>` +
          `<div class="hint">${hint}</div>` +
          `</div>`
      )
      .join("");
    return `<div class="metrics-section">${label}</div>${cards}`;
  }

  function renderMetrics(detail) {
    el.metrics.innerHTML =
      metricCards(detail.comparison, "Gemini vs TSM") +
      metricCards(detail.comparison_kimi, "Kimi vs TSM");
  }

  function renderProviderMetrics(detail) {
    const pc = detail.provider_comparison;
    if (!pc) {
      el.providerMetrics.innerHTML =
        `<div class="metric" style="grid-column:1/-1">` +
        `<div class="label">Provider comparison</div>` +
        `<div class="value muted">n/a</div>` +
        `<div class="hint">Need both Gemini and Kimi agnostic annotations for this video.</div></div>`;
      return;
    }
    const f1 = pc.boundary_f1 || {};
    const modelA = pc.model_a || detail.agnostic_model || "Gemini";
    const modelB = pc.model_b || detail.agnostic_kimi_model || "Kimi";
    el.providerMetrics.innerHTML = [
      [
        "Events",
        `${pc.n_events_a} / ${pc.n_events_b}`,
        `${modelA} vs ${modelB} event counts.`,
      ],
      [
        "F1@2s",
        fmtF1(f1["2s"]?.f1),
        `Boundary agreement within 2s (${modelB} pred vs ${modelA} ref).`,
      ],
      [
        "F1@5s",
        fmtF1(f1["5s"]?.f1),
        "Boundary agreement within 5s.",
      ],
      [
        "F1@10s",
        fmtF1(f1["10s"]?.f1),
        "Loose boundary agreement within 10s.",
      ],
    ]
      .map(
        ([label, value, hint]) =>
          `<div class="metric">` +
          `<div class="label">${label}</div>` +
          `<div class="value">${value}</div>` +
          `<div class="hint">${hint}</div>` +
          `</div>`
      )
      .join("");
  }

  function readDisplayFps() {
    const raw = Number(el.displayFps.value);
    if (!Number.isFinite(raw) || raw <= 0) return state.displayFps || 1;
    return Math.min(30, Math.max(0.1, raw));
  }

  function currentTimeSec() {
    if (!state.nFrames) return 0;
    return Math.min(state.duration, state.frameIndex / state.displayFps);
  }

  function stopPlayback() {
    state.playing = false;
    if (state.playTimer != null) {
      clearInterval(state.playTimer);
      state.playTimer = null;
    }
    el.btnPlay.textContent = "▶";
  }

  function clearFrameCache() {
    for (const url of state.frameCache.values()) {
      URL.revokeObjectURL(url);
    }
    state.frameCache.clear();
    state.prefetching.clear();
    state.fetchInFlight = 0;
    state.fetchWaiters = [];
    state.warmReady = false;
    if (state.warmTimer != null) {
      clearInterval(state.warmTimer);
      state.warmTimer = null;
    }
  }

  function frameApiUrl(index) {
    const t = Math.min(state.duration, index / state.displayFps);
    return (
      `/api/frame?path=${encodeURIComponent(state.videoPath)}` +
      `&t=${encodeURIComponent(t.toFixed(3))}` +
      `&fps=${encodeURIComponent(state.displayFps)}` +
      `&w=${encodeURIComponent(state.frameWidth)}`
    );
  }

  function cacheKey(index) {
    return `${state.videoPath}|${state.displayFps}|${state.frameWidth}|${index}`;
  }

  function acquireFetchSlot() {
    const limit = state.warmReady ? 6 : 2;
    if (state.fetchInFlight < limit) {
      state.fetchInFlight += 1;
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      state.fetchWaiters.push(resolve);
    }).then(() => {
      state.fetchInFlight += 1;
    });
  }

  function releaseFetchSlot() {
    state.fetchInFlight = Math.max(0, state.fetchInFlight - 1);
    const next = state.fetchWaiters.shift();
    if (next) next();
  }

  async function ensureFrame(index) {
    if (!state.videoPath || index < 0 || index >= state.nFrames) return null;
    const key = cacheKey(index);
    if (state.frameCache.has(key)) return state.frameCache.get(key);
    if (state.prefetching.has(key)) {
      while (state.prefetching.has(key)) {
        await new Promise((r) => setTimeout(r, 40));
      }
      return state.frameCache.get(key) || null;
    }
    state.prefetching.add(key);
    await acquireFetchSlot();
    try {
      if (state.frameCache.has(key)) return state.frameCache.get(key);
      const res = await fetch(frameApiUrl(index));
      if (!res.ok) throw new Error(`frame ${index}: ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      state.frameCache.set(key, url);
      return url;
    } finally {
      state.prefetching.delete(key);
      releaseFetchSlot();
    }
  }

  function prefetchAround(center) {
    const radius = state.warmReady ? 10 : 3;
    const order = [];
    for (let d = 1; d <= radius; d += 1) {
      order.push(center + d, center - d);
    }
    for (const idx of order) {
      if (idx < 0 || idx >= state.nFrames) continue;
      const key = cacheKey(idx);
      if (state.frameCache.has(key) || state.prefetching.has(key)) continue;
      ensureFrame(idx).catch(() => {});
    }
  }

  function updateScrubberMeta(extra = "") {
    const base =
      `scrub ${state.displayFps} fps · ${state.nFrames} steps · ` +
      `src ${state.sourceFps.toFixed(2)} fps · ${mmss(state.duration)}`;
    el.scrubberMeta.textContent = extra ? `${base} · ${extra}` : base;
  }

  function startWarmAndPoll() {
    if (!state.videoPath) return;
    if (state.warmTimer != null) {
      clearInterval(state.warmTimer);
      state.warmTimer = null;
    }
    const qsWarm =
      `path=${encodeURIComponent(state.videoPath)}` +
      `&fps=${encodeURIComponent(state.displayFps)}` +
      `&w=${encodeURIComponent(state.frameWidth)}`;
    fetchJSON(`/api/warm?${qsWarm}`)
      .then((st) => {
        state.warmReady = st.status === "done" || st.cached >= st.total;
        updateScrubberMeta(
          state.warmReady
            ? `cached ${st.cached}/${st.total}`
            : `warming ${st.cached}/${st.total}`
        );
        if (state.warmReady) prefetchAround(state.frameIndex);
      })
      .catch(() => {});
    state.warmTimer = setInterval(() => {
      fetchJSON(`/api/warm_status?${qsWarm}`)
        .then((st) => {
          if (st.status === "running") {
            state.warmReady = false;
            updateScrubberMeta(`warming ${st.cached}/${st.total}`);
          } else if (st.status === "done" || st.cached >= st.total) {
            state.warmReady = true;
            updateScrubberMeta(`cached ${st.cached}/${st.total}`);
            clearInterval(state.warmTimer);
            state.warmTimer = null;
            prefetchAround(state.frameIndex);
          } else if (st.status === "error") {
            updateScrubberMeta("warm failed");
            clearInterval(state.warmTimer);
            state.warmTimer = null;
          } else {
            updateScrubberMeta(`cache ${st.cached}/${st.total}`);
          }
        })
        .catch(() => {});
    }, 1500);
  }

  function applyDisplayGrid({ keepTime = false } = {}) {
    const tKeep = keepTime ? currentTimeSec() : 0;
    state.displayFps = readDisplayFps();
    el.displayFps.value = String(state.displayFps);
    clearFrameCache();
    if (!state.duration) {
      state.nFrames = 0;
      return;
    }
    state.nFrames = Math.max(1, Math.floor(state.duration * state.displayFps) + 1);
    el.scrubber.disabled = false;
    el.scrubber.min = "0";
    el.scrubber.max = String(Math.max(0, state.nFrames - 1));
    el.scrubber.step = "1";
    updateScrubberMeta();
    const idx = keepTime
      ? Math.round(tKeep * state.displayFps)
      : 0;
    setFrameIndex(idx, { force: true });
    startWarmAndPoll();
  }

  async function setupFrameViewer(detail) {
    stopPlayback();
    clearFrameCache();
    state.videoPath = null;
    state.duration = 0;
    state.sourceFps = 1;
    state.nFrames = 0;
    state.frameIndex = 0;
    el.frame.removeAttribute("src");
    el.scrubber.disabled = true;
    el.scrubber.value = "0";
    el.scrubberTime.textContent = "0:00";
    el.scrubberMeta.textContent = "no video";
    el.overlay.classList.add("hidden");
    if (!detail.video_path || !detail.video_exists) return;

    const meta = await fetchJSON(
      `/api/video_meta?path=${encodeURIComponent(detail.video_path)}`
    );
    state.videoPath = detail.video_path;
    state.duration = Number(meta.duration_sec) || detail.duration_sec || 0;
    state.sourceFps = Math.max(1e-3, Number(meta.fps) || 1);
    applyDisplayGrid({ keepTime: false });
  }

  async function showFrameAtIndex(index) {
    if (!state.videoPath || !state.nFrames) return;
    const req = ++state.frameReq;
    try {
      const url = await ensureFrame(index);
      if (req !== state.frameReq || !url) return;
      el.frame.src = url;
      prefetchAround(index);
    } catch (err) {
      if (req !== state.frameReq) return;
      setStatus(String(err.message || err), true);
    }
  }

  function setFrameIndex(index, { force = false, fetchFrame = true } = {}) {
    if (!state.nFrames && !force) return;
    const clamped = Math.max(0, Math.min(state.nFrames - 1, Math.round(index)));
    state.frameIndex = clamped;
    el.scrubber.value = String(clamped);
    const t = currentTimeSec();
    el.scrubberTime.textContent = `${mmss(t)} · #${clamped}`;
    updatePlayhead();
    if (fetchFrame) showFrameAtIndex(clamped);
  }

  function activeOracleAt(t) {
    const events = state.sample?.oracle_events || [];
    return (
      events.find((e) => t >= e.start_sec && t < e.end_sec) ||
      events.find((e) => Math.abs(t - e.start_sec) < 0.25) ||
      null
    );
  }

  function scrollActiveCard(listEl, selector, prevKey) {
    if (!listEl) return prevKey;
    const card = listEl.querySelector(selector);
    if (!card) return null;
    const id =
      card.dataset.agnosticId ||
      card.dataset.eventGraphKey ||
      card.dataset.id ||
      null;
    if (id && id !== prevKey) {
      card.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
    }
    return id;
  }

  function updatePlayhead() {
    const detail = state.sample;
    if (!detail) return;
    const t = currentTimeSec();
    const activeOracle = activeOracleAt(t);
    const activeAgnostic = activeAgnosticAt(t);
    const activeAgnosticKimi = activeAgnosticKimiAt(t);
    const activeEventGraph = activeEventGraphAt(t);
    state.activeEventId = activeOracle ? activeOracle.event_id : null;
    state.activeAgnosticId = activeAgnostic ? activeAgnostic.event_id : null;
    state.activeAgnosticKimiId = activeAgnosticKimi
      ? activeAgnosticKimi.event_id
      : null;
    state.activeEventGraphKey = activeEventGraph
      ? `${activeEventGraph.runIndex}:${activeEventGraph.event.event_id}`
      : null;

    const overlayEvent =
      activeAgnostic ||
      activeAgnosticKimi ||
      activeOracle ||
      (activeEventGraph ? activeEventGraph.event : null);
    if (overlayEvent) {
      const kind = activeAgnostic
        ? "gemini"
        : activeAgnosticKimi
          ? "kimi"
          : activeOracle
            ? "oracle"
            : `eg:${activeEventGraph.runIndex + 1}`;
      el.overlay.classList.remove("hidden");
      const ev = activeEventGraph ? activeEventGraph.event : overlayEvent;
      el.overlayTime.textContent =
        `${mmss(ev.start_sec)} – ${mmss(ev.end_sec)} · ` +
        `${kind} · ${ev.event_id}`;
      el.overlayDesc.textContent = ev.description || ev.event_id;
    } else {
      el.overlay.classList.add("hidden");
    }

    document.querySelectorAll("#event-list .event-card").forEach((card) => {
      card.classList.toggle("active", card.dataset.id === state.activeEventId);
    });
    document.querySelectorAll("#agnostic-list .event-card").forEach((card) => {
      card.classList.toggle(
        "active",
        card.dataset.agnosticId === state.activeAgnosticId
      );
    });
    document.querySelectorAll("#agnostic-kimi-list .event-card").forEach((card) => {
      card.classList.toggle(
        "active",
        card.dataset.agnosticId === state.activeAgnosticKimiId
      );
    });
    document.querySelectorAll("#event-graph-columns .event-card").forEach((card) => {
      card.classList.toggle(
        "active",
        card.dataset.eventGraphKey === state.activeEventGraphKey
      );
    });

    if (state.activeAgnosticId) {
      state.scrolledAgnosticId = scrollActiveCard(
        el.agnosticList,
        `#agnostic-list .event-card.active`,
        state.scrolledAgnosticId
      );
    } else {
      state.scrolledAgnosticId = null;
    }
    if (state.activeAgnosticKimiId) {
      state.scrolledAgnosticKimiId = scrollActiveCard(
        el.agnosticKimiList,
        `#agnostic-kimi-list .event-card.active`,
        state.scrolledAgnosticKimiId
      );
    } else {
      state.scrolledAgnosticKimiId = null;
    }
    if (state.activeEventGraphKey) {
      state.scrolledEventGraphKey = scrollActiveCard(
        el.eventGraphColumns,
        `#event-graph-columns .event-card.active`,
        state.scrolledEventGraphKey
      );
    } else {
      state.scrolledEventGraphKey = null;
    }
    if (state.activeEventId) {
      state.scrolledEventId = scrollActiveCard(
        el.eventList,
        `#event-list .event-card.active`,
        state.scrolledEventId
      );
    } else {
      state.scrolledEventId = null;
    }

    renderTracks(detail, t);
  }

  function seekTo(sec) {
    if (!state.nFrames) return;
    stopPlayback();
    setFrameIndex(sec * state.displayFps);
  }

  function stepFrames(delta) {
    if (!state.nFrames) return;
    stopPlayback();
    setFrameIndex(state.frameIndex + delta);
  }

  function togglePlay() {
    if (!state.nFrames) return;
    if (state.playing) {
      stopPlayback();
      return;
    }
    state.playing = true;
    el.btnPlay.textContent = "⏸";
    const intervalMs = Math.max(50, Math.round(1000 / state.displayFps));
    state.playTimer = setInterval(() => {
      if (state.frameIndex >= state.nFrames - 1) {
        stopPlayback();
        return;
      }
      setFrameIndex(state.frameIndex + 1);
    }, intervalMs);
  }

  function coveringTsm(event, tsmEvents) {
    return tsmEvents.filter((seg) =>
      overlaps(seg.start_sec, seg.end_sec, event.start_sec, event.end_sec)
    );
  }

  function escapeHtml(text) {
    return String(text ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function activeEventGraphAt(t) {
    for (let runIndex = 0; runIndex < eventGraphRuns(state.sample || {}).length; runIndex++) {
      const events = eventGraphRuns(state.sample)[runIndex].events || [];
      const hit =
        events.find((e) => t >= e.start_sec && t < e.end_sec) ||
        events.find((e) => Math.abs(t - e.start_sec) < 0.25);
      if (hit) return { runIndex, event: hit };
    }
    return null;
  }

  function activeAgnosticAt(t) {
    const events = state.sample?.agnostic_events || [];
    return (
      events.find((e) => t >= e.start_sec && t < e.end_sec) ||
      events.find((e) => Math.abs(t - e.start_sec) < 0.25) ||
      null
    );
  }

  function activeAgnosticKimiAt(t) {
    const events = state.sample?.agnostic_kimi_events || [];
    return (
      events.find((e) => t >= e.start_sec && t < e.end_sec) ||
      events.find((e) => Math.abs(t - e.start_sec) < 0.25) ||
      null
    );
  }

  function renderAgnosticEvents(detail, provider) {
    const isKimi = provider === "kimi";
    const events = isKimi
      ? detail.agnostic_kimi_events || []
      : detail.agnostic_events || [];
    const listEl = isKimi ? el.agnosticKimiList : el.agnosticList;
    const countEl = isKimi ? el.agnosticKimiCount : el.agnosticCount;
    const metaEl = isKimi ? el.agnosticKimiMeta : el.agnosticMeta;
    const comparison = isKimi ? detail.comparison_kimi : detail.comparison;
    const cardClass = isKimi ? "event-card kimi" : "event-card agnostic";
    const model = isKimi ? detail.agnostic_kimi_model : detail.agnostic_model;
    const promptVer = isKimi
      ? detail.agnostic_kimi_prompt_version
      : detail.agnostic_prompt_version;

    listEl.innerHTML = "";
    if (isKimi) {
      state.scrolledAgnosticKimiId = null;
    } else {
      state.scrolledAgnosticId = null;
    }
    countEl.textContent = events.length ? `(${events.length})` : "(none)";
    if (model || promptVer) {
      metaEl.textContent = [model, promptVer].filter(Boolean).join(" · ");
    } else {
      metaEl.textContent = "";
    }
    if (!events.length) {
      listEl.innerHTML =
        `<div class="muted" style="padding:4px 0">No ${isKimi ? "Kimi" : "Gemini"} agnostic annotation for this video.</div>`;
      return;
    }
    const metricsById = {};
    for (const ev of comparison?.events || []) {
      metricsById[ev.event_id] = ev;
    }
    const sorted = [...events].sort(
      (a, b) => a.start_sec - b.start_sec || a.end_sec - b.end_sec
    );
    sorted.forEach((e, i) => {
      const span = e.end_sec - e.start_sec;
      const parents = (e.parent_event_ids || []).join(", ");
      const cats = (e.categories || []).join(", ");
      const m = metricsById[e.event_id];
      const cover = coveringTsm(e, detail.tsm_events);
      const coverText = cover.length
        ? `TSM cover: ${cover.length} seg → ${mmss(cover[0].start_sec)}–${mmss(cover[cover.length - 1].end_sec)}`
        : "No overlapping TSM segment";
      const metricText = m
        ? `overseg ${m.over_segmentation ?? Math.max(0, (m.fragmentation || 0) - 1)} · contam ${((m.contamination_ratio || 0) * 100).toFixed(0)}% · Δt ${(m.start_boundary_error_sec ?? 0).toFixed(1)}/${(m.end_boundary_error_sec ?? 0).toFixed(1)}s`
        : "";
      const card = document.createElement("div");
      card.className = cardClass;
      card.dataset.agnosticId = e.event_id;
      card.innerHTML = `
        <div class="meta">
          <span>${i + 1}. ${escapeHtml(e.event_id)}</span>
          <span>${mmss(e.start_sec)} – ${mmss(e.end_sec)} (${Math.round(span)}s)</span>
        </div>
        <div class="title">${escapeHtml(e.granularity || "event")}${
          parents ? " · child of " + escapeHtml(parents) : " · root"
        }</div>
        <div class="desc">${escapeHtml(e.description)}</div>
        <div class="cover">${escapeHtml(cats || "no categories")} · ${escapeHtml(coverText)}${
          metricText ? " · " + escapeHtml(metricText) : ""
        }</div>
      `;
      card.addEventListener("click", () => seekTo(e.start_sec));
      listEl.appendChild(card);
    });
  }

  function renderEventGraphRuns(detail) {
    el.eventGraphColumns.innerHTML = "";
    state.scrolledEventGraphKey = null;
    const runs = eventGraphRuns(detail);
    const total = runs.reduce((n, run) => n + (run.events?.length || 0), 0);
    el.eventGraphCount.textContent = runs.length
      ? `(${runs.length} run${runs.length === 1 ? "" : "s"}, ${total} spans)`
      : "(none)";
    if (!runs.length) {
      el.eventGraphColumns.innerHTML =
        `<div class="muted" style="padding:4px 0">No EventGraph retrieval. Run tests.debug_event_graph_pilot into each dir.</div>`;
      return;
    }
    runs.forEach((run, runIndex) => {
      const col = document.createElement("div");
      col.className = "event-graph-col";
      const list = document.createElement("div");
      list.className = "event-list";
      const events = run.events || [];
      if (!events.length) {
        list.innerHTML =
          `<div class="muted" style="padding:4px 0">No JSON for this sample.</div>`;
      } else {
        const sorted = [...events].sort(
          (a, b) => (a.rank || 0) - (b.rank || 0) || a.start_sec - b.start_sec
        );
        sorted.forEach((e) => {
          const span = e.end_sec - e.start_sec;
          const score =
            e.retrieval_score == null || Number.isNaN(Number(e.retrieval_score))
              ? "n/a"
              : Number(e.retrieval_score).toFixed(3);
          const card = document.createElement("div");
          card.className = "event-card event-graph" + (e.is_seed ? " seed" : "");
          card.dataset.eventGraphKey = `${runIndex}:${e.event_id}`;
          card.innerHTML = `
            <div class="meta">
              <span>#${e.rank || "?"} · seg ${e.segment_idx}${e.is_seed ? " · seed" : ""}</span>
              <span>${mmss(e.start_sec)} – ${mmss(e.end_sec)} (${Math.round(span)}s)</span>
            </div>
            <div class="title">${escapeHtml(e.event_id)}</div>
            <div class="cover">score ${escapeHtml(score)}</div>
          `;
          card.addEventListener("click", () => seekTo(e.start_sec));
          list.appendChild(card);
        });
      }
      col.innerHTML = egRunHeaderHtml(run, runIndex);
      col.appendChild(list);
      el.eventGraphColumns.appendChild(col);
    });
  }

  function renderEvents(detail) {
    el.eventList.innerHTML = "";
    state.scrolledEventId = null;
    const oracle = detail.oracle_events || [];
    el.oracleCount.textContent = oracle.length ? `(${oracle.length})` : "(none)";
    oracle.forEach((e, i) => {
      const card = document.createElement("div");
      card.className = "event-card";
      if (e.query_roles.includes("answer_bearing")) card.classList.add("answer");
      card.dataset.id = e.event_id;
      const roles = e.query_roles.join(", ");
      card.innerHTML = `
        <div class="meta">
          <span>${i + 1}. ${escapeHtml(e.event_id)}</span>
          <span>${mmss(e.start_sec)} – ${mmss(e.end_sec)}</span>
        </div>
        <div class="title">${escapeHtml(e.granularity || "event")}${
          roles ? " · " + escapeHtml(roles) : ""
        }</div>
        <div class="desc">${escapeHtml(e.description)}</div>
        <div class="cover">query-conditioned (metrics use agnostic track)</div>
      `;
      card.addEventListener("click", () => seekTo(e.start_sec));
      el.eventList.appendChild(card);
    });
  }

  function renderGraph(detail) {
    const svg = el.graph;
    const events = detail.oracle_events;
    const edges = detail.oracle_edges || [];
    const w = svg.clientWidth || 300;
    const h = 140;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);

    if (!events.length) {
      svg.innerHTML = `<text x="12" y="24" fill="#9aa3b5" font-size="12">No events</text>`;
      return;
    }

    const sorted = [...events].sort((a, b) => a.start_sec - b.start_sec);
    const nodes = sorted.map((e, i) => {
      const x = 40 + (i * (w - 80)) / Math.max(1, sorted.length - 1 || 1);
      return { id: e.event_id, x: sorted.length === 1 ? w / 2 : x, y: 70, e };
    });
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

    let markup = "";
    for (const edge of edges) {
      const a = byId[edge.source];
      const b = byId[edge.target];
      if (!a || !b) continue;
      const mx = (a.x + b.x) / 2;
      markup += `<path d="M${a.x},${a.y} Q${mx},${a.y - 36} ${b.x},${b.y}" fill="none" stroke="#5b8cff" stroke-width="1.5" />`;
    }
    for (const n of nodes) {
      const answer = n.e.query_roles.includes("answer_bearing");
      markup += `<circle cx="${n.x}" cy="${n.y}" r="10" fill="${answer ? "#3ecf8e" : "#5b8cff"}" />`;
      markup += `<text x="${n.x}" y="${n.y + 28}" text-anchor="middle" fill="#9aa3b5" font-size="10">${n.id}</text>`;
    }
    if (!edges.length) {
      markup += `<text x="12" y="18" fill="#9aa3b5" font-size="11">No parent→child edges</text>`;
    }
    svg.innerHTML = markup;
  }

  function renderTracks(detail, playhead = null) {
    const svg = el.tracks;
    const duration = Math.max(detail.duration_sec || 1, 1);
    const w = svg.clientWidth || 640;
    const left = 70;
    const trackW = w - left - 8;
    const tsmY = 16;
    const gemY = 40;
    const kimiY = 64;
    const oraY = 88;
    const runs = eventGraphRuns(detail);
    const egStartY = 112;
    const rowStep = 24;
    const barH = 14;
    const height = Math.max(170, egStartY + runs.length * rowStep + 34);
    svg.setAttribute("viewBox", `0 0 ${w} ${height}`);
    svg.setAttribute("height", String(height));

    const oracle = detail.oracle_events || [];
    const agnostic = detail.agnostic_events || [];
    const agnosticKimi = detail.agnostic_kimi_events || [];
    const allAgnostic = [...agnostic, ...agnosticKimi];
    const covering = new Set();
    detail.tsm_events.forEach((seg, i) => {
      if (
        allAgnostic.some((e) =>
          overlaps(seg.start_sec, seg.end_sec, e.start_sec, e.end_sec)
        )
      ) {
        covering.add(i);
      }
    });

    let markup = "";
    markup += `<text x="0" y="${tsmY + 11}" fill="#9aa3b5" font-size="11">TSM</text>`;
    markup += `<text x="0" y="${gemY + 11}" fill="#9aa3b5" font-size="11">Gemini</text>`;
    markup += `<text x="0" y="${kimiY + 11}" fill="#9aa3b5" font-size="11">Kimi</text>`;
    markup += `<text x="0" y="${oraY + 11}" fill="#9aa3b5" font-size="11">Oracle</text>`;
    markup += `<rect x="${left}" y="${tsmY}" width="${trackW}" height="${barH}" fill="#222632" />`;
    markup += `<rect x="${left}" y="${gemY}" width="${trackW}" height="${barH}" fill="#222632" />`;
    markup += `<rect x="${left}" y="${kimiY}" width="${trackW}" height="${barH}" fill="#222632" />`;
    markup += `<rect x="${left}" y="${oraY}" width="${trackW}" height="${barH}" fill="#222632" />`;
    runs.forEach((run, runIndex) => {
      const y = egStartY + runIndex * rowStep;
      markup += `<text x="0" y="${y + 11}" fill="#9aa3b5" font-size="11">${escapeHtml(run.label || `EG${runIndex + 1}`).slice(0, 10)}</text>`;
      markup += `<rect x="${left}" y="${y}" width="${trackW}" height="${barH}" fill="#222632" />`;
    });

    detail.tsm_events.forEach((seg, i) => {
      const x = left + (seg.start_sec / duration) * trackW;
      const ww = Math.max(1, ((seg.end_sec - seg.start_sec) / duration) * trackW);
      const fill = covering.has(i) ? "#c4cad8" : "#6b7385";
      markup += `<rect x="${x}" y="${tsmY}" width="${ww}" height="${barH}" fill="${fill}" stroke="#0d0f14" stroke-width="0.6" data-t="${seg.start_sec}" />`;
    });

    agnostic.forEach((e, i) => {
      const x = left + (e.start_sec / duration) * trackW;
      const ww = Math.max(2, ((e.end_sec - e.start_sec) / duration) * trackW);
      markup += `<rect x="${x}" y="${gemY}" width="${ww}" height="${barH}" fill="#d4a017" stroke="#0d0f14" stroke-width="0.5" data-t="${e.start_sec}" rx="2" />`;
      if (ww >= 12) {
        markup += `<text x="${x + ww / 2}" y="${gemY + 11}" text-anchor="middle" fill="#0b1020" font-size="9" font-weight="600">${i + 1}</text>`;
      }
    });

    agnosticKimi.forEach((e, i) => {
      const x = left + (e.start_sec / duration) * trackW;
      const ww = Math.max(2, ((e.end_sec - e.start_sec) / duration) * trackW);
      markup += `<rect x="${x}" y="${kimiY}" width="${ww}" height="${barH}" fill="#17a2d4" stroke="#0d0f14" stroke-width="0.5" data-t="${e.start_sec}" rx="2" />`;
      if (ww >= 12) {
        markup += `<text x="${x + ww / 2}" y="${kimiY + 11}" text-anchor="middle" fill="#0b1020" font-size="9" font-weight="600">${i + 1}</text>`;
      }
    });

    oracle.forEach((e, i) => {
      const x = left + (e.start_sec / duration) * trackW;
      const ww = Math.max(2, ((e.end_sec - e.start_sec) / duration) * trackW);
      const answer = e.query_roles.includes("answer_bearing");
      const fill = answer ? "#3ecf8e" : "#5b8cff";
      markup += `<rect x="${x}" y="${oraY}" width="${ww}" height="${barH}" fill="${fill}" data-t="${e.start_sec}" rx="2" />`;
      if (ww >= 14) {
        markup += `<text x="${x + ww / 2}" y="${oraY + 11}" text-anchor="middle" fill="#0b1020" font-size="10" font-weight="600">${i + 1}</text>`;
      }
    });

    runs.forEach((run, runIndex) => {
      const y = egStartY + runIndex * rowStep;
      (run.events || []).forEach((e) => {
        const x = left + (e.start_sec / duration) * trackW;
        const ww = Math.max(2, ((e.end_sec - e.start_sec) / duration) * trackW);
        const fill = e.is_seed ? EG_EVENT_COLORS.seed : EG_EVENT_COLORS.fill;
        markup += `<rect x="${x}" y="${y}" width="${ww}" height="${barH}" fill="${fill}" stroke="#0d0f14" stroke-width="0.5" data-t="${e.start_sec}" rx="2" />`;
        if (ww >= 12) {
          markup += `<text x="${x + ww / 2}" y="${y + 11}" text-anchor="middle" fill="#0b1020" font-size="9" font-weight="600">${e.rank || ""}</text>`;
        }
      });
    });

    // comparison cover brackets from report (Gemini)
    for (const m of detail.comparison?.events || []) {
      const [c0, c1] = m.tsm_cover_sec || [];
      if (c0 == null || c1 == null) continue;
      const x = left + (c0 / duration) * trackW;
      const ww = Math.max(2, ((c1 - c0) / duration) * trackW);
      markup += `<rect x="${x}" y="${tsmY - 3}" width="${ww}" height="${barH + 6}" fill="none" stroke="#e06c75" stroke-width="1" stroke-dasharray="3 2" opacity="0.85" />`;
    }

    if (playhead != null) {
      const x = left + (playhead / duration) * trackW;
      markup += `<line x1="${x}" y1="8" x2="${x}" y2="${height - 14}" stroke="#fff" stroke-width="1.2" />`;
    }

    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      const t = f * duration;
      const x = left + f * trackW;
      markup += `<line x1="${x}" y1="${height - 12}" x2="${x}" y2="${height - 8}" stroke="#2e3445" />`;
      markup += `<text x="${x}" y="${height - 1}" fill="#9aa3b5" font-size="10" text-anchor="${f === 0 ? "start" : f === 1 ? "end" : "middle"}">${mmss(t)}</text>`;
    });

    svg.innerHTML = markup;
  }

  el.tracks.addEventListener("click", (ev) => {
    const detail = state.sample;
    if (!detail) return;
    const rect = el.tracks.getBoundingClientRect();
    const left = 70;
    const trackW = rect.width - left - 8;
    const x = ev.clientX - rect.left - left;
    if (x < 0 || trackW <= 0) return;
    const t = (x / trackW) * Math.max(detail.duration_sec || 1, 1);
    seekTo(t);
  });

  el.loadBtn.addEventListener("click", () => {
    loadSamples().catch((err) => setStatus(String(err.message || err), true));
  });
  el.resetBtn.addEventListener("click", () => {
    loadDefaults()
      .then(() => loadSamples())
      .catch((err) => setStatus(String(err.message || err), true));
  });
  el.sampleSelect.addEventListener("change", () => {
    loadSample(el.sampleSelect.value).catch((err) =>
      setStatus(String(err.message || err), true)
    );
  });
  let scrubDebounce = null;
  el.scrubber.addEventListener("input", () => {
    stopPlayback();
    const idx = Number(el.scrubber.value);
    setFrameIndex(idx, { fetchFrame: false });
    if (scrubDebounce != null) clearTimeout(scrubDebounce);
    scrubDebounce = setTimeout(() => {
      scrubDebounce = null;
      showFrameAtIndex(state.frameIndex);
    }, 80);
  });
  el.scrubber.addEventListener("change", () => {
    if (scrubDebounce != null) {
      clearTimeout(scrubDebounce);
      scrubDebounce = null;
    }
    showFrameAtIndex(state.frameIndex);
  });
  el.btnPrevFrame.addEventListener("click", () => stepFrames(-1));
  el.btnNextFrame.addEventListener("click", () => stepFrames(1));
  el.btnPlay.addEventListener("click", togglePlay);
  el.displayFps.addEventListener("change", () => {
    if (!state.videoPath) {
      state.displayFps = readDisplayFps();
      el.displayFps.value = String(state.displayFps);
      return;
    }
    const wasPlaying = state.playing;
    stopPlayback();
    applyDisplayGrid({ keepTime: true });
    if (wasPlaying) togglePlay();
  });
  window.addEventListener("keydown", (ev) => {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;
    if (ev.code === "Space") {
      ev.preventDefault();
      togglePlay();
    } else if (ev.key === "," || ev.key === "ArrowLeft") {
      ev.preventDefault();
      stepFrames(-1);
    } else if (ev.key === "." || ev.key === "ArrowRight") {
      ev.preventDefault();
      stepFrames(1);
    }
  });
  window.addEventListener("resize", () => {
    if (state.sample) {
      renderTracks(state.sample, currentTimeSec());
      renderGraph(state.sample);
    }
  });

  loadDefaults()
    .then(() => loadSamples())
    .catch((err) => setStatus(String(err.message || err), true));
})();
