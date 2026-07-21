(() => {
  const state = {
    paths: {
      oracle_dir: "",
      agnostic_dir: "",
      tsm_dir: "",
      report_json: "",
      questions_jsonl: "",
    },
    samples: [],
    sample: null,
    activeEventId: null,
    activeAgnosticId: null,
    scrolledEventId: null,
    scrolledAgnosticId: null,
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

  const el = {
    oracleDir: document.getElementById("oracle-dir"),
    agnosticDir: document.getElementById("agnostic-dir"),
    tsmDir: document.getElementById("tsm-dir"),
    reportJson: document.getElementById("report-json"),
    questionsJsonl: document.getElementById("questions-jsonl"),
    loadBtn: document.getElementById("load-btn"),
    resetBtn: document.getElementById("reset-btn"),
    sampleSelect: document.getElementById("sample-select"),
    videoName: document.getElementById("video-name"),
    status: document.getElementById("status"),
    query: document.getElementById("query"),
    answer: document.getElementById("answer"),
    metrics: document.getElementById("metrics"),
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
    agnosticList: document.getElementById("agnostic-list"),
    agnosticCount: document.getElementById("agnostic-count"),
    oracleCount: document.getElementById("oracle-count"),
    graph: document.getElementById("graph"),
    resolved: document.getElementById("resolved"),
  };

  function qs() {
    const p = new URLSearchParams({
      oracle_dir: state.paths.oracle_dir,
      agnostic_dir: state.paths.agnostic_dir,
      tsm_dir: state.paths.tsm_dir,
      report_json: state.paths.report_json,
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
      tsm_dir: el.tsmDir.value.trim(),
      report_json: el.reportJson.value.trim(),
      questions_jsonl: el.questionsJsonl.value.trim(),
    };
  }

  function writePathInputs(paths) {
    el.oracleDir.value = paths.oracle_dir || "";
    el.agnosticDir.value = paths.agnostic_dir || "";
    el.tsmDir.value = paths.tsm_dir || "";
    el.reportJson.value = paths.report_json || "";
    el.questionsJsonl.value = paths.questions_jsonl || "";
    state.paths = { ...paths };
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
      tsm_dir: data.resolved.tsm_dir,
      report_json: data.resolved.report_json,
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
      opt.textContent =
        `${s.sample_id} · ${vid}.mp4 · ${s.n_oracle_events} oracle / ` +
        `${s.n_agnostic_events || 0} agnostic / ${s.n_tsm_events} TSM`;
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
    renderMetrics(detail);
    renderGraph(detail);
    renderAgnosticEvents(detail);
    renderEvents(detail);
    renderTracks(detail);
    setupFrameViewer(detail).catch((err) =>
      setStatus(String(err.message || err), true)
    );
    setStatus(
      detail.video_exists
        ? `Sample ${detail.sample_id} · ${videoFile || detail.video_id} · ${detail.oracle_events.length} oracle · ${detail.tsm_events.length} TSM`
        : `Sample ${detail.sample_id} loaded, but video file is missing: ${detail.video_path || "(none)"}`,
      !detail.video_exists
    );
  }

  function fmtF1(f1) {
    return f1 == null || Number.isNaN(f1) ? "–" : Number(f1).toFixed(2);
  }

  function renderMetrics(detail) {
    const c = detail.comparison;
    if (!c || !c.summary) {
      el.metrics.innerHTML =
        `<div class="metric"><div class="label">Metrics</div>` +
        `<div class="value muted">n/a</div>` +
        `<div class="hint">TSM vs query-agnostic report missing for this video.</div></div>`;
      return;
    }
    const s = c.summary;
    const f1 = c.boundary_f1 || {};
    const fmtMean = (value) =>
      value == null || Number.isNaN(value) ? "–" : Number(value).toFixed(2);
    el.metrics.innerHTML = [
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
    const id = card.dataset.agnosticId || card.dataset.id || null;
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
    state.activeEventId = activeOracle ? activeOracle.event_id : null;
    state.activeAgnosticId = activeAgnostic ? activeAgnostic.event_id : null;

    // Prefer agnostic overlay when reviewing that track; fall back to oracle.
    const overlayEvent = activeAgnostic || activeOracle;
    if (overlayEvent) {
      const kind = activeAgnostic ? "agnostic" : "oracle";
      el.overlay.classList.remove("hidden");
      el.overlayTime.textContent =
        `${mmss(overlayEvent.start_sec)} – ${mmss(overlayEvent.end_sec)} · ` +
        `${kind} · ${overlayEvent.event_id}`;
      el.overlayDesc.textContent = overlayEvent.description;
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

    // Keep Gemini event cards in view while scrubbing frames.
    if (state.activeAgnosticId) {
      state.scrolledAgnosticId = scrollActiveCard(
        el.agnosticList,
        `#agnostic-list .event-card.active`,
        state.scrolledAgnosticId
      );
    } else {
      state.scrolledAgnosticId = null;
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

  function activeAgnosticAt(t) {
    const events = state.sample?.agnostic_events || [];
    return (
      events.find((e) => t >= e.start_sec && t < e.end_sec) ||
      events.find((e) => Math.abs(t - e.start_sec) < 0.25) ||
      null
    );
  }

  function renderAgnosticEvents(detail) {
    const events = detail.agnostic_events || [];
    el.agnosticList.innerHTML = "";
    state.scrolledAgnosticId = null;
    el.agnosticCount.textContent = events.length
      ? `(${events.length})`
      : "(none)";
    if (!events.length) {
      el.agnosticList.innerHTML =
        `<div class="muted" style="padding:4px 0">No query-agnostic annotation for this video.</div>`;
      return;
    }
    const metricsById = {};
    for (const ev of detail.comparison?.events || []) {
      metricsById[ev.event_id] = ev;
    }
    const sorted = [...events].sort(
      (a, b) => a.start_sec - b.start_sec || a.end_sec - b.end_sec
    );
    sorted.forEach((e, i) => {
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
      card.className = "event-card agnostic";
      card.dataset.agnosticId = e.event_id;
      card.innerHTML = `
        <div class="meta">
          <span>${i + 1}. ${escapeHtml(e.event_id)}</span>
          <span>${mmss(e.start_sec)} – ${mmss(e.end_sec)}</span>
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
      el.agnosticList.appendChild(card);
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
    const agnY = 44;
    const gemY = 72;
    const barH = 14;
    const height = 118;
    svg.setAttribute("viewBox", `0 0 ${w} ${height}`);

    const oracle = detail.oracle_events || [];
    const agnostic = detail.agnostic_events || [];
    const covering = new Set();
    detail.tsm_events.forEach((seg, i) => {
      if (
        agnostic.some((e) =>
          overlaps(seg.start_sec, seg.end_sec, e.start_sec, e.end_sec)
        )
      ) {
        covering.add(i);
      }
    });

    let markup = "";
    markup += `<text x="0" y="${tsmY + 11}" fill="#9aa3b5" font-size="11">TSM</text>`;
    markup += `<text x="0" y="${agnY + 11}" fill="#9aa3b5" font-size="11">Agnostic</text>`;
    markup += `<text x="0" y="${gemY + 11}" fill="#9aa3b5" font-size="11">Oracle</text>`;
    markup += `<rect x="${left}" y="${tsmY}" width="${trackW}" height="${barH}" fill="#222632" />`;
    markup += `<rect x="${left}" y="${agnY}" width="${trackW}" height="${barH}" fill="#222632" />`;
    markup += `<rect x="${left}" y="${gemY}" width="${trackW}" height="${barH}" fill="#222632" />`;

    detail.tsm_events.forEach((seg, i) => {
      const x = left + (seg.start_sec / duration) * trackW;
      const ww = Math.max(1, ((seg.end_sec - seg.start_sec) / duration) * trackW);
      const fill = covering.has(i) ? "#c4cad8" : "#6b7385";
      markup += `<rect x="${x}" y="${tsmY}" width="${ww}" height="${barH}" fill="${fill}" stroke="#0d0f14" stroke-width="0.6" data-t="${seg.start_sec}" />`;
    });

    agnostic.forEach((e, i) => {
      const x = left + (e.start_sec / duration) * trackW;
      const ww = Math.max(2, ((e.end_sec - e.start_sec) / duration) * trackW);
      markup += `<rect x="${x}" y="${agnY}" width="${ww}" height="${barH}" fill="#d4a017" stroke="#0d0f14" stroke-width="0.5" data-t="${e.start_sec}" rx="2" />`;
      if (ww >= 12) {
        markup += `<text x="${x + ww / 2}" y="${agnY + 11}" text-anchor="middle" fill="#0b1020" font-size="9" font-weight="600">${i + 1}</text>`;
      }
    });

    oracle.forEach((e, i) => {
      const x = left + (e.start_sec / duration) * trackW;
      const ww = Math.max(2, ((e.end_sec - e.start_sec) / duration) * trackW);
      const answer = e.query_roles.includes("answer_bearing");
      const fill = answer ? "#3ecf8e" : "#5b8cff";
      markup += `<rect x="${x}" y="${gemY}" width="${ww}" height="${barH}" fill="${fill}" data-t="${e.start_sec}" rx="2" />`;
      if (ww >= 14) {
        markup += `<text x="${x + ww / 2}" y="${gemY + 11}" text-anchor="middle" fill="#0b1020" font-size="10" font-weight="600">${i + 1}</text>`;
      }
    });

    // comparison cover brackets from report
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
