/* InsightFlow 单页界面：提交问题 → 订阅事件流 → 渲染报告。
 *
 * 状态都从服务端来：事件流负责「过程」，报告接口负责「结果」。
 * 断线重连由浏览器 EventSource 自带（会带 Last-Event-ID），
 * 我们只在收到终态事件时主动关闭，避免它反复重连一个已经结束的运行。
 */

const $ = (sel) => document.querySelector(sel);

const STATUS = {
  queued: ["排队中", ""],
  running: ["进行中", "run"],
  waiting_input: ["等待补充", "warn"],
  recoverable: ["可恢复", "warn"],
  completed: ["已完成", "ok"],
  partial: ["部分完成", "warn"],
  failed: ["失败", "bad"],
  cancelled: ["已取消", ""],
};

const STOP_REASON = {
  no_sources: "没有找到可用来源：检索有结果，但没有一篇正文能成功读取。",
  search_failed: "检索失败：可能是没有配置搜索 key，或上游搜索服务不可用。这不等于「没有新动态」。",
  model_fallback: "模型不可用，已退回模板生成：结构没问题，但结论措辞由模板拼装。",
  "budget:max_search_calls": "检索次数用完，提前结束。",
  "budget:max_source_fetches": "抓取次数用完，提前结束。",
  "budget:max_active_walltime_seconds": "运行时间超过上限，提前结束。",
  "budget:max_llm_calls": "模型调用次数用完，提前结束。",
};

const KIND = {
  official: "官方",
  paper: "论文",
  repo: "代码库",
  media: "媒体",
  community: "社区",
  public_data: "公开数据",
};

const ROLE = { official: "一手", original: "一手", reprint: "转载", secondary: "二手", unknown: "未标注" };

const SCOPE = { none: "未抓取", snippet: "仅摘要", partial: "部分正文", full: "全文" };

const state = { runId: null, stream: null, lastSeq: 0, running: false, markdown: "", searchConfigured: false };

/* ---------- 小工具 ---------- */

function hint(text, isError = false) {
  const el = $("#hint");
  el.textContent = text || "";
  el.classList.toggle("error", Boolean(isError) && Boolean(text));
}

function fmtClock(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString("zh-CN", { hour12: false });
}

function fmtDate(iso) {
  if (!iso) return "无发布时间";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "无发布时间";
  return d.toLocaleDateString("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" });
}

function newKey() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return `k-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function setStatus(el, status) {
  const [label, cls] = STATUS[status] || [status, ""];
  el.textContent = label;
  el.className = `status ${cls}`;
}

/* ---------- 提交与订阅 ---------- */

async function startRun(event) {
  event.preventDefault();
  const query = $("#query").value.trim();
  if (query.length < 2) {
    hint("问题太短，至少写两个字符", true);
    return;
  }
  const mode = $("#mode").value;
  const body = { query, days: Number($("#days").value), mode };
  if (mode === "replay") {
    const fixture = $("#fixture").value;
    if (fixture) body.fixture_id = fixture;
  }

  $("#btn-run").disabled = true;
  hint("正在提交…");
  try {
    const res = await fetch("/api/research/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": newKey() },
      body: JSON.stringify(body),
    });
    const payload = await res.json().catch(() => ({}));
    if (res.status === 409) {
      const existing = await findRun(query);
      if (existing) {
        hint("同一个问题已经跑过，直接打开上次的运行");
        attach(existing);
        return;
      }
      hint(payload.detail || "同一问题已存在运行", true);
      $("#btn-run").disabled = false;
      return;
    }
    if (!res.ok) {
      hint(payload.detail || `提交失败（HTTP ${res.status}）`, true);
      $("#btn-run").disabled = false;
      return;
    }
    hint("");
    attach(payload.run_id);
  } catch (err) {
    hint(`提交失败：${err.message}`, true);
    $("#btn-run").disabled = false;
  }
}

async function findRun(query) {
  try {
    const res = await fetch("/api/research/runs?limit=50");
    const payload = await res.json();
    const runs = (payload.data || []).filter(
      (r) => r.request && r.request.query === query && r.status !== "failed" && r.status !== "cancelled"
    );
    return runs.length ? runs[0].run_id : null;
  } catch {
    return null;
  }
}

function attach(runId) {
  closeStream();
  state.runId = runId;
  state.lastSeq = 0;
  state.running = true;
  state.markdown = "";
  $("#timeline").innerHTML = "";
  $("#progress").hidden = false;
  setStatus($("#status-pill"), "queued");
  $("#counter").textContent = "";
  $("#btn-cancel").disabled = false;
  $("#btn-run").disabled = true;
  resetReport();

  const stream = new EventSource(`/api/research/runs/${runId}/events`);
  state.stream = stream;
  stream.onmessage = (message) => {
    let event;
    try {
      event = JSON.parse(message.data);
    } catch {
      return;
    }
    if (typeof event.seq === "number") {
      if (event.seq <= state.lastSeq) return;
      state.lastSeq = event.seq;
    }
    handleEvent(event);
  };
  stream.onerror = () => {
    // 浏览器会自动重连；若已彻底关闭，则改用状态轮询兜底。
    if (stream.readyState === EventSource.CLOSED) scheduleStatusCheck(runId);
  };
}

function closeStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
}

function scheduleStatusCheck(runId) {
  window.setTimeout(async () => {
    if (!state.running || state.runId !== runId) return;
    const run = await fetchRun(runId);
    if (!run) return;
    if (STATUS[run.status] && ["completed", "partial", "failed", "cancelled"].includes(run.status)) {
      finish(run);
    } else {
      scheduleStatusCheck(runId);
    }
  }, 1500);
}

async function cancelRun() {
  if (!state.runId) return;
  $("#btn-cancel").disabled = true;
  hint("已请求取消，等它停在当前阶段…");
  try {
    await fetch(`/api/research/runs/${state.runId}/cancel`, { method: "POST" });
  } catch (err) {
    hint(`取消失败：${err.message}`, true);
  }
}

/* ---------- 事件渲染 ---------- */

function handleEvent(event) {
  const data = event.data || {};
  const li = document.createElement("li");
  const view = describe(event);
  li.className = view.tone || "";

  const time = document.createElement("span");
  time.className = "t-time";
  time.textContent = fmtClock(event.created_at);

  const title = document.createElement("span");
  title.className = "t-title";
  title.textContent = view.title;
  title.appendChild(time);
  li.appendChild(title);

  (view.subs || []).forEach((text) => {
    const sub = document.createElement("span");
    sub.className = "t-sub";
    sub.textContent = text;
    li.appendChild(sub);
  });

  if (view.chips && view.chips.length) {
    const wrap = document.createElement("div");
    wrap.className = "chips";
    view.chips.forEach((text) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = text;
      wrap.appendChild(chip);
    });
    li.appendChild(wrap);
  }

  $("#timeline").appendChild(li);
  const timeline = $("#timeline");
  timeline.scrollTop = timeline.scrollHeight;

  if (["run_started", "plan_ready", "search_completed", "source_fetched", "synthesis_ready"].includes(event.type)) {
    setStatus($("#status-pill"), "running");
  }
  if (["run_finished", "run_cancelled", "run_failed"].includes(event.type)) {
    li.classList.add("term");
    const status = data.status || (event.type === "run_cancelled" ? "cancelled" : "");
    if (status) setStatus($("#status-pill"), status);
    finishFromEvent(event, status);
  }
}

function describe(event) {
  const d = event.data || {};
  switch (event.type) {
    case "run_queued":
      return {
        title: "已排队",
        subs: [`模式 ${d.mode === "replay" ? "回放快照" : "在线检索"}`, `窗口 ${fmtDate(d.window_start)} → ${fmtDate(d.window_end)}`],
      };
    case "run_started":
      return { title: "开始执行", tone: "done" };
    case "plan_ready":
      return {
        title: `规划了 ${(d.queries || []).length} 条检索词`,
        tone: "done",
        chips: d.queries || [],
      };
    case "search_completed":
      return {
        title: `检索「${d.query}」`,
        tone: "done",
        subs: [`命中 ${d.hits ?? 0} 条，新增 ${d.new_hits ?? 0} 条${d.deduped ? `，去重 ${d.deduped} 条` : ""}`],
      };
    case "search_failed":
      return { title: `检索「${d.query}」失败`, tone: "warn", subs: [`错误码 ${d.error_code}`] };
    case "source_fetched":
      return {
        title: `读到了「${d.title || d.url}」`,
        tone: "done",
        subs: [`${fmtDate(d.published_at)} · ${KIND[d.kind] || d.kind || "来源"} · ${ROLE[d.role] || d.role || "未标注"} · ${SCOPE[d.fetch_scope] || d.fetch_scope || ""}`],
      };
    case "source_failed":
      return {
        title: "这一篇没读成",
        tone: "warn",
        subs: [`${d.url}`, `错误码 ${d.error_code}${d.retryable ? "（可重试）" : ""}`],
      };
    case "decision":
      return {
        title: d.action === "supplement" ? "模型决定：继续补搜" : "模型决定：停止补搜",
        tone: d.action === "supplement" ? "info" : "done",
        subs: [
          d.reason,
          d.decided_by === "fallback" ? "模型决策失败，已退回规则决策" : "",
        ].filter(Boolean),
        chips: d.queries || [],
      };
    case "gaps_found": {
      const gaps = d.gaps || [];
      if (!gaps.length) {
        return { title: "覆盖检查通过", tone: "done", subs: ["用户关心的主题都有来源"] };
      }
      return {
        title: `发现 ${gaps.length} 处覆盖缺口`,
        tone: "warn",
        subs: gaps.map((gap) => gap.reason),
      };
    }
    case "supplement_started":
      return {
        title: `第 ${d.round} 轮补搜`,
        tone: "info",
        subs: d.reasons || [],
        chips: d.queries || [],
      };
    case "supplement_completed":
      return {
        title: `补搜带回 ${d.new_sources ?? 0} 个可读来源`,
        tone: "done",
        subs: [`尝试 ${d.attempted ?? 0} 个，现有来源 ${d.total_sources ?? 0} 个`],
      };
    case "supplement_stopped":
      return {
        title: d.reason === "limit" ? "补搜达到上限" : d.reason === "budget" ? "预算用完，停止补搜" : "补搜没有新来源",
        tone: "warn",
        subs: [d.reason === "no_new_sources" ? "再查也没有新页面，按现有资料写" : "剩余缺口会写进报告"],
      };
    case "synthesis_ready":
      return {
        title: `归纳出 ${d.findings ?? 0} 条结论`,
        tone: "done",
        subs: d.degraded ? ["模型调用失败，已用模板兜底"] : [],
      };
    case "report_ready":
      return { title: "报告已生成", tone: "done", subs: [`状态 ${STATUS[d.status] ? STATUS[d.status][0] : d.status}`] };
    case "run_finished":
      return {
        title: "运行结束",
        subs: [d.stop_reason ? STOP_REASON[d.stop_reason] || `结束原因 ${d.stop_reason}` : "正常完成"],
      };
    case "run_cancelled":
      return { title: "已取消", subs: ["停在当前阶段，未生成报告"] };
    case "run_failed":
      return { title: "运行失败", tone: "bad", subs: [d.error || "未知错误"] };
    default:
      return { title: event.type, subs: [] };
  }
}

/* ---------- 结束与报告 ---------- */

function finishFromEvent(event, status) {
  closeStream();
  const data = event.data || {};
  if (data.counters) {
    const c = data.counters;
    $("#counter").textContent = `检索 ${c.search_calls ?? 0} · 抓取 ${c.source_fetches ?? 0} · 模型 ${c.llm_calls ?? 0}`;
  }
  state.running = false;
  $("#btn-cancel").disabled = true;
  $("#btn-run").disabled = false;
  loadReport(state.runId, data.stop_reason || null);
}

function finish(run) {
  closeStream();
  if (run.counters) {
    $("#counter").textContent = `检索 ${run.counters.search_calls ?? 0} · 抓取 ${run.counters.source_fetches ?? 0} · 模型 ${run.counters.llm_calls ?? 0}`;
  }
  state.running = false;
  $("#btn-cancel").disabled = true;
  $("#btn-run").disabled = false;
  setStatus($("#status-pill"), run.status);
  loadReport(run.run_id, run.stop_reason || null);
}

function resetReport() {
  $("#report-body").hidden = true;
  $("#report-body").innerHTML = "";
  $("#report-toolbar").hidden = true;
  $("#report-extra").hidden = true;
  $("#empty-state").hidden = true;
  $("#banner")?.remove();
}

function showBanner(text) {
  $("#banner")?.remove();
  const banner = document.createElement("div");
  banner.id = "banner";
  banner.className = "banner";
  banner.textContent = text;
  $("#report-toolbar").insertAdjacentElement("afterend", banner);
}

async function fetchRun(runId) {
  try {
    const res = await fetch(`/api/research/runs/${runId}`);
    if (!res.ok) return null;
    const payload = await res.json();
    return payload.data;
  } catch {
    return null;
  }
}

async function loadReport(runId, stopReason) {
  const detail = await fetchRun(runId);
  const run = detail?.run || null;
  const sources = detail?.sources || [];

  $("#report-toolbar").hidden = false;
  $("#empty-state").hidden = true;
  if (run) {
    setStatus($("#report-status"), run.status);
    const bits = [];
    if (run.report_id) bits.push(`报告 ${run.report_id}`);
    if (run.request) bits.push(`窗口 ${fmtDate(run.request.window_start)} → ${fmtDate(run.request.window_end)}`);
    $("#report-meta").textContent = bits.join(" · ");
  }

  if (stopReason && STOP_REASON[stopReason]) {
    showBanner(STOP_REASON[stopReason]);
  }

  if (!run || run.status === "cancelled") {
    if (run && run.status === "cancelled") {
      $("#report-body").hidden = false;
      $("#report-body").innerHTML = "<p>这次运行被取消了，没有生成报告。需要的话重新点一次「开始研究」。</p>";
    }
    return;
  }

  let payload = null;
  try {
    const res = await fetch(`/api/research/runs/${runId}/report`);
    if (res.ok) payload = await res.json();
  } catch {
    /* 下面统一处理 */
  }

  if (!payload) {
    $("#report-body").hidden = false;
    $("#report-body").innerHTML = "<p>这次没有生成报告。</p>";
    if (stopReason && !STOP_REASON[stopReason]) showBanner(`结束原因：${stopReason}`);
    return;
  }

  const report = payload.data.report;
  state.markdown = payload.data.markdown || "";
  $("#report-body").innerHTML = payload.data.html || "<p>（报告为空）</p>";
  $("#report-body").hidden = false;
  $("#report-extra").hidden = false;
  setStatus($("#report-status"), report.status === "published" ? "completed" : "partial");
  renderChecks(report.checks || []);
  renderSources(sources, report);
}

function renderChecks(checks) {
  const box = $("#checks");
  box.innerHTML = "";
  if (!checks.length) return;
  checks.forEach((check) => {
    const span = document.createElement("span");
    span.className = `check ${check.status}`;
    span.textContent = `${check.status === "pass" ? "✓" : check.status === "warn" ? "!" : "✕"} ${check.check}`;
    span.title = check.detail || "";
    box.appendChild(span);
  });
}

function renderSources(sources, report) {
  const box = $("#sources");
  box.innerHTML = "";
  if (!sources.length) return;

  const title = document.createElement("h4");
  title.textContent = `来源（${sources.length}）`;
  box.appendChild(title);

  sources.forEach((source, index) => {
    const row = document.createElement("div");
    row.className = source.status === "fetched" ? "source" : "source failed";

    const idx = document.createElement("span");
    idx.className = "idx";
    idx.textContent = String(index + 1).padStart(2, "0");

    const link = document.createElement("a");
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    link.textContent = source.title || source.url;

    const meta = document.createElement("span");
    meta.className = "meta";
    if (source.status === "fetched") {
      meta.textContent = `${fmtDate(source.published_at)} · ${SCOPE[source.fetch_scope] || source.fetch_scope} · ${ROLE[source.role] || source.role}`;
    } else {
      meta.textContent = `读取失败（${source.error_code || "unknown"}）`;
    }

    row.append(idx, link, meta);
    box.appendChild(row);
  });

  if (report && report.limitations && report.limitations.length) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = `本条报告承认的局限：${report.limitations.join("；")}`;
    note.style.marginTop = "10px";
    box.appendChild(note);
  }
}

function downloadMarkdown() {
  if (!state.markdown) return;
  const blob = new Blob([state.markdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${state.runId || "report"}.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/* ---------- 初始化 ---------- */

async function loadMeta() {
  try {
    const res = await fetch("/api/health");
    const payload = await res.json();
    const data = payload.data || {};
    state.searchConfigured = Boolean(data.search_configured);
    const search = $("#pill-search");
    if (data.search_configured) {
      search.textContent = "搜索：已配置";
      search.className = "pill ok";
    } else {
      search.textContent = "搜索：未配置 key（用回放）";
      search.className = "pill warn";
    }
    const model = $("#pill-model");
    if (data.model_configured) {
      model.textContent = "模型：已配置";
      model.className = "pill ok";
    } else {
      model.textContent = "模型：模板模式";
      model.className = "pill warn";
    }
  } catch {
    $("#pill-search").textContent = "搜索：无法连接";
    $("#pill-model").textContent = "模型：无法连接";
  }

  try {
    const res = await fetch("/api/fixtures");
    const payload = await res.json();
    const fixtures = payload.data || [];
    const select = $("#fixture");
    select.innerHTML = "";
    fixtures.forEach((fixture) => {
      const option = document.createElement("option");
      option.value = fixture.id;
      option.textContent = `${fixture.id}（${fixture.items} 个页面${fixture.synthetic ? "，虚构示例" : ""}）`;
      select.appendChild(option);
    });
    if (!fixtures.length) {
      $("#fixture-field").hidden = true;
      hint("没有找到回放资料，请先运行 make replay 生成，或改用在线检索", true);
    }
  } catch {
    $("#fixture-field").hidden = true;
  }
}

function syncMode() {
  $("#fixture-field").hidden = $("#mode").value !== "replay";
}

document.addEventListener("DOMContentLoaded", () => {
  $("#form").addEventListener("submit", startRun);
  $("#btn-cancel").addEventListener("click", cancelRun);
  $("#btn-download").addEventListener("click", downloadMarkdown);
  $("#btn-help").addEventListener("click", () => $("#help").showModal());
  $("#mode").addEventListener("change", syncMode);
  document.querySelectorAll(".chip[data-q]").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("#query").value = chip.dataset.q;
      $("#query").focus();
    });
  });
  loadMeta();
  syncMode();

  // 深链：`/?run=<run_id>` 直接打开一次运行（事件会从头重放，历史完整）。
  const params = new URLSearchParams(window.location.search);
  const sharedRun = params.get("run");
  const sharedQuery = params.get("q");
  if (sharedQuery) $("#query").value = sharedQuery;
  if (sharedRun) attach(sharedRun);
});