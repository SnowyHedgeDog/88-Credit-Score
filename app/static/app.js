/* RemitSentry analyst app — single-page front end (no build step). */
import { scatterTransfers, flowGraph } from "./charts.js";

const $ = (sel, root = document) => root.querySelector(sel);
const view = $("#view");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const money = (x, dp = 0) => (x == null ? "—" : "S$" + Number(x).toLocaleString("en-SG", { minimumFractionDigits: dp, maximumFractionDigits: dp }));
const short = (x) => {
  if (x == null) return "—";
  const a = Math.abs(x);
  if (a >= 1e6) return (x / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e3) return (x / 1e3).toFixed(a >= 1e4 ? 0 : 1) + "K";
  return Math.round(x).toString();
};
const pct = (x, dp = 1) => (x == null ? "—" : (x * 100).toFixed(dp) + "%");
const dt = (iso, opts = { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }) =>
  iso ? new Date(iso).toLocaleString("en-SG", opts) : "—";
const dday = (iso) => dt(iso, { day: "numeric", month: "short", year: "numeric" });
const STATUS = { critical: "Critical", watch: "Watch", clear: "Clear", external: "Outside our book" };
const plural = (n, one, many = one + "s") => `${Number(n).toLocaleString()} ${n === 1 ? one : many}`;

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (e) { /* no body */ }
  if (!res.ok) throw new Error(body?.detail || `Request failed (${res.status})`);
  return body;
}

function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast"; t.setAttribute("role", "status"); t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2600);
}

function ladder(score, status, size = "") {
  let bars = "";
  for (let i = 1; i <= 10; i++) bars += `<span class="${i <= score ? "on" : ""}" style="height:${30 + i * 7}%"></span>`;
  return `<span class="ladder on-${status} ${size}" aria-label="Score ${score} of 10">${bars}</span>`;
}
const pill = (status) => `<span class="pill"><i style="background:var(--${status === "external" ? "external" : status})"></i>${STATUS[status]}</span>`;
const dueChip = (days) => days == null ? "" : days < 0
  ? `<span class="due late">${-days} day${days === -1 ? "" : "s"} overdue</span>`
  : `<span class="due ${days <= 1 ? "late" : ""}">${days === 0 ? "Due today" : plural(days, "day") + " left"}</span>`;
const DECISION = { escalated: "Escalated", filed: "STR filed", cleared: "Cleared" };
const decisionTag = (d) => d ? `<span class="decision ${d}">${DECISION[d]}</span>` : "";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = { batches: [], batchId: null, summary: null, accountsIndex: null };
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};

async function refreshBatches(selectId) {
  state.batches = await api("/api/batches");
  const wanted = selectId || state.batchId || store.get("remitsentry-batch");
  state.batchId = state.batches.find((b) => b.id === wanted) ? wanted : state.batches[0]?.id || null;
  if (state.batchId) store.set("remitsentry-batch", state.batchId);
  state.summary = state.batchId ? await api(`/api/batches/${state.batchId}/summary`) : null;
  state.accountsIndex = null;
  renderStrip();
}

async function accountsIndex() {
  if (!state.accountsIndex && state.batchId) {
    state.accountsIndex = (await api(`/api/batches/${state.batchId}/accounts?limit=100000`)).rows;
  }
  return state.accountsIndex || [];
}

function renderStrip() {
  const strip = $("#strip");
  if (!state.batchId) { strip.hidden = true; return; }
  strip.hidden = false;
  $("#batch-select").innerHTML = state.batches.map((b) =>
    `<option value="${b.id}" ${b.id === state.batchId ? "selected" : ""}>${esc(b.name)} (${dt(b.scored_at, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false })})</option>`).join("");
  const m = state.summary.meta;
  $("#counts").innerHTML = ["critical", "watch", "clear"].map((s) =>
    `<a class="count" href="#/queue?min=${s === "critical" ? 7 : s === "watch" ? 4 : 1}"><i style="background:var(--${s})"></i><b>${m[s]}</b>${s}</a>`).join("") +
    `<span class="count">of ${m.accounts} accounts</span>`;
  tick();
}

function tick() {
  const el = $("#clock");
  if (!el || !state.summary) return;
  const now = new Date();
  const time = now.toLocaleTimeString("en-SG", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "Asia/Singapore" });
  const day = now.toLocaleDateString("en-SG", { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "Asia/Singapore" });
  const s = state.summary;
  el.innerHTML = `<b>${time}</b> <span class="muted">SGT</span><small>${day}</small><small>STRs from this batch due ${dday(s.str_due)}</small>`;
}
setInterval(tick, 20000);

$("#batch-select").addEventListener("change", async (e) => {
  await refreshBatches(e.target.value);
  route();
});

$("#theme").addEventListener("click", () => {
  const root = document.documentElement;
  const next = root.dataset.theme === "light" ? "dark" : "light";
  root.dataset.theme = next; store.set("remitsentry-theme", next);
  $("#theme").textContent = next === "light" ? "Dark" : "Light";
  route();
});
$("#theme").textContent = document.documentElement.dataset.theme === "light" ? "Dark" : "Light";

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------
function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "") || "batches";
  const [path, query] = raw.split("?");
  const parts = path.split("/").map(decodeURIComponent);
  return { page: parts[0], arg: parts[1], params: new URLSearchParams(query || "") };
}

let routeToken = 0;
async function route() {
  const { page, arg, params } = parseHash();
  const token = ++routeToken;
  document.querySelectorAll("nav.tabs a").forEach((a) => { if (a.dataset.page !== page) a.removeAttribute("aria-current"); else a.setAttribute("aria-current", "page"); });
  closeDrawer();
  const pages = { batches: pageBatches, queue: pageQueue, account: pageAccount, flow: pageFlow };
  const fn = pages[page] || pageBatches;
  if (!state.batchId && page !== "batches") {
    view.innerHTML = emptyState();
    return;
  }
  try {
    await fn(arg, params, token);
  } catch (e) {
    if (token !== routeToken) return;
    view.innerHTML = `<div class="notice error"><b>That page could not load.</b> ${esc(e.message)}</div>`;
  }
  if (token === routeToken) window.scrollTo({ top: 0 });
}
window.addEventListener("hashchange", route);
const go = (hash) => { if (location.hash === hash) route(); else location.hash = hash; };

const emptyState = () => `<div class="empty"><h2>No batch scored yet</h2>
  <p>Score a batch of transactions to fill the alert queue.</p>
  <a class="btn primary" href="#/batches">Go to batches</a></div>`;

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------
async function pageBatches() {
  const s = state.summary;
  const m = s?.meta;
  view.innerHTML = `
    <div class="page-head">
      <h1>Batches</h1>
      <p class="lede">A batch is one upload of settled transactions, usually the previous day's. The model scores every transfer, and every customer account gets a score from 1 to 10 with the reasons behind it.</p>
    </div>
    ${m ? `
      ${m.source === "synthetic" ? `<div class="note">You're viewing the synthetic sample batch. It uses the same file format as the team's data, with laundering patterns planted so the demo has known answers.</div>` : ""}
      ${m.source === "real" && m.description ? `<div class="note">${esc(m.description)}</div>` : ""}
      <div class="notice">
        <b>${esc(m.name)}</b> scored ${esc(m.scored_at_label)}.
        ${m.critical ? `${plural(m.critical, "account")} ${m.critical === 1 ? "is" : "are"} critical and waiting in the alert queue. File any confirmed STRs by ${dday(s.str_due)}.` : "No account is critical."}
        <div class="row" style="margin-top:10px">
          <a class="btn primary" href="#/queue">Open the alert queue</a>
          <span class="muted">${plural(m.transactions, "transfer")} from ${dday(m.from)} to ${dday(m.to)}, ${plural(m.model_alerts, "model alert")} at threshold ${m.threshold.toFixed(3)}</span>
        </div>
        ${m.notes?.length ? `<p class="muted" style="margin:8px 0 0">${m.notes.map(esc).join("<br>")}</p>` : ""}
      </div>
      <section class="section" id="versus"></section>` : ""}
    <section class="section">
      <h2>Score a new batch</h2>
      <form id="upload" class="stack">
        <div class="grid-2">
          ${dropZone("transactions", "Transactions (CSV, required)", "txn_timestamp, sender_account_key, receiver_account_key, amount_sgd, currency_sent, currency_received, payment_method")}
          ${dropZone("customers", "Customer profiles (CSV, required)", "account_key, customer_type, occupation_or_industry, employment_status, monthly_income_sgd, expected_monthly_amount_sgd, expected_monthly_txn_count, expects_international_activity, account_open_channel, account_open_date")}
        </div>
        <div class="row">
          <label class="field" style="flex:1;min-width:240px"><span>Batch name</span>
            <input type="text" name="name" placeholder="Daily batch, ${new Date().toLocaleDateString("en-SG", { day: "numeric", month: "short", year: "numeric" })}">
          </label>
          <div class="row" style="align-self:flex-end">
            <button class="btn primary" type="submit" id="score-btn" disabled>Score batch</button>
            <button class="btn" type="button" id="demo-btn">Load sample batch</button>
          </div>
        </div>
        <div id="upload-msg"></div>
      </form>
    </section>
    <section class="section">
      <h2>Batch log (${state.batches.length})</h2>
      ${state.batches.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Batch</th><th>Scored at</th><th class="r">Accounts</th><th class="r">Transfers</th><th class="r">Critical</th><th class="r">Watch</th><th></th></tr></thead>
        <tbody>${state.batches.map((b) => `<tr>
          <td><b>${esc(b.name)}</b>${b.id === state.batchId ? ' <span class="faint">(viewing)</span>' : ""}<br>
            <span class="faint">${dday(b.from)} to ${dday(b.to)}, ${b.source === "synthetic" ? "synthetic sample data" : b.source === "real" ? "real held-out test data" : "uploaded"}</span></td>
          <td class="num">${esc(b.scored_at_label)}</td>
          <td class="r num">${b.accounts.toLocaleString()}</td>
          <td class="r num">${b.transactions.toLocaleString()}</td>
          <td class="r num critical">${b.critical}</td>
          <td class="r num watch">${b.watch}</td>
          <td class="r"><div class="row" style="justify-content:flex-end;flex-wrap:nowrap">
            <button class="btn small" data-open="${b.id}" ${b.id === state.batchId ? "disabled" : ""}>Open</button>
            <button class="btn small danger" data-del="${b.id}">Delete</button></div></td>
        </tr>`).join("")}</tbody></table></div>`
      : `<div class="empty"><p>No batches yet. Load the sample batch or upload two CSV files above.</p></div>`}
    </section>`;

  if (m) renderVersus(s.rules);
  wireUpload();
  view.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", async () => {
    await refreshBatches(b.dataset.open); route();
  }));
  view.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
    const batch = state.batches.find((x) => x.id === b.dataset.del);
    if (!confirm(`Delete "${batch.name}"? Its scores and analyst decisions will be removed.`)) return;
    await api(`/api/batches/${b.dataset.del}`, { method: "DELETE" });
    toast("Batch deleted");
    await refreshBatches(); route();
  }));
}

function dropZone(name, label, cols) {
  return `<div class="field"><span>${label}</span>
    <label class="drop" data-drop="${name}">
      <input type="file" name="${name}" accept=".csv,text/csv">
      <span class="btn small" aria-hidden="true">Choose file</span>
      <span class="fname muted">or drop a CSV here</span>
    </label>
    <details class="faint" style="font-size:12.5px"><summary>Columns it needs</summary>${cols}</details></div>`;
}

function wireUpload() {
  const form = $("#upload");
  const scoreBtn = $("#score-btn");
  const update = () => { scoreBtn.disabled = !(form.transactions.files.length && form.customers.files.length); };
  form.querySelectorAll(".drop").forEach((zone) => {
    const input = zone.querySelector("input");
    const show = () => { zone.querySelector(".fname").textContent = input.files[0]?.name || "or drop a CSV here"; zone.querySelector(".fname").classList.toggle("muted", !input.files.length); update(); };
    input.addEventListener("change", show);
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("over"); input.files = e.dataTransfer.files; show(); });
  });
  const busy = (btn, on, text) => { btn.disabled = on; btn.innerHTML = on ? `<span class="spinner"></span> ${text}` : btn.dataset.label; };
  scoreBtn.dataset.label = scoreBtn.textContent;
  const demo = $("#demo-btn"); demo.dataset.label = demo.textContent;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    busy(scoreBtn, true, "Scoring…");
    $("#upload-msg").innerHTML = "";
    try {
      const fd = new FormData(form);
      const meta = await api("/api/batches", { method: "POST", body: fd });
      await refreshBatches(meta.id);
      toast("Batch scored");
      route();
    } catch (err) {
      $("#upload-msg").innerHTML = `<div class="notice error"><b>The batch was not scored.</b> ${esc(err.message)}</div>`;
      busy(scoreBtn, false);
    }
  });
  demo.addEventListener("click", async () => {
    busy(demo, true, "Scoring sample batch…");
    try {
      const meta = await api("/api/batches/demo", { method: "POST" });
      await refreshBatches(meta.id);
      toast("Sample batch scored");
      route();
    } catch (err) {
      $("#upload-msg").innerHTML = `<div class="notice error"><b>The sample batch was not scored.</b> ${esc(err.message)}</div>`;
      busy(demo, false);
    }
  });
}

function renderVersus(r) {
  const el = $("#versus");
  const labelled = r.has_labels;
  el.innerHTML = `
    <h2>Flat rule vs RemitSentry on this batch</h2>
    <p class="lede" style="margin-bottom:14px">Small firms often flag any account that crosses a fixed amount or transfer count. Same batch, same accounts:</p>
    <div class="versus">
      <div>
        <div class="muted">Flat rule: any transfer of
          <input type="number" id="rule-amt" value="${r.amount}" min="1000" step="1000" style="width:110px;min-height:28px;padding:3px 8px"> or more,
          or over <input type="number" id="rule-cnt" value="${r.count}" min="1" step="1" style="width:64px;min-height:28px;padding:3px 8px"> transfers</div>
        <div class="big num" style="margin-top:12px">${r.rule_alerts}</div><div class="muted">accounts to review</div>
        ${labelled ? `
          <div class="line"><span>Laundering accounts caught</span><b class="num">${r.rule_caught} of ${r.labelled_laundering}</b></div>
          <div class="line"><span>False alarms</span><b class="num critical">${r.rule_false}</b></div>` : ""}
      </div>
      <div>
        <div class="muted">RemitSentry: accounts scoring 7 or more</div>
        <div class="big num" style="margin-top:12px">${r.app_alerts}</div><div class="muted">accounts to review</div>
        ${labelled ? `
          <div class="line"><span>Laundering accounts caught</span><b class="num">${r.app_caught} of ${r.labelled_laundering}</b></div>
          <div class="line"><span>False alarms</span><b class="num ${r.app_false ? "critical" : "clear"}">${r.app_false}</b></div>` : ""}
      </div>
    </div>
    <p class="faint" style="margin-top:8px">${r.both} account${r.both === 1 ? "" : "s"} flagged by both. ${labelled ? "Laundering labels come with the batch; analysts would not have them in production." : "This batch has no laundering labels, so only alert counts are compared."}</p>`;
  const refetch = async () => {
    const amt = Number($("#rule-amt").value) || 20000, cnt = Number($("#rule-cnt").value) || 30;
    const r2 = await api(`/api/batches/${state.batchId}/rules?amount=${amt}&count=${cnt}`);
    renderVersus(r2);
  };
  $("#rule-amt").addEventListener("change", refetch);
  $("#rule-cnt").addEventListener("change", refetch);
}

// ---------------------------------------------------------------------------
// Alert queue
// ---------------------------------------------------------------------------
async function pageQueue(_arg, params, token) {
  const s = state.summary;
  const m = s.meta;
  const f = {
    min: Number(params.get("min") || 1),
    types: params.get("types") ?? "Business,Retail",
    reason: params.get("reason") || "",
    q: params.get("q") || "",
  };
  const pctCrit = m.accounts ? (m.critical / m.accounts * 100).toFixed(1) : 0;
  const maxCount = Math.max(...s.histogram.map((h) => h.count), 1);
  view.innerHTML = `
    <div class="page-head">
      <h1>Alert queue</h1>
      <p class="lede">Work from the top. ${m.critical} of ${m.accounts} accounts (${pctCrit}%) scored 7 or above and need an analyst; the rest can wait or be cleared. Click a row to open the account.</p>
    </div>
    <div class="grid-main" style="align-items:end">
      <div>
        <div class="muted" style="margin-bottom:22px">Accounts by score</div>
        <div class="hist">${s.histogram.map((h) => `<div title="Score ${h.score}: ${h.count} accounts" style="height:${Math.max(2, h.count / maxCount * 100)}%;background:var(--${h.score >= s.critical_at ? "critical" : h.score >= s.watch_at ? "watch" : "clear"})"><span>${h.count || ""}</span></div>`).join("")}</div>
        <div class="hist-axis">${s.histogram.map((h) => `<span>${h.score}</span>`).join("")}</div>
      </div>
      <div class="stack">
        <div class="row" style="align-items:flex-end">
          <label class="field" style="flex:1;min-width:180px"><span>Minimum score: <b id="min-label">${f.min}</b></span>
            <input type="range" id="f-min" min="1" max="10" value="${f.min}"></label>
          <div class="field"><span>Customer</span><div class="chips">
            ${["Business", "Retail"].map((t) => `<button type="button" class="chip-toggle" data-type="${t}" aria-pressed="${f.types.split(",").includes(t)}">${t}</button>`).join("")}</div></div>
        </div>
        <div class="row">
          <label class="field" style="flex:1;min-width:200px"><span>Reason</span>
            <select id="f-reason"><option value="">Any reason</option>
              ${s.reasons.map((r) => `<option value="${r.code}" ${r.code === f.reason ? "selected" : ""}>${esc(r.title)} (${r.count})</option>`).join("")}</select></label>
          <label class="field" style="flex:1;min-width:200px"><span>Search</span>
            <input type="search" id="f-q" value="${esc(f.q)}" placeholder="Account number"></label>
        </div>
      </div>
    </div>
    <div class="section" style="margin-top:22px">
      <div class="muted" id="q-count" style="margin-bottom:8px"></div>
      <div class="table-wrap" style="max-height:70vh"><table>
        <thead><tr><th>Risk</th><th class="r">Score</th><th>Status</th><th>Account</th><th class="name-col">Customer</th><th>Type</th><th>Reasons</th><th>STR due</th><th class="r">Transfers</th><th class="r">In</th><th class="r">Out</th></tr></thead>
        <tbody id="q-rows"></tbody></table></div>
    </div>`;

  const sync = () => {
    const p = new URLSearchParams({ min: f.min, types: f.types, reason: f.reason, q: f.q });
    history.replaceState(null, "", `#/queue?${p}`);
    load();
  };
  let timer;
  $("#f-min").addEventListener("input", (e) => { f.min = Number(e.target.value); $("#min-label").textContent = f.min; clearTimeout(timer); timer = setTimeout(sync, 120); });
  view.querySelectorAll("[data-type]").forEach((b) => b.addEventListener("click", () => {
    b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") !== "true");
    f.types = [...view.querySelectorAll("[data-type][aria-pressed=true]")].map((x) => x.dataset.type).join(",");
    sync();
  }));
  $("#f-reason").addEventListener("change", (e) => { f.reason = e.target.value; sync(); });
  $("#f-q").addEventListener("input", (e) => { f.q = e.target.value; clearTimeout(timer); timer = setTimeout(sync, 200); });

  async function load() {
    const p = new URLSearchParams({ min_score: f.min, types: f.types || "none", reason: f.reason, q: f.q });
    const data = await api(`/api/batches/${state.batchId}/accounts?${p}`);
    if (token !== routeToken) return;
    $("#q-count").textContent = `Showing ${data.rows.length} of ${data.total} matching accounts`;
    const named = data.rows.some((r) => r.name);
    $("#q-rows").innerHTML = data.rows.length ? data.rows.map((r) => `
      <tr class="click ${r.decision === "cleared" || r.decision === "filed" ? "done" : ""}" data-key="${esc(r.key)}" tabindex="0">
        <td>${ladder(r.score, r.status)}</td>
        <td class="r num"><b>${r.score}</b></td>
        <td>${pill(r.status)}</td>
        <td class="num" style="white-space:nowrap">${esc(r.key)}</td>
        <td class="name-col">${esc(r.name)} ${decisionTag(r.decision)}</td>
        <td>${esc(r.type)}</td>
        <td>${r.reasons.slice(0, 3).map((x) => `<span class="tag">${esc(x.title)}</span>`).join("")}${r.reasons.length > 3 ? `<span class="faint">+${r.reasons.length - 3}</span>` : ""}</td>
        <td>${r.decision === "filed" ? "Filed" : dueChip(r.str_days_left)}</td>
        <td class="r num">${r.txns}</td>
        <td class="r num">${short(r.in_sgd)}</td>
        <td class="r num">${short(r.out_sgd)}</td>
      </tr>`).join("") : `<tr><td colspan="11" class="empty">No accounts match these filters. Lower the minimum score or clear the search.</td></tr>`;
    view.querySelectorAll(".name-col").forEach((el) => { el.hidden = !named; });
    if (!named) view.querySelectorAll("tr[data-key]").forEach((tr, i) => {
      const d = data.rows[i].decision;
      if (d) tr.children[3].insertAdjacentHTML("beforeend", " " + decisionTag(d));
    });
    view.querySelectorAll("tr[data-key]").forEach((tr) => {
      const open = () => go(`#/account/${encodeURIComponent(tr.dataset.key)}`);
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    });
  }
  await load();
}

// ---------------------------------------------------------------------------
// Account
// ---------------------------------------------------------------------------
const ACTION = {
  critical: "Escalate to an investigator. If the activity is confirmed, file a suspicious transaction report with STRO.",
  watch: "Review when there is capacity. Check the counterparties and the customer's stated purpose.",
  clear: "No action needed. The account stays in monitoring.",
};

function hero(a, compact = false) {
  // Bars 1..10. Bar 1 is the base point; then each reason fills its points.
  const segs = [{ label: "base", pts: 1 }].concat(a.reasons.map((r) => ({ label: r.title, pts: r.points })));
  const key = [];
  let at = 0;
  for (const s of segs) {
    if (at >= 10) break;
    const span = Math.min(s.pts, 10 - at);
    key.push(`<div style="grid-column:${at + 1} / span ${span}" title="${esc(s.label)}"><b>${s.label === "base" ? "base" : "+" + s.pts}</b> ${s.label === "base" ? "" : esc(compact ? "" : s.label)}</div>`);
    at += span;
  }
  const bars = Array.from({ length: 10 }, (_, i) => `<div style="height:${28 + i * 8}%;${i < a.score ? `background:var(--${a.status})` : ""}"></div>`).join("");
  const over = a.raw_points > 10 ? `<p class="faint" style="margin:6px 0 0;font-size:12.5px">The reasons add up to ${a.raw_points}; scores stop at 10.</p>` : "";
  return `
    <div class="score-hero">
      <div class="big-score ${a.status}" style="${compact ? "font-size:96px" : ""}">${a.score}<small>/10</small></div>
      <div><div class="big-ladder" style="${compact ? "height:84px" : ""}">${bars}</div>
        <div class="ladder-key">${key.join("")}</div>${over}</div>
    </div>
    <h2 style="font-size:28px;margin:18px 0 2px">${esc(a.name || a.key)}</h2>
    <div class="muted">${a.name ? esc(a.key) + ", " + esc(a.type.toLowerCase()) : esc(a.type)} account opened ${esc(a.profile["Account opened"])}</div>
    <div class="action-box ${a.status}"><p><b class="${a.status}">${STATUS[a.status]}.</b> ${ACTION[a.status]}
      ${a.status === "critical" && a.str_due ? `<br><span class="muted">STR filing due ${dday(a.str_due)}:</span> ${a.decision?.decision === "filed" ? "filed" : dueChip(a.str_days_left)}` : ""}</p></div>`;
}

function reasonList(a) {
  if (!a.reasons.length) return `<p class="muted">Nothing unusual. Every transfer scored low and no behaviour check fired.</p>`;
  const kinds = { model: "model", behaviour: "behaviour check", network: "network" };
  return a.reasons.map((r) => `
    <div class="reason">
      <div class="reason-head"><h3>${esc(r.title)}<span class="kind">${kinds[r.kind]}</span></h3><span class="points ${a.status}">+${r.points} point${r.points === 1 ? "" : "s"}</span></div>
      <p class="why">${esc(r.explain)}</p>
      <p class="evidence">${linkKeys(esc(r.evidence))}</p>
    </div>`).join("");
}
const linkKeys = (html) => html.replace(/\b(ACC-\d+|\d{2,7}_[0-9A-F]{6,12})\b/g, (k) => `<a href="#/account/${encodeURIComponent(k)}">${k}</a>`);

async function pageAccount(arg, params, token) {
  const idx = await accountsIndex();
  if (!idx.length) { view.innerHTML = emptyState(); return; }
  const key = arg || idx[0].key;
  const compareKey = params.get("compare");
  const [a, b] = await Promise.all([
    api(`/api/batches/${state.batchId}/accounts/${encodeURIComponent(key)}`),
    compareKey ? api(`/api/batches/${state.batchId}/accounts/${encodeURIComponent(compareKey)}`) : null,
  ]);
  if (token !== routeToken) return;
  const options = (sel) => idx.map((r) => `<option value="${esc(r.key)}" ${r.key === sel ? "selected" : ""}>${r.score}  ${esc(r.key)}${r.name ? "  " + esc(r.name) : ""}</option>`).join("");

  view.innerHTML = `
    <div class="page-head">
      <h1>Account</h1>
      <p class="lede">Why an account got its score, the evidence behind it, and every transfer it made in the batch.</p>
    </div>
    <div class="row" style="align-items:flex-end;margin-bottom:26px">
      <label class="field" style="flex:1;min-width:280px;max-width:460px"><span>Account</span><select id="acc-sel">${options(key)}</select></label>
      ${b ? `<label class="field" style="flex:1;min-width:280px;max-width:460px"><span>Compare with</span><select id="cmp-sel">${options(compareKey)}</select></label>` : ""}
      <label class="switch" style="margin-bottom:8px"><input type="checkbox" id="cmp-toggle" ${b ? "checked" : ""}> Compare with another account</label>
    </div>
    ${b ? `
      <div class="grid-2">
        <div>${hero(a, true)}<div class="section"><h2>Why it scored ${a.score}</h2>${reasonList(a)}</div></div>
        <div>${hero(b, true)}<div class="section"><h2>Why it scored ${b.score}</h2>${reasonList(b)}</div></div>
      </div>` : `
      <div class="grid-main">
        <div>${hero(a)}
          <div class="row" style="margin-top:16px">
            ${a.status !== "clear" ? `<button class="btn primary" id="str-btn">Draft STR</button>` : ""}
            <a class="btn" href="#/flow/${encodeURIComponent(a.key)}">Trace where the money went</a>
          </div>
          <div id="decision" style="margin-top:16px"></div>
        </div>
        <div class="panel">
          <h2>Customer profile</h2>
          <dl class="profile">
            ${Object.entries(a.profile).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}
            <dt>Money in</dt><dd class="num">${money(a.in_sgd)}</dd>
            <dt>Money out</dt><dd class="num">${money(a.out_sgd)}</dd>
            <dt>Transfers</dt><dd class="num">${a.txns}</dd>
            <dt>Highest model score</dt><dd class="num">${a.max_model_score.toFixed(3)}</dd>
          </dl>
          ${a.labelled_laundering != null ? `<p class="faint" style="margin:12px 0 0;font-size:12.5px">${a.labelled_laundering ? "Labelled as laundering in this batch's data." : "Not labelled as laundering in this batch's data."}</p>` : ""}
        </div>
      </div>
      <div class="grid-main section">
        <div><h2>Why it scored ${a.score}</h2>${reasonList(a)}</div>
        <div><h2>Transfers over time</h2>
          <div class="legend"><span><i style="background:var(--critical)"></i>Model score 0.75 or more</span><span><i style="background:var(--faint)"></i>Other transfers</span><span>Up: money in. Down: money out.</span></div>
          <div id="scatter"></div></div>
      </div>
      <section class="section">
        <div class="row" style="justify-content:space-between"><h2 style="margin:0">Transfers</h2>
          <label class="switch"><input type="checkbox" id="only-flagged" ${a.transfers.some((t) => t.flagged) ? "checked" : ""}> Only transfers the model scored 0.75 or more</label></div>
        <div class="table-wrap" style="margin-top:12px;max-height:60vh"><table>
          <thead><tr><th>Time</th><th>Direction</th><th class="r">Amount</th><th>Counterparty</th><th>Method</th><th>Currency</th><th class="r">Model score</th><th>What raised the score</th><th>Transfer</th></tr></thead>
          <tbody id="tx-rows"></tbody></table></div>
      </section>`}`;

  $("#acc-sel").addEventListener("change", (e) => go(`#/account/${encodeURIComponent(e.target.value)}${b ? `?compare=${encodeURIComponent(compareKey)}` : ""}`));
  $("#cmp-toggle").addEventListener("change", (e) => {
    if (e.target.checked) {
      const other = idx.find((r) => r.key !== key && r.status !== a.status) || idx.find((r) => r.key !== key);
      go(`#/account/${encodeURIComponent(key)}?compare=${encodeURIComponent(other.key)}`);
    } else go(`#/account/${encodeURIComponent(key)}`);
  });
  if (b) { $("#cmp-sel").addEventListener("change", (e) => go(`#/account/${encodeURIComponent(key)}?compare=${encodeURIComponent(e.target.value)}`)); return; }

  scatterTransfers($("#scatter"), a.transfers);
  const rows = () => {
    const only = $("#only-flagged").checked;
    const list = a.transfers.filter((t) => !only || t.flagged);
    $("#tx-rows").innerHTML = list.length ? list.map((t) => `
      <tr>
        <td class="num" style="white-space:nowrap">${dt(t.time)}</td>
        <td>${t.direction}</td>
        <td class="r num">${money(t.amount, 2)}</td>
        <td>${t.counterparty_status === "external" ? `<span class="muted">${esc(t.counterparty)}</span>` : `<a href="#/account/${encodeURIComponent(t.counterparty)}">${esc(t.counterparty)}</a>`}
          <span class="faint">${esc(t.counterparty_name)}</span></td>
        <td>${esc(t.method)}</td><td>${esc(t.currency)}</td>
        <td class="r num ${t.alert ? "critical" : t.flagged ? "watch" : ""}"><b>${t.score.toFixed(3)}</b></td>
        <td style="font-size:12.5px">${t.drivers.slice(0, 2).map((d) => `${esc(d.label)}: ${esc(d.value)}`).join("<br>") || '<span class="faint">—</span>'}</td>
        <td class="num faint">${esc(t.txn_id)}</td>
      </tr>`).join("") : `<tr><td colspan="9" class="empty">No transfers scored 0.75 or more. Untick the filter to see all ${a.transfers.length}.</td></tr>`;
  };
  $("#only-flagged").addEventListener("change", rows);
  rows();
  if ($("#str-btn")) $("#str-btn").addEventListener("click", () => openStr(a));
  renderDecision(a);
}

function renderDecision(a) {
  const el = $("#decision");
  if (!el) return;
  const d = a.decision;
  if (d) {
    el.innerHTML = `<div class="panel" style="padding:12px 16px"><div class="row" style="justify-content:space-between">
      <div>${decisionTag(d.decision)} <span class="muted">${esc(d.at)}</span>${d.note ? `<div style="margin-top:4px">${esc(d.note)}</div>` : ""}</div>
      <button class="btn small" id="undo">Undo</button></div></div>`;
    $("#undo").addEventListener("click", () => decide(a, null, ""));
    return;
  }
  el.innerHTML = `<details class="panel" style="padding:12px 16px"><summary style="cursor:pointer"><b>Record a decision</b> <span class="muted">for the audit trail</span></summary>
    <div class="stack" style="margin-top:12px">
      <textarea id="d-note" rows="2" placeholder="Note, e.g. customer could not explain the source of funds"></textarea>
      <div class="row">
        <button class="btn" data-d="escalated">Escalate</button>
        <button class="btn" data-d="filed">Mark STR filed</button>
        <button class="btn" data-d="cleared">Clear</button>
      </div></div></details>`;
  el.querySelectorAll("[data-d]").forEach((btn) => btn.addEventListener("click", () => decide(a, btn.dataset.d, $("#d-note").value)));
}

async function decide(a, decision, note) {
  a.decision = await api(`/api/batches/${state.batchId}/accounts/${encodeURIComponent(a.key)}/decision`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision, note }),
  });
  state.accountsIndex = null;
  toast(decision ? `${DECISION[decision]}: ${a.key}` : "Decision removed");
  renderDecision(a);
}

// ---------------------------------------------------------------------------
// STR drawer
// ---------------------------------------------------------------------------
async function openStr(a) {
  const drawer = $("#drawer");
  drawer.innerHTML = `<header><h2 style="margin:0">Draft STR, ${esc(a.key)}</h2><button class="icon-btn" id="d-close">Close</button></header>
    <div class="body"><p class="muted"><span class="spinner"></span> Drafting from the reasons and transfers…</p></div>`;
  drawer.classList.add("open"); $("#scrim").classList.add("open"); drawer.setAttribute("aria-hidden", "false");
  $("#d-close").addEventListener("click", closeDrawer);
  const { text } = await api(`/api/batches/${state.batchId}/accounts/${encodeURIComponent(a.key)}/str`);
  drawer.querySelector(".body").innerHTML = `<p class="muted" style="margin-top:0">Laid out in the same order as the STRO STR form on SONAR, filled in from this account's data. Fields marked [To complete] need the analyst. Check every line before filing. This is a sample layout and may differ from your firm's practice.</p><pre id="str-text" contenteditable="true" spellcheck="false">${esc(text)}</pre>`;
  drawer.insertAdjacentHTML("beforeend", `<footer>
    <button class="btn" id="str-copy">Copy text</button>
    <button class="btn" id="str-dl">Download .txt</button>
    <a class="btn" href="/api/batches/${state.batchId}/accounts/${encodeURIComponent(a.key)}/transactions.csv" download>Transactions .csv</a>
    <button class="btn primary" id="str-filed">Mark STR filed</button></footer>`);
  $("#str-copy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("#str-text").innerText); toast("Copied"); } catch (e) { toast("Copy blocked by the browser; select the text instead"); }
  });
  $("#str-dl").addEventListener("click", () => {
    const blob = new Blob([$("#str-text").innerText], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob); link.download = `STR-draft-${a.key}.txt`; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  });
  $("#str-filed").addEventListener("click", async () => { await decide(a, "filed", "STR filed from RemitSentry draft"); closeDrawer(); });
}
function closeDrawer() {
  $("#drawer").classList.remove("open"); $("#scrim").classList.remove("open"); $("#drawer").setAttribute("aria-hidden", "true");
}
$("#scrim").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

// ---------------------------------------------------------------------------
// Money flow
// ---------------------------------------------------------------------------
const AMOUNT_STEPS = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000];

async function pageFlow(arg, params, token) {
  const idx = await accountsIndex();
  if (!idx.length) { view.innerHTML = emptyState(); return; }
  const all = params.get("view") === "all";
  const center = arg || idx[0].key;
  const hops = Number(params.get("hops") || 2);
  const minIdx = Number(params.get("min") ?? 3);
  view.innerHTML = `
    <div class="page-head">
      <h1>Money flow</h1>
      <p class="lede">${all ? "Every high-risk account and the money moving between them, so suspected rings stand out at once. Click a customer to trace its money; double-click to open it." : "Who sent money to an account, where it went next, and which of those counterparties are high risk themselves. Click a customer to re-centre; double-click to open it."}</p>
    </div>
    <div class="row" style="align-items:flex-end;margin-bottom:12px;gap:20px">
      <div class="field"><span>View</span><div class="seg">
        <button type="button" data-v="one" aria-pressed="${!all}">One account</button>
        <button type="button" data-v="all" aria-pressed="${all}">All high-risk accounts</button></div></div>
      ${all ? "" : `<label class="field" style="flex:1;min-width:280px;max-width:440px"><span>Centre on</span>
        <select id="c-sel">${idx.map((r) => `<option value="${esc(r.key)}" ${r.key === center ? "selected" : ""}>${r.score}  ${esc(r.key)}${r.name ? "  " + esc(r.name) : ""}</option>`).join("")}</select></label>
      <div class="field"><span>Reach</span><div class="seg">
        <button type="button" data-h="1" aria-pressed="${hops === 1}">Direct only</button>
        <button type="button" data-h="2" aria-pressed="${hops === 2}">Two steps</button></div></div>`}
      <label class="field" style="width:240px"><span>Hide transfers below <b id="min-lab">${money(AMOUNT_STEPS[minIdx])}</b> in total</span>
        <input type="range" id="min-r" min="0" max="${AMOUNT_STEPS.length - 1}" value="${minIdx}"></label>
      ${all ? "" : `<a class="btn" href="#/account/${encodeURIComponent(center)}">Open account</a>`}
    </div>
    <div class="legend">
      <span><i style="background:var(--critical)"></i>Critical customer</span><span><i style="background:var(--watch)"></i>Watch</span>
      <span><i style="background:var(--clear)"></i>Clear</span><span><i style="background:var(--external);border-radius:2px"></i>Outside our book (other bank or overseas)</span>
      <span><i class="line" style="background:var(--critical)"></i>Includes a transfer scored 0.75+</span><span>Thicker line, more money. Arrows show direction.</span>
    </div>
    <div class="flow-box" id="flow" style="${all ? "height:700px" : ""}"><div class="tip" id="tip"></div></div>
    <section class="section"><h2>Flows in this view</h2>
      <div class="table-wrap" style="max-height:50vh"><table>
        <thead><tr><th>From</th><th class="r">Score</th><th>To</th><th class="r">Score</th><th class="r">Total</th><th class="r">Transfers</th><th class="r">Scored 0.75+</th><th>First</th><th>Last</th></tr></thead>
        <tbody id="f-rows"></tbody></table></div></section>`;

  const setParams = (h, mi, c = center) => go(`#/flow/${encodeURIComponent(c)}?hops=${h}&min=${mi}`);
  view.querySelectorAll("[data-v]").forEach((b) => b.addEventListener("click", () =>
    b.dataset.v === "all" ? go(`#/flow?view=all&min=${minIdx}`) : setParams(hops, minIdx)));
  if (!all) {
    $("#c-sel").addEventListener("change", (e) => setParams(hops, minIdx, e.target.value));
    view.querySelectorAll("[data-h]").forEach((b) => b.addEventListener("click", () => setParams(Number(b.dataset.h), minIdx)));
  }
  $("#min-r").addEventListener("input", (e) => { $("#min-lab").textContent = money(AMOUNT_STEPS[e.target.value]); });
  $("#min-r").addEventListener("change", (e) => all ? go(`#/flow?view=all&min=${e.target.value}`) : setParams(hops, Number(e.target.value)));

  const g = await api(all
    ? `/api/batches/${state.batchId}/network?min_amount=${AMOUNT_STEPS[minIdx]}`
    : `/api/batches/${state.batchId}/graph?center=${encodeURIComponent(center)}&hops=${hops}&min_amount=${AMOUNT_STEPS[minIdx]}`);
  if (token !== routeToken) return;
  const byId = Object.fromEntries(g.nodes.map((n) => [n.id, n]));
  flowGraph($("#flow"), g, {
    onRecentre: (id) => setParams(hops, minIdx, id),
    onOpen: (id) => go(`#/account/${encodeURIComponent(id)}`),
    money,
    dense: all,
  });
  const sc = (id) => byId[id]?.score == null ? '<span class="faint">outside</span>' : `<b class="${byId[id].kind}">${byId[id].score}</b>`;
  const name = (id) => byId[id]?.kind === "external" ? `<span class="muted">${esc(id)}</span>` : `<a href="#/account/${encodeURIComponent(id)}">${esc(id)}</a>${byId[id]?.label && byId[id].label !== id ? ` <span class="faint">${esc(byId[id].label)}</span>` : ""}`;
  const links = [...g.links].sort((x, y) => y.flagged - x.flagged || y.total - x.total);
  $("#f-rows").innerHTML = links.length ? links.map((l) => `<tr>
      <td>${name(l.source)}</td><td class="r">${sc(l.source)}</td><td>${name(l.target)}</td><td class="r">${sc(l.target)}</td>
      <td class="r num">${money(l.total, 2)}</td><td class="r num">${l.n}</td><td class="r num ${l.flagged ? "critical" : "faint"}">${l.flagged}</td>
      <td class="num">${esc(l.first)}</td><td class="num">${esc(l.last)}</td></tr>`).join("")
    : `<tr><td colspan="9" class="empty">No transfers above ${money(AMOUNT_STEPS[minIdx])}. Lower the amount filter.</td></tr>`;
}

// ---------------------------------------------------------------------------
(async function boot() {
  try {
    await refreshBatches();
  } catch (e) {
    view.innerHTML = `<div class="notice error"><b>The app could not reach its server.</b> Start it with <code>python -m app.server</code> from the repo folder, then reload this page.</div>`;
    return;
  }
  route();
})();
