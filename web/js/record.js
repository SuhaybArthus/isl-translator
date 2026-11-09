import { initTheme } from "./theme.js";
import { createTracker } from "./tracker.js";
import { createSteady } from "./steady.js";

const $ = (id) => document.getElementById(id);
const num = (id) => Number($(id).value);
const media = $("media"), FPS = 25, PRE = 12, POST = 8;       // same padding the live segmenter uses
let state = "idle";                  // idle, countdown, recording, pause (between automatic takes) or review
let take = [], rec = [], trim = [0, 0], tick = 0, pauseT = 0, total = 1, done = 0, saved = 0, skipped = 0;
let counts = {}, known = [], toastTimer = 0, actx = null;
const NOSIGN = "No sign";             // idle behaviour: kept whole, and hands do not need to be tracked
const validLabel = (s) => /^[A-Za-z0-9][A-Za-z0-9 ()\-]{0,39}$/.test(s);
const hint = (m) => { $("hint").textContent = m; };
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

initTheme($("themeBtn"));

function toast(msg, ms = 3200) {
  const t = $("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), ms);
}
function beep(freq = 700, dur = 0.12) {
  try {
    actx = actx || new AudioContext();
    const o = actx.createOscillator(), g = actx.createGain();
    o.frequency.value = freq; g.gain.value = 0.08;
    o.connect(g); g.connect(actx.destination); o.start(); o.stop(actx.currentTime + dur);
  } catch { /* sound is optional */ }
}

/* ---- words and server ---- */
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
  return r.json();
}
const allWords = () => [...new Set([...known, NOSIGN, ...Object.keys(counts)])].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
function currentWord() {
  const sel = $("wordSel").value;
  if (sel !== "__new__") return sel;
  const typed = $("label").value.trim();
  return allWords().find((w) => w.toLowerCase() === typed.toLowerCase()) || typed;   // reuse an existing spelling
}
function fillSelect() {
  const sel = $("wordSel"), cur = sel.value;
  const opt = (v, t, extra = {}) => Object.assign(document.createElement("option"), { value: v, textContent: t, ...extra });
  sel.replaceChildren(opt("", "Choose a word", { disabled: true }), opt("__new__", "Add a new word"),
    ...allWords().map((w) => opt(w, counts[w] ? `${w} (${counts[w]} saved)` : w)));
  sel.value = [...sel.options].some((o) => o.value === cur) ? cur : "";
}
async function loadLists() {
  try {
    const { labels } = await api("/api/labels");
    known = labels;
    counts = await api("/api/recordings");
    $("conn").classList.add("ok"); $("connText").textContent = "Connected";
  } catch (e) {
    $("conn").classList.remove("ok");
    $("connText").textContent = e.message === "Failed to fetch" ? "Server not reachable" : "Server problem: " + e.message;
  }
  fillSelect(); renderCounts();
}
function renderCounts() {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const sum = entries.reduce((s, [, n]) => s + n, 0);
  $("total").textContent = sum ? `${sum} takes across ${entries.length} words.` : "Nothing saved yet.";
  const max = Math.max(1, ...entries.map((e) => e[1]));
  $("saved").replaceChildren(...entries.slice(0, 12).map(([w, n]) => {
    const row = document.createElement("div"); row.className = "bar";
    const name = Object.assign(document.createElement("span"), { className: "name", textContent: w, title: w });
    const track = document.createElement("span"); track.className = "track";
    const fill = document.createElement("span"); fill.className = "fill"; fill.style.width = (n / max * 100) + "%";
    track.append(fill);
    row.append(name, track, Object.assign(document.createElement("span"), { className: "pct", textContent: n }));
    return row;
  }));
  const w = currentWord(), n = counts[w] || 0;
  $("count").textContent = w === NOSIGN
    ? "Record what you do when you are NOT signing: resting, reaching for the mouse, adjusting your hair, talking. Aim for 100 or more varied takes."
    : $("wordSel").value === "__new__" && w && !known.includes(w) && !n
    ? "This is a new word the model does not know. It needs many takes, and training has to include it."
    : n ? `${n} take${n === 1 ? "" : "s"} saved for this word.` : "No takes saved for this word yet.";
}
const pack = (frames) => { const a = new Float32Array(frames.length * 225); frames.forEach((f, i) => a.set(f, i * 225)); return a; };
async function upload(frames) {
  const w = currentWord();
  const r = await api(`/api/recordings?label=${encodeURIComponent(w)}&hand=${$("hand").value}`,
    { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: pack(frames).buffer });
  counts[w] = r.count; fillSelect(); renderCounts();
  return r;
}

/* ---- tracking ---- */
const steady = createSteady((f) => {
  if (state !== "recording") return;
  take.push(f);
  if (take.length >= FPS * num("lenSel")) stopTake();
});
const tracker = createTracker({
  video: $("video"), canvas: $("overlay"),
  onFrame(a) { steady.feed(a, performance.now()); },
  onEvent(type, detail) {
    if (type === "started") {
      document.body.classList.add("running"); steady.reset();
      $("camBtn").textContent = "Stop camera"; $("camBtn").classList.add("stop"); $("camBtn").classList.remove("primary");
      $("recBtn").disabled = false;
      hint("Loading the tracking model. The first frames can take a few seconds.");
    } else if (type === "ready") hint("Tracking. Keep both hands in view so the points show on them.");
    else if (type === "stopped") {
      cancelAll(); document.body.classList.remove("running");
      $("camBtn").textContent = "Start camera"; $("camBtn").classList.remove("stop"); $("camBtn").classList.add("primary");
      $("recBtn").disabled = true;
    } else if (type === "error") hint(detail);
  },
});
$("video").addEventListener("loadedmetadata", () => {
  const v = $("video");
  if (v.videoWidth) media.style.aspectRatio = `${v.videoWidth} / ${v.videoHeight}`;
});

/* ---- trimming ---- */
// Keeps the stretch where a wrist is above the rest line (same rule as the live segmenter), plus padding.
function autoTrim(fr) {
  const up = fr.map((f) => {
    const sw = Math.hypot(f[33] - f[36], f[34] - f[37]);
    if (sw < 1e-6) return false;
    const line = (f[34] + f[37]) / 2 + 1.6 * sw;
    return ((f[45] || f[46]) && f[46] < line) || ((f[48] || f[49]) && f[49] < line);
  });
  const first = up.indexOf(true), last = up.lastIndexOf(true);
  return first < 0 ? [0, fr.length - 1] : [Math.max(0, first - PRE), Math.min(fr.length - 1, last + POST)];
}
const bothHands = () => $("hand").value === "both";
// share of frames where a hand is tracked; for two-handed signs both hands must be tracked
const handsShare = (fr, both = false) => (fr.length ? fr.filter((f) => {
  const l = f.slice(99, 162).some((x) => x), r = f.slice(162, 225).some((x) => x);
  return both ? l && r : l || r;
}).length / fr.length : 0);

function drawFrame(f) {
  const c = $("pv"), x = c.getContext("2d");
  x.clearRect(0, 0, c.width, c.height);
  for (const [s, e, col, r] of [[0, 33, "#E6ECF2", 3], [33, 54, css("--blue"), 4], [54, 75, css("--saffron"), 4]]) {
    x.fillStyle = col;
    for (let i = s; i < e; i++) {
      const px = f[i * 3], py = f[i * 3 + 1];
      if (!px && !py) continue;
      x.beginPath(); x.arc((1 - px) * c.width, py * c.height, r, 0, 6.283); x.fill();   // mirrored like the camera preview
    }
  }
}
function updateTrim(which) {
  let a = num("trimA"), b = num("trimB");
  if (a > b) { if (which === "A") { b = a; $("trimB").value = b; } else { a = b; $("trimA").value = a; } }
  trim = [a, b];
  const kept = rec.slice(a, b + 1);
  $("rFrames").textContent = kept.length; $("rSecs").textContent = (kept.length / FPS).toFixed(1) + " s";
  $("rHands").textContent = Math.round(handsShare(kept, bothHands()) * 100) + "% of frames";
  drawFrame(rec[which === "B" ? b : a]);
}

/* ---- recording flow ---- */
function show(word, sub = "", dim = false) {
  $("word").textContent = word; $("word").classList.toggle("dim", dim); $("sub").textContent = sub;
}
function cancelAll() {
  clearInterval(tick); clearTimeout(pauseT); state = "idle"; take = []; rec = [];
  media.classList.remove("recording"); $("review").hidden = true;
  $("recBtn").textContent = "Record a take"; show("Ready", "", true);
}
function startCountdown() {
  const w = currentWord();
  if (!w) { toast("Choose a word first."); $("wordSel").focus(); state = "idle"; return; }
  if (!allWords().includes(w) && !validLabel(w)) {
    toast("Use letters, numbers, spaces, brackets and hyphens for a new word."); $("label").focus(); state = "idle"; return;
  }
  state = "countdown"; $("recBtn").textContent = "Cancel";
  let n = num("cdSel");
  const msg = total > 1 ? `Take ${done + 1} of ${total}. Get in position.` : "Get in position, then sign once and lower your hands.";
  show(String(n), msg); beep(600);
  tick = setInterval(() => {
    n--;
    if (n > 0) { show(String(n), msg); beep(600); return; }
    clearInterval(tick); take = []; steady.reset(); state = "recording";
    media.classList.add("recording"); $("recBtn").textContent = "Stop";
    show("Recording", "Sign now, then lower your hands. It stops by itself."); beep(900, 0.25);
  }, 1000);
}
async function stopTake() {
  if (state !== "recording") return;
  media.classList.remove("recording"); beep(500, 0.3);
  const frames = take; take = [];
  const [a, b] = currentWord() === NOSIGN ? [0, frames.length - 1] : autoTrim(frames);
  if (total > 1) { await autoSave(frames, a, b); return; }
  state = "review"; rec = frames; $("recBtn").textContent = "Record a take"; $("recBtn").disabled = true;
  $("trimA").max = $("trimB").max = String(frames.length - 1);
  $("trimA").value = a; $("trimB").value = b;
  $("review").hidden = false;
  show("Review the take", "The sign is trimmed for you. Move the sliders to adjust it.", true);
  updateTrim("A");
}
async function autoSave(frames, a, b) {
  state = "pause";
  const kept = frames.slice(a, b + 1);
  if (kept.length < 20 || (currentWord() !== NOSIGN && handsShare(kept, bothHands()) < 0.5)) { skipped++; toast(`Skipped take ${done + 1}: ${bothHands() ? "both hands were" : "your hands were"} not tracked well enough.`); }
  else { try { await upload(kept); saved++; } catch (e) { skipped++; toast("Could not save: " + e.message); } }
  done++;
  if (done < total) {
    show("Next take soon", `${saved} saved, ${skipped} skipped.`, true);
    pauseT = setTimeout(startCountdown, 1500);
  } else {
    state = "idle"; $("recBtn").textContent = "Record a take";
    show("Done", `${saved} saved, ${skipped} skipped.`, true); toast(`Finished: ${saved} saved, ${skipped} skipped.`);
  }
}
function closeReview() {
  state = "idle"; rec = []; $("review").hidden = true; $("recBtn").disabled = !tracker.running; show("Ready", "", true);
}

$("recBtn").addEventListener("click", () => {
  if (state === "idle") { total = num("nSel"); done = 0; saved = 0; skipped = 0; startCountdown(); }
  else if (state === "countdown" || state === "pause") cancelAll();
  else if (state === "recording") stopTake();
});
$("camBtn").addEventListener("click", () => (tracker.running ? tracker.stop() : tracker.startCam()));
$("trimA").addEventListener("input", () => updateTrim("A"));
$("trimB").addEventListener("input", () => updateTrim("B"));
$("discardBtn").addEventListener("click", closeReview);
$("saveBtn").addEventListener("click", async () => {
  const kept = rec.slice(trim[0], trim[1] + 1);
  if (kept.length < 20) { toast("That is under a second of signing. Widen the trim or record again."); return; }
  try { const r = await upload(kept); toast(`Saved take ${r.count} for ${currentWord()}.`); closeReview(); }
  catch (e) { toast("Could not save: " + e.message); }
});
$("label").addEventListener("input", renderCounts);
$("wordSel").addEventListener("change", () => {
  const isNew = $("wordSel").value === "__new__";
  $("label").hidden = !isNew;
  if (isNew) $("label").focus();
  renderCounts();
});
addEventListener("keydown", (e) => {
  if (e.code !== "Space" || ["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) return;
  e.preventDefault(); if (!$("recBtn").disabled) $("recBtn").click();
});

loadLists();