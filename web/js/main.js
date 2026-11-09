import { initTheme } from "./theme.js";
import { runIntro } from "./intro.js";
import { createTracker } from "./tracker.js";

const $ = (id) => document.getElementById(id);
const media = $("media");
const MIN_MARGIN = 0.1;                    // the best guess must beat the runner-up by this much
const IGNORED = {
  still: "Your hands were mostly still, so nothing was recognized.",
  hands: "Your hands were not in view enough to recognize a sign.",
  "no sign": "That did not look like a sign.",
};
let ws = null, open = false, words = [], recorded = [], restLine = null;
let cal = null, fpsEma = 0, lastT = 0, toastTimer = 0;

initTheme($("themeBtn"));
runIntro();

function toast(msg, ms = 3500) {
  const t = $("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), ms);
}
const hint = (m) => { $("hint").textContent = m; };

/* ---- server ---- */
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { open = true; setConn(true); if (restLine) sendJson({ cmd: "config", rest_line: restLine }); };
  ws.onclose = () => { open = false; setConn(false); media.classList.remove("signing"); setTimeout(connect, 1000); };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
}
const sendJson = (o) => open && ws.send(JSON.stringify(o));
function setConn(ok) {
  $("conn").classList.toggle("ok", ok);
  $("connText").textContent = ok ? "Connected" : "Server not reachable, retrying";
}
function handle(m) {
  if (m.type === "state") media.classList.toggle("signing", m.signing);
  else if (m.type === "prediction") showPrediction(m);
  else if (m.type === "ignored") {
    $("motion").textContent = m.motion.toFixed(2);
    $("bars").replaceChildren(Object.assign(document.createElement("p"), { className: "note", textContent: IGNORED[m.reason] || IGNORED.still }));
    $("word").textContent = "Waiting for a sign"; $("word").classList.add("dim");
    $("near").textContent = "";
  } else if (m.type === "error") hint("Server rejected a frame: " + m.detail);
}

/* ---- results ---- */
function showPrediction(m) {
  const top = m.top || [], thr = Number($("thr").value) / 100;
  $("srv").textContent = m.latency_ms.toFixed(1);
  $("nframes").textContent = m.n_frames;
  if (m.motion != null) $("motion").textContent = m.motion.toFixed(2);
  const bars = $("bars"); bars.replaceChildren();
  top.slice(0, 3).forEach((t, i) => {
    const row = document.createElement("div"); row.className = "bar" + (i ? "" : " top");
    const name = Object.assign(document.createElement("span"), { className: "name", textContent: t.label, title: t.label });
    const track = document.createElement("span"); track.className = "track";
    const fill = document.createElement("span"); fill.className = "fill"; fill.style.width = (t.p * 100).toFixed(0) + "%";
    track.append(fill);
    const pct = Object.assign(document.createElement("span"), { className: "pct", textContent: (t.p * 100).toFixed(0) + "%" });
    row.append(name, track, pct); bars.append(row);
  });
  const best = top[0], word = $("word");
  const margin = top.length > 1 ? best.p - top[1].p : best.p;
  if (best && best.p >= thr && margin >= MIN_MARGIN) {
    word.textContent = best.label; word.classList.remove("dim"); $("near").textContent = "";
    words.push(best.label); renderTranscript();
  } else if (best) {
    word.textContent = "Not sure"; word.classList.add("dim");
    $("near").textContent = `Closest guess: ${best.label} (${(best.p * 100).toFixed(0)}%)`;
  }
}
function renderTranscript() {
  const el = $("sentence");
  el.textContent = words.length ? words.join(" ") : "Recognized words will collect here.";
  el.classList.toggle("empty", !words.length);
}

/* ---- rest-position calibration ---- */
// Finds how high your resting wrists sit (in shoulder widths below the shoulders) and puts the
// "hands are up" line a margin above that, so resting hands never count as signing.
function restHeight(a) {
  const sw = Math.hypot(a[33] - a[36], a[34] - a[37]);
  if (sw < 1e-6) return null;
  const mid = (a[34] + a[37]) / 2;
  return Math.min((a[46] - mid) / sw, (a[49] - mid) / sw);
}
function finishCalibration() {
  const v = cal.filter((x) => x != null).sort((a, b) => a - b);
  cal = null; $("calBtn").textContent = "Calibrate rest position";
  if (v.length < 15) { toast("Could not see your shoulders and wrists. Move back so both are in frame."); return; }
  const p10 = v[Math.floor(v.length * 0.1)];
  restLine = Math.round(Math.min(1.8, Math.max(0.6, p10 - 0.3)) * 100) / 100;
  sendJson({ cmd: "config", rest_line: restLine });
  $("rest").textContent = restLine.toFixed(2) + " (calibrated)";
  toast(p10 < 0.9 ? `Rest line set to ${restLine}. Your hands rest quite high, so lowering them will work better.`
                  : `Rest line set to ${restLine}.`);
}

/* ---- steady 25 fps stream ---- */
// The model was trained on 25 fps video, but the browser tracks at about 12 to 17 fps. Missing frames are
// filled in by blending the two nearest tracked frames, so the server always sees smooth 25 fps motion.
const DT = 40;
let prev = null, prevT = 0, nextOut = 0;
function blend(a, b, t) {
  const o = new Float32Array(225);
  for (let i = 0; i < 75; i++) {
    const k = i * 3;
    const both = (a[k] || a[k + 1] || a[k + 2]) && (b[k] || b[k + 1] || b[k + 2]);
    const s = t < 0.5 ? a : b;
    for (let j = 0; j < 3; j++) o[k + j] = both ? a[k + j] + (b[k + j] - a[k + j]) * t : s[k + j];
  }
  return o;
}
function emit(a) {
  if (recorded.length < 4000) recorded.push(a);
  if (open && ws.bufferedAmount < 200000) ws.send(a.buffer);
  if (cal) { cal.push(restHeight(a)); if (cal.length >= 40) finishCalibration(); }
}
function feed(a, now) {
  if (tracker.mode === "file" || !prev || now - prevT > 500) { prev = a; prevT = now; nextOut = now + DT; emit(a); return; }
  while (nextOut <= now) { emit(blend(prev, a, (nextOut - prevT) / (now - prevT))); nextOut += DT; }
  prev = a; prevT = now;
}

/* ---- tracker ---- */
const tracker = createTracker({
  video: $("video"), canvas: $("overlay"),
  onFrame(a) {
    const now = performance.now();
    if (lastT) { fpsEma = fpsEma ? fpsEma * 0.9 + (1000 / (now - lastT)) * 0.1 : 1000 / (now - lastT); $("fps").textContent = fpsEma.toFixed(0); }
    lastT = now;
    const l = a.slice(99, 162).some((x) => x), r = a.slice(162, 225).some((x) => x);
    $("hands").textContent = l && r ? "both" : l ? "left only" : r ? "right only" : "none";
    feed(a, now);
  },
  onEvent(type, detail) {
    if (type === "started") {
      document.body.classList.add("running"); recorded = []; prev = null; fpsEma = 0; lastT = 0;
      $("camBtn").textContent = detail === "cam" ? "Stop camera" : "Stop video";
      $("camBtn").classList.add("stop"); $("camBtn").classList.remove("primary");
      media.classList.toggle("mirror", $("mirror").checked && detail === "cam");
      hint("Loading the tracking model. The first frames can take a few seconds.");
    } else if (type === "ready") { hint("Tracking."); if (!restLine && tracker.mode === "cam") startCalibration(); }
    else if (type === "stopped") {
      document.body.classList.remove("running"); media.classList.remove("signing");
      $("camBtn").textContent = "Start camera"; $("camBtn").classList.remove("stop"); $("camBtn").classList.add("primary");
      cal = null; sendJson({ cmd: "reset" });
    } else if (type === "file-end") {
      if (open) { const z = new Float32Array(225); for (let i = 0; i < 12; i++) ws.send(z.buffer); }  // close the last sign
      hint(`Finished the file: ${recorded.length} frames tracked.`);
    } else if (type === "error") hint(detail);
  },
});

// the box follows the real video shape, so there are never black bars and the image is never stretched
$("video").addEventListener("loadedmetadata", () => {
  const v = $("video");
  if (v.videoWidth) media.style.aspectRatio = `${v.videoWidth} / ${v.videoHeight}`;
});

/* ---- buttons ---- */
$("camBtn").addEventListener("click", () => (tracker.running ? tracker.stop() : tracker.startCam()));
$("file").addEventListener("change", (e) => { if (e.target.files[0]) tracker.startFile(e.target.files[0]); e.target.value = ""; });
function startCalibration() {
  cal = []; $("calBtn").textContent = "Hold still...";
  toast("Let your hands rest as they normally do for two seconds.", 2500);
}
$("calBtn").addEventListener("click", () => {
  if (!tracker.running || tracker.mode !== "cam") { toast("Start the camera first, then let your hands rest where they normally do."); return; }
  startCalibration();
});
$("resetBtn").addEventListener("click", () => { sendJson({ cmd: "reset" }); media.classList.remove("signing"); });
$("clearBtn").addEventListener("click", () => {
  words = []; renderTranscript();
  $("word").textContent = "Waiting for a sign"; $("word").classList.add("dim"); $("near").textContent = "";
});
$("copyBtn").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(words.join(" ")); toast("Copied the transcript."); }
  catch { toast("Copy was blocked by the browser."); }
});
$("dlBtn").addEventListener("click", () => {
  if (!recorded.length) { toast("No landmarks recorded yet."); return; }
  const all = new Float32Array(recorded.length * 225);
  recorded.forEach((a, i) => all.set(a, i * 225));
  const link = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([all.buffer])), download: `landmarks_${tracker.mode}_${recorded.length}frames.f32` });
  link.click(); URL.revokeObjectURL(link.href);
});
$("thr").addEventListener("input", () => { $("thrv").textContent = $("thr").value + "%"; });
$("mirror").addEventListener("change", () => media.classList.toggle("mirror", $("mirror").checked && tracker.mode === "cam"));
$("showLm").addEventListener("change", () => tracker.setLandmarks($("showLm").checked));

connect();