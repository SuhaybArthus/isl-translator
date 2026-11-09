const VER = "0.5.1675471629";
const FPS = 25;                                   // INCLUDE clips are about 25 fps

// Same layout the model was trained on: pose(33) + left hand(21) + right hand(21), x/y/z each.
// Undetected landmarks stay 0. Frames are sent unmirrored; only the preview is flipped.
function pack(r) {
  const a = new Float32Array(225);
  const put = (lms, off, n) => {
    if (!lms) return;
    for (let i = 0; i < n && i < lms.length; i++) {
      a[(off + i) * 3] = lms[i].x; a[(off + i) * 3 + 1] = lms[i].y; a[(off + i) * 3 + 2] = lms[i].z;
    }
  };
  put(r.poseLandmarks, 0, 33); put(r.leftHandLandmarks, 33, 21); put(r.rightHandLandmarks, 54, 21);
  return a;
}

export function createTracker({ video, canvas, onFrame, onEvent }) {
  const ctx = canvas.getContext("2d");
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  let holistic = null, running = false, mode = "cam", busy = false, lastSend = 0, token = 0, ready = false;
  let showLm = true;

  function init() {
    if (holistic) return;
    holistic = new Holistic({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@${VER}/${f}` });
    holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true, enableSegmentation: false,
      refineFaceLandmarks: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    holistic.onResults((r) => {
      if (!ready) { ready = true; onEvent("ready"); }
      onFrame(pack(r), r);
      draw(r);
    });
  }

  function dots(lms, rad, color, w, h) {
    if (!lms) return;
    ctx.fillStyle = color;
    for (const p of lms) { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, rad, 0, 6.283); ctx.fill(); }
  }
  function draw(r) {
    const w = video.videoWidth, h = video.videoHeight;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    ctx.clearRect(0, 0, w, h);
    if (!showLm) return;
    dots(r.poseLandmarks, 3, "rgba(230,236,242,.85)", w, h);
    dots(r.leftHandLandmarks, 4, css("--blue"), w, h);
    dots(r.rightHandLandmarks, 4, css("--saffron"), w, h);
  }

  async function loop() {
    requestAnimationFrame(loop);
    if (!running || mode !== "cam" || busy || video.readyState < 2 || !video.videoWidth) return;
    const now = performance.now();
    if (now - lastSend < 1000 / FPS) return;
    busy = true; lastSend = now;
    try { await holistic.send({ image: video }); } catch (e) { onEvent("error", e.message); }
    busy = false;
  }

  function stop() {
    running = false; token++;
    const s = video.srcObject;
    if (s) s.getTracks().forEach((t) => t.stop());
    video.srcObject = null; video.removeAttribute("src"); video.load();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    onEvent("stopped");
  }
  function begin(m) { init(); mode = m; ready = false; running = true; onEvent("started", m); }

  async function startCam() {
    stop();
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 540 }, aspectRatio: { ideal: 16 / 9 }, facingMode: "user" }, audio: false });
      await video.play();
    } catch (e) { onEvent("error", "Could not open the camera: " + e.message); return; }
    begin("cam");
  }

  const seekTo = (t) => new Promise((res) => {
    if (Math.abs(video.currentTime - t) < 1e-6) return res();
    const h = () => { video.removeEventListener("seeked", h); res(); };
    video.addEventListener("seeked", h); video.currentTime = t;
  });
  // File mode steps one frame at a time so every frame is tracked, however slow this computer is.
  function startFile(file) {
    stop();
    video.src = URL.createObjectURL(file); video.muted = true;
    begin("file");
    video.addEventListener("error", () => onEvent("error", "This browser cannot play that file. Try an .mp4."), { once: true });
    video.addEventListener("loadeddata", async () => {
      const mine = ++token, dt = 1 / FPS;
      video.pause();
      for (let i = 0; i * dt < video.duration && mine === token; i++) {
        await seekTo((i + 0.5) * dt);
        if (mine !== token) return;
        try { await holistic.send({ image: video }); } catch (e) { onEvent("error", e.message); return; }
      }
      if (mine === token) onEvent("file-end");
    }, { once: true });
  }

  requestAnimationFrame(loop);
  return { startCam, startFile, stop, setLandmarks: (v) => { showLm = v; if (!v) ctx.clearRect(0, 0, canvas.width, canvas.height); },
    get mode() { return mode; }, get running() { return running; } };
}