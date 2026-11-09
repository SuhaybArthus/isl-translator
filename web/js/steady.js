// Turns tracked frames (about 12 to 17 per second in the browser) into a steady 25 fps stream,
// the speed the model was trained on, by blending the two nearest tracked frames.
const DT = 40;

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

export function createSteady(emit) {
  let prev = null, prevT = 0, nextOut = 0;
  return {
    reset() { prev = null; },
    feed(a, now) {
      if (!prev || now - prevT > 500) { prev = a; prevT = now; nextOut = now + DT; emit(a); return; }
      while (nextOut <= now) { emit(blend(prev, a, (nextOut - prevT) / (now - prevT))); nextOut += DT; }
      prev = a; prevT = now;
    },
  };
}