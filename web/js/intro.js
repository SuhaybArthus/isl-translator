// Hand skeleton in MediaPipe's 21-point layout, drawn in a 100 x 120 box.
const PTS = [[50,110],[34,98],[22,84],[14,70],[8,58],[36,64],[34,44],[33,30],[32,18],[48,60],[48,38],
  [48,22],[48,8],[60,62],[62,42],[63,28],[64,16],[72,68],[76,52],[79,40],[81,30]];
const EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],
  [13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const TIPS = [4, 8, 12, 16, 20];
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, delay) => {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (delay != null) n.style.setProperty("--d", `${delay}ms`);
  return n;
};

// Two hands start pressed together, open into palms, light up, then a web of lines links them.
function hand(side, outer) {
  const g = el("g", { transform: outer });
  const h = el("g", { class: "h " + side });
  EDGES.forEach(([a, b], i) => h.append(el("line", { x1: PTS[a][0], y1: PTS[a][1], x2: PTS[b][0], y2: PTS[b][1], pathLength: 1 }, i * 40)));
  PTS.forEach(([x, y], i) => h.append(el("circle", { cx: x, cy: y, r: 2.6 }, 500 + i * 30)));
  g.append(h);
  return g;
}

export function runIntro() {
  const root = document.getElementById("intro");
  if (sessionStorage.getItem("isl-intro") || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    root.classList.add("gone");
    return;
  }
  sessionStorage.setItem("isl-intro", "1");
  const art = document.getElementById("introArt");
  const web = el("g", { class: "web" });
  TIPS.forEach((t, i) => [i, (i + 1) % 5].forEach((j, k) => web.append(el("line", {
    x1: PTS[t][0], y1: PTS[t][1], x2: 210 - PTS[TIPS[j]][0], y2: PTS[TIPS[j]][1], pathLength: 1 }, 1700 + (i * 2 + k) * 60))));
  art.append(hand("l", "translate(0 0)"), hand("r", "translate(210 0) scale(-1 1)"), web,
    el("path", { d: "M -20 112 C 40 -34, 170 -34, 230 104", pathLength: 1 }, 2000));

  let done = false;
  const leave = () => {
    if (done) return;
    done = true;
    root.classList.add("leave");
    setTimeout(() => root.classList.add("gone"), 750);
    removeEventListener("keydown", leave);
  };
  setTimeout(leave, 3600);
  root.addEventListener("click", leave);
  addEventListener("keydown", leave);
}