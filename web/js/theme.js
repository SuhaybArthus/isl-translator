const KEY = "isl-theme";
const root = document.documentElement;

export function initTheme(btn) {
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch {}
  apply(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  btn.addEventListener("click", () => apply(root.dataset.theme === "dark" ? "light" : "dark", true));

  function apply(t, save) {
    root.dataset.theme = t;
    btn.textContent = t === "dark" ? "Light mode" : "Dark mode";
    if (save) { try { localStorage.setItem(KEY, t); } catch {} }
  }
}
