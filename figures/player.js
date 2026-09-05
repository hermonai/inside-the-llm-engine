/* Discrete pedagogical states: no automatic motion, no interpolated fake data. */
'use strict';
const frames = [...document.querySelectorAll('.frame')];
const slider = document.getElementById('step');
const play = document.getElementById('play');
let current = 0;
let timer = null;
function stop() {
  clearInterval(timer);
  timer = null;
  play.textContent = 'Play';
}
function show(value) {
  current = Math.max(0, Math.min(frames.length - 1, value));
  frames.forEach((frame, index) => { frame.hidden = index !== current; });
  slider.value = current;
  document.getElementById('state').textContent = `Frame ${current + 1} of ${frames.length}`;
}
slider.addEventListener('input', () => { stop(); show(Number(slider.value)); });
document.getElementById('prev').addEventListener('click', () => { stop(); show(current - 1); });
document.getElementById('next').addEventListener('click', () => { stop(); show(current + 1); });
play.addEventListener('click', () => {
  if (timer) { stop(); return; }
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { show((current + 1) % frames.length); return; }
  play.textContent = 'Pause';
  if (current === frames.length - 1) show(0);
  timer = setInterval(() => {
    show(current + 1);
    if (current === frames.length - 1) stop();
  }, 1300);
});
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
