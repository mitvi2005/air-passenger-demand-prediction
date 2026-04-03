const _tt = document.getElementById('themeToggle');
if (localStorage.getItem('apx-theme') === 'light') document.body.classList.add('light');
_tt && _tt.addEventListener('click', () => {
  document.body.classList.toggle('light');
  localStorage.setItem('apx-theme', document.body.classList.contains('light') ? 'light' : 'dark');
});

// Mobile nav
const _hb = document.getElementById('navHamburger');
const _nl = document.getElementById('navLinks');
_hb && _hb.addEventListener('click', () => _nl && _nl.classList.toggle('open'));

// Animate data-w bars
function animBars() {
  document.querySelectorAll('[data-w]').forEach(el => { el.style.width = el.dataset.w; });
}
window.addEventListener('load', () => setTimeout(animBars, 400));

// Colors helper — maps to #111FA2 / #FFF6F6 system
function C() {
  const dark = !document.body.classList.contains('light');
  return {
    ac:     '#111FA2',
    acl:    '#3d4fd4',
    acll:   '#7b8ae8',
    warm:   '#9e6969',
    ok:     '#4ECBA8',
    err:    '#E05C5C',
    warn:   '#E8B84B',
    purple: '#8B5CF6',
    orange: '#F97316',
    teal:   '#06B6D4',
    txt:  dark ? '#FFF6F6' : '#0a0e2e',
    sub:  dark ? '#8890c4' : '#3a4080',
    mut:  dark ? '#3d4570' : '#8890c4',
    grid: dark ? 'rgba(17,31,162,.15)' : 'rgba(17,31,162,.1)',
    card: dark ? '#111a3d' : '#ffffff',
    surf: dark ? '#0d1230' : '#ffffff',
    dark,
  };
}
function isDark() { return !document.body.classList.contains('light'); }

function setChartDefaults() {
  const c = C();
  if (!window.Chart) return;
  Chart.defaults.color = c.txt;
  Chart.defaults.borderColor = c.grid;
  Chart.defaults.font.family = "'Space Mono',monospace";
  Chart.defaults.font.size = 10;
}

// Modal helpers
function openMo(id)  { const m = document.getElementById(id); m && m.classList.add('open'); document.body.style.overflow = 'hidden'; }
function closeMo(id) { const m = document.getElementById(id); m && m.classList.remove('open'); document.body.style.overflow = ''; }
document.addEventListener('click', e => {
  if (e.target.classList.contains('mo')) { e.target.classList.remove('open'); document.body.style.overflow = ''; }
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { document.querySelectorAll('.mo.open').forEach(m => m.classList.remove('open')); document.body.style.overflow = ''; }
});
