'use strict';
const sizeControl = document.querySelector('#model-size');
const fpsButtons = [...document.querySelectorAll('[data-fps]')];
let fps = 0.1;
let data;
const benchmarkNames = { grounding: ['Charades-STA', 'ActivityNet', 'NExT-GQA'], qa: ['Video-MME', 'MVBench', 'LongVideoBench', 'MMVU', 'VideoMMMU', 'MVP'] };
function row(task, model) { return data[task].find(r => r.size === sizeControl.value && r.fps === fps && r.model === model); }
function metric(task, title, subtitle) {
  const base = row(task, 'Qwen3.5');
  const saver = row(task, 'SAVER');
  const baseScore = base.values.at(-1);
  const saverScore = saver.values.at(-1);
  return `<article><div class="metric-top"><div><h3>${title}</h3><p class="metric-subtitle">${subtitle}</p></div><span class="gain">+${(saverScore - baseScore).toFixed(1)} points</span></div>${[base, saver].map(r => `<div class="bar-item"><span class="bar-label">${r.model}</span><div class="bar-track" aria-hidden="true"><div class="bar-fill ${r.model === 'SAVER' ? 'saver' : ''}" style="width:${r.values.at(-1)}%"></div></div><span class="bar-value">${r.values.at(-1).toFixed(1)}%</span></div>`).join('')}<p class="frame-meta">${base.frames} average observed frames · ${sizeControl.value} · ${fps.toFixed(1)} fps</p></article>`;
}
function table(task, title) {
  const base = row(task, 'Qwen3.5');
  const saver = row(task, 'SAVER');
  const names = [...benchmarkNames[task], 'Macro-average'];
  return `<div class="table-scroll" tabindex="0" role="region" aria-label="${title} benchmark results"><table class="benchmark-table"><caption>${title} · ${sizeControl.value} at ${fps.toFixed(1)} fps</caption><thead><tr><th scope="col">Benchmark</th><th scope="col">Qwen3.5</th><th scope="col">SAVER</th><th scope="col">Δ points</th></tr></thead><tbody>${names.map((name, i) => { const delta = saver.values[i] - base.values[i]; return `<tr><th scope="row">${name}</th><td>${base.values[i].toFixed(1)}</td><td class="ours">${saver.values[i].toFixed(1)}</td><td class="${delta < 0 ? 'negative' : 'ours'}">${delta >= 0 ? '+' : ''}${delta.toFixed(1)}</td></tr>`; }).join('')}</tbody></table></div>`;
}
function render() {
  if (!data) return;
  document.querySelector('#result-summary').innerHTML = `<div class="summary-grid">${metric('grounding', 'Temporal grounding', 'Average mIoU over 3 benchmarks')}${metric('qa', 'Video question answering', 'Average accuracy over 6 benchmarks')}</div>`;
  document.querySelector('#benchmark-tables').innerHTML = table('grounding', 'Temporal grounding (mIoU, %)') + table('qa', 'Video QA (accuracy, %)');
}
sizeControl.addEventListener('change', render);
fpsButtons.forEach(button => button.addEventListener('click', () => {
  fps = Number(button.dataset.fps);
  fpsButtons.forEach(b => b.setAttribute('aria-pressed', String(b === button)));
  render();
}));
fetch('results.json?v=five-rates').then(response => { if (!response.ok) throw new Error('Result data unavailable'); return response.json(); }).then(json => { data = json; render(); }).catch(() => {
  document.querySelector('#result-summary').innerHTML = '<p>For 2B at 0.1 fps, SAVER reaches 30.0% average grounding mIoU versus 18.4% for Qwen3.5, and 50.8% average QA accuracy versus 45.9%. Interactive results could not load; <a href="results.json">view the result data</a> or reload the page.</p>';
  fpsButtons.forEach(b => b.disabled = true);
  sizeControl.disabled = true;
});
document.querySelector('#copy-citation').addEventListener('click', async () => {
  const text = document.querySelector('#bibtex').textContent;
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(text);
    status.textContent = 'BibTeX copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
