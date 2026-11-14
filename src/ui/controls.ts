// Lightweight DOM control helpers. Every numeric control is bounded and clamps on
// input, so the renderer never receives out-of-range values.

export interface SliderOpts {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  format?: (v: number) => string;
  onInput: (v: number) => void;
}

export function slider(opts: SliderOpts): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'control';
  const row = document.createElement('div');
  row.className = 'row';
  const label = document.createElement('label');
  label.textContent = opts.label;
  const val = document.createElement('span');
  val.className = 'val';
  const fmt = opts.format ?? ((v: number) => v.toFixed(2));
  val.textContent = fmt(opts.value);
  row.append(label, val);

  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(opts.min);
  input.max = String(opts.max);
  input.step = String(opts.step);
  input.value = String(opts.value);
  const fill = () => {
    const pct = ((+input.value - opts.min) / (opts.max - opts.min)) * 100;
    input.style.setProperty('--fill', `${pct}%`);
  };
  fill();
  input.addEventListener('input', () => {
    const v = clamp(+input.value, opts.min, opts.max);
    val.textContent = fmt(v);
    fill();
    opts.onInput(v);
  });

  wrap.append(row, input);
  return wrap;
}

export function colorPicker(label: string, value: string, onInput: (hex: string) => void): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'control';
  const row = document.createElement('div');
  row.className = 'row';
  const l = document.createElement('label');
  l.textContent = label;
  const val = document.createElement('span');
  val.className = 'val';
  val.textContent = value;
  row.append(l, val);
  const input = document.createElement('input');
  input.type = 'color';
  input.value = value;
  input.addEventListener('input', () => {
    val.textContent = input.value;
    onInput(input.value);
  });
  wrap.append(row, input);
  return wrap;
}

export function checkbox(label: string, value: boolean, onInput: (v: boolean) => void): HTMLElement {
  const wrap = document.createElement('label');
  wrap.className = 'check';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = value;
  input.addEventListener('change', () => onInput(input.checked));
  const span = document.createElement('span');
  span.textContent = label;
  wrap.append(input, span);
  return wrap;
}

export function button(label: string, onclick: () => void, cls = 'btn'): HTMLButtonElement {
  const b = document.createElement('button');
  b.className = cls;
  b.type = 'button';
  b.textContent = label;
  b.addEventListener('click', onclick);
  return b;
}

export function setSliderValue(el: HTMLElement, v: number): void {
  const input = el.querySelector('input[type="range"]') as HTMLInputElement | null;
  if (!input) return;
  input.value = String(v);
  input.dispatchEvent(new Event('input'));
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

export function labeledBlock(title: string): HTMLElement {
  const d = document.createElement('div');
  d.className = 'light-block';
  const h = document.createElement('h3');
  h.textContent = title;
  d.append(h);
  return d;
}
