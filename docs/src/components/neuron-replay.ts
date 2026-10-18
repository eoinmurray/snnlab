// Playback only: the recording supplies every state and event. No neuron model runs here.
interface Recording {
  schema: 'snnlab.neuron-replay/v1';
  dt_ms: number;
  duration_ms: number;
  rest_mv: number;
  threshold_mv: number;
  capacitance_nf: number;
  tau_ms: number;
  current_na: number;
  input_departures_ms: number[];
  input_arrivals_ms: number[];
  time_ms: number[];
  voltage_mv: number[];
  spike_times_ms: number[];
}

/** Index of the last recorded value at or before t; -1 means no value yet. */
export function lastAtOrBefore(values: readonly number[], t: number): number {
  let lo = 0, hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (values[mid]! <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

export function validateRecording(value: unknown): Recording {
  if (!value || typeof value !== 'object') throw new Error('Missing recording');
  const r = value as Recording;
  if (r.schema !== 'snnlab.neuron-replay/v1') throw new Error('Unsupported recording schema');
  for (const key of ['dt_ms', 'duration_ms', 'rest_mv', 'threshold_mv', 'capacitance_nf', 'tau_ms', 'current_na'] as const) {
    if (typeof r[key] !== 'number' || !Number.isFinite(r[key])) throw new Error(`Invalid ${key}`);
  }
  if (r.dt_ms <= 0 || r.duration_ms <= 0 || r.threshold_mv <= r.rest_mv || r.capacitance_nf <= 0 || r.tau_ms <= 0) throw new Error('Invalid recording metadata');
  for (const key of ['time_ms', 'voltage_mv', 'spike_times_ms', 'input_departures_ms', 'input_arrivals_ms'] as const) {
    const a = r[key];
    if (!Array.isArray(a) || !a.every(n => typeof n === 'number' && Number.isFinite(n))) throw new Error(`Invalid ${key}`);
    if (key !== 'voltage_mv' && a.some((n, i) => n < 0 || n > r.duration_ms || (i > 0 && n <= a[i - 1]!))) throw new Error(`Invalid time order in ${key}`);
  }
  if (r.time_ms.length < 2 || r.time_ms.length !== r.voltage_mv.length || r.time_ms[0] !== 0 || r.time_ms.at(-1) !== r.duration_ms) throw new Error('Incomplete voltage samples');
  if (r.time_ms.some((t, i) => Math.abs(t - i * r.dt_ms) > 1e-6)) throw new Error('Inconsistent sample clock');
  if (r.input_departures_ms.length !== r.input_arrivals_ms.length || r.input_arrivals_ms.some((t, i) => Math.abs(t - r.input_departures_ms[i]! - 6) > 1e-6)) throw new Error('Inconsistent input travel times');
  const sampleTimes = new Set(r.time_ms);
  if (r.spike_times_ms.some(t => !sampleTimes.has(t))) throw new Error('Spike is not at a recorded integration end');
  return r;
}

async function mountReplay(host: HTMLElement, signal: AbortSignal): Promise<() => void> {
  const surface = host.querySelector<HTMLElement>('.replay-scene')!;
  const status = host.querySelector<HTMLElement>('.replay-status')!;
  const play = host.querySelector<HTMLButtonElement>('[data-play]')!;
  const replay = host.querySelector<HTMLButtonElement>('[data-replay]')!;
  const speed = host.querySelector<HTMLSelectElement>('[data-speed]')!;
  const scrub = host.querySelector<HTMLInputElement>('[data-scrub]')!;
  const timeOutput = host.querySelector<HTMLOutputElement>('[data-time]')!;
  const [pixi, response] = await Promise.all([
    import('pixi.js'),
    fetch(host.dataset.recordingUrl!, { signal }),
  ]);
  if (!response.ok) throw new Error(`Recording request failed (${response.status})`);
  const recording = validateRecording(await response.json());
  if (signal.aborted || !host.isConnected) return () => {};
  const { Application, Graphics } = pixi;
  const app = new Application();
  try {
    await app.init({ width: 800, height: 310, backgroundAlpha: 0, antialias: true,
      resolution: Math.min(window.devicePixelRatio || 1, 2), autoDensity: true, autoStart: false });
  } catch (error) {
    // A partially initialized renderer is not guaranteed to support destroy().
    try { app.destroy(true, { children: true }); } catch { /* init failed */ }
    throw error;
  }
  if (signal.aborted || !host.isConnected) {
    app.destroy(true, { children: true });
    return () => {};
  }
  const graphics = new Graphics();
  const trace = new Graphics();
  const clip = new Graphics();
  app.stage.addChild(graphics, trace, clip);
  trace.mask = clip;
  app.canvas.setAttribute('aria-hidden', 'true');
  surface.appendChild(app.canvas);
  const labels = document.createElement('div');
  labels.className = 'replay-labels';
  labels.setAttribute('aria-hidden', 'true');
  surface.appendChild(labels);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let playing = !reducedMotion.matches;
  let visible = false;
  let disposed = false;
  let clock = 0;
  let rate = 1;
  let frame = 0;
  let previousWallTime: number | undefined;
  let width = 800, height = 310;
  let palette = { blue: 0x176d91, pink: 0xbc3866, ink: 0x263343, grid: 0xd6dfe7 };
  let geometry = { cx: 170, cy: 130, radius: 48, inputX: 20, outputX: 330, gaugeH: 70, plot: { x: 420, y: 55, w: 330, h: 190 } };
  let voltageLabel: HTMLSpanElement;
  let leftLabel: HTMLSpanElement;
  let rightLabel: HTMLSpanElement;
  const spikeSet = new Set(recording.spike_times_ms);
  const topVoltage = Math.max(recording.threshold_mv + 5, ...recording.voltage_mv);
  const bottomVoltage = Math.min(recording.rest_mv, ...recording.voltage_mv);

  function label(text: string, x: number, y: number, color = ''): HTMLSpanElement {
    const node = document.createElement('span');
    node.className = `replay-label ${color}`;
    node.textContent = text;
    node.style.left = `${100 * x / width}%`;
    node.style.top = `${100 * y / height}%`;
    labels.appendChild(node);
    return node;
  }
  function voltageY(v: number) {
    const p = geometry.plot;
    return p.y + (topVoltage - v) / (topVoltage - bottomVoltage) * p.h;
  }
  function layout() {
    width = Math.max(240, surface.clientWidth);
    height = surface.clientHeight;
    const stacked = width < 600;
    const sceneWidth = stacked ? width : width * .47;
    const radius = stacked ? 43 : Math.min(51, sceneWidth * .14);
    geometry = {
      cx: sceneWidth / 2, cy: stacked ? 106 : 130, radius,
      inputX: 18, outputX: sceneWidth - 18, gaugeH: radius * 1.4,
      plot: stacked
        ? { x: 45, y: 265, w: width - 63, h: height - 308 }
        : { x: sceneWidth + 46, y: 55, w: width - sceneWidth - 64, h: height - 115 },
    };
    app.renderer.resize(width, height);
    labels.replaceChildren();
    const { cx, cy, inputX, outputX, plot, radius: r } = geometry;
    label('Input Iⱼ', inputX + 23, cy - r - 30, 'blue');
    label('Spike', outputX - 15, cy - r - 30, 'pink');
    label('Membrane voltage Vₘ', cx, cy + r + 22);
    voltageLabel = label('', cx, cy + r + 45, 'blue');
    label('Vₘ (mV)', plot.x + 1, plot.y - 25);
    for (const v of [recording.rest_mv, recording.rest_mv + 5, recording.rest_mv + 10, recording.threshold_mv, topVoltage]) {
      label(String(v).replace('-', '−'), plot.x - 24, voltageY(v));
    }
    label('Vₜₕ', plot.x + plot.w - 15, voltageY(recording.threshold_mv) - 14, 'pink');
    leftLabel = label('', plot.x, plot.y + plot.h + 21);
    label('t (ms)', plot.x + plot.w / 2, plot.y + plot.h + 21);
    rightLabel = label('', plot.x + plot.w - 8, plot.y + plot.h + 21);
    clip.clear().rect(plot.x, plot.y, plot.w, plot.h).fill(0xffffff);
    updatePalette();
  }
  function updatePalette() {
    const css = getComputedStyle(host);
    const color = (name: string, fallback: number) => {
      const value = css.getPropertyValue(name).trim();
      return /^#[0-9a-f]{6}$/i.test(value) ? Number.parseInt(value.slice(1), 16) : fallback;
    };
    palette = { blue: color('--replay-blue', 0x176d91), pink: color('--replay-pink', 0xbc3866), ink: color('--replay-ink', 0x263343), grid: color('--replay-grid', 0xd6dfe7) };
    draw();
  }
  function draw() {
    if (disposed) return;
    const { cx, cy, radius, inputX, outputX, gaugeH, plot } = geometry;
    const { blue, pink, grid } = palette;
    const index = Math.max(0, lastAtOrBefore(recording.time_ms, clock));
    const voltage = recording.voltage_mv[index]!;
    const spikeIndex = lastAtOrBefore(recording.spike_times_ms, clock);
    graphics.clear();
    graphics.moveTo(inputX, cy).lineTo(cx - radius, cy).stroke({ color: grid, width: 2 });
    graphics.moveTo(cx + radius, cy).lineTo(outputX, cy).stroke({ color: grid, width: 2 });
    graphics.moveTo(outputX - 9, cy - 6).lineTo(outputX, cy).lineTo(outputX - 9, cy + 6).stroke({ color: grid, width: 2 });
    graphics.circle(cx, cy, radius).fill({ color: blue, alpha: .035 }).stroke({ color: blue, width: 1.5 });
    if (spikeIndex >= 0) {
      const alpha = Math.max(0, 1 - (clock - recording.spike_times_ms[spikeIndex]!) / 5);
      if (alpha > 0) graphics.circle(cx, cy, radius + 6).stroke({ color: pink, width: 3, alpha });
    }
    const gaugeY = cy - gaugeH / 2;
    graphics.roundRect(cx - 13, gaugeY, 26, gaugeH, 4).fill({ color: grid, alpha: .65 });
    const fillHeight = Math.max(2, Math.min(1, Math.max(0, (voltage - recording.rest_mv) / (recording.threshold_mv - recording.rest_mv))) * gaugeH);
    graphics.rect(cx - 11, gaugeY + gaugeH - fillHeight, 22, fillHeight).fill({ color: blue, alpha: .85 });
    for (let x = cx - 21; x < cx + 21; x += 8) graphics.moveTo(x, gaugeY).lineTo(x + 4, gaugeY).stroke({ color: pink, width: 2 });
    // Departures and arrivals come from JSON; no periodic input generator exists.
    for (let i = Math.max(0, lastAtOrBefore(recording.input_departures_ms, clock - 6)); i < recording.input_departures_ms.length; i++) {
      const departure = recording.input_departures_ms[i]!;
      const arrival = recording.input_arrivals_ms[i]!;
      if (departure > clock) break;
      if (clock >= arrival) continue;
      const fraction = (clock - departure) / (arrival - departure);
      graphics.circle(inputX + fraction * (cx - radius - inputX), cy, 5).fill(blue);
    }
    for (let i = Math.max(0, lastAtOrBefore(recording.spike_times_ms, clock - 6)); i <= spikeIndex; i++) {
      const age = clock - recording.spike_times_ms[i]!;
      if (age < 0 || age >= 6) continue;
      graphics.circle(cx + radius + age / 6 * (outputX - cx - radius), cy, 5).fill(pink);
    }
    graphics.moveTo(plot.x, plot.y).lineTo(plot.x, plot.y + plot.h).lineTo(plot.x + plot.w, plot.y + plot.h).stroke({ color: grid, width: 1.5 });
    const thresholdY = voltageY(recording.threshold_mv);
    for (let x = plot.x; x < plot.x + plot.w; x += 10) graphics.moveTo(x, thresholdY).lineTo(Math.min(x + 5, plot.x + plot.w), thresholdY).stroke({ color: pink, width: 1.5 });
    const cutoff = Math.max(0, clock - 120);
    const xAt = (t: number) => plot.x + (t - cutoff) / 120 * plot.w;
    trace.clear();
    const start = Math.max(0, lastAtOrBefore(recording.time_ms, cutoff));
    trace.moveTo(xAt(recording.time_ms[start]!), voltageY(recording.voltage_mv[start]!));
    for (let i = start + 1; i <= index; i++) {
      const time = recording.time_ms[i]!;
      const x = xAt(time);
      // This threshold point is a schematic event marker, NOT a measured
      // pre-reset voltage. Recorded post-reset samples remain unchanged.
      if (spikeSet.has(time)) trace.lineTo(x, thresholdY);
      trace.lineTo(x, voltageY(recording.voltage_mv[i]!));
    }
    trace.stroke({ color: blue, width: 2 });
    for (let i = Math.max(0, lastAtOrBefore(recording.spike_times_ms, cutoff)); i <= spikeIndex; i++) {
      const t = recording.spike_times_ms[i]!;
      if (t >= cutoff) graphics.circle(xAt(t), thresholdY, 2.5).fill(pink);
    }
    voltageLabel.textContent = `${voltage.toFixed(1)} mV`;
    leftLabel.textContent = cutoff.toFixed(0);
    rightLabel.textContent = (cutoff + 120).toFixed(0);
    scrub.value = String(clock);
    scrub.setAttribute('aria-valuetext', `${clock.toFixed(2)} of ${recording.duration_ms} milliseconds`);
    timeOutput.value = `${clock.toFixed(2)} / ${recording.duration_ms} ms`;
    host.dataset.time = String(clock);
    host.dataset.voltage = String(voltage);
    host.dataset.spikes = String(spikeIndex + 1);
    app.render();
  }
  function syncPlayback() {
    cancelAnimationFrame(frame);
    frame = 0;
    previousWallTime = undefined;
    play.textContent = playing ? 'Pause' : 'Play';
    play.setAttribute('aria-label', playing ? 'Pause neuron recording' : 'Play neuron recording');
    if (playing && visible && !document.hidden && !disposed) frame = requestAnimationFrame(tick);
  }
  function tick(now: number) {
    if (disposed) return;
    // One recording clock drives the samples, pulses, events, plot and scrubber.
    if (previousWallTime !== undefined) clock = Math.min(recording.duration_ms, clock + Math.min(100, now - previousWallTime) * .04 * rate);
    previousWallTime = now;
    draw();
    if (clock >= recording.duration_ms) { playing = false; syncPlayback(); }
    else frame = requestAnimationFrame(tick);
  }
  const on = (target: EventTarget, event: string, handler: EventListener) => target.addEventListener(event, handler, { signal });
  on(play, 'click', () => {
    if (!playing && clock >= recording.duration_ms) clock = 0;
    playing = !playing;
    draw(); syncPlayback();
  });
  on(replay, 'click', () => {
    clock = 0;
    // A reduced-motion user must explicitly press Play to start motion.
    playing = !reducedMotion.matches;
    draw(); syncPlayback();
  });
  on(speed, 'change', () => { rate = Number(speed.value); previousWallTime = undefined; });
  on(scrub, 'input', () => {
    clock = Math.max(0, Math.min(recording.duration_ms, Number(scrub.value)));
    playing = false;
    draw(); syncPlayback();
  });
  on(document, 'visibilitychange', syncPlayback);
  on(reducedMotion, 'change', () => { if (reducedMotion.matches) { playing = false; syncPlayback(); } });
  const visibility = new IntersectionObserver(entries => {
    visible = entries.some(entry => entry.isIntersecting);
    syncPlayback();
  });
  visibility.observe(host);
  const resize = new ResizeObserver(layout);
  resize.observe(surface);
  const theme = new MutationObserver(updatePalette);
  theme.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme', 'style'] });
  theme.observe(document.body, { attributes: true, attributeFilter: ['class', 'data-theme', 'style'] });
  const dark = matchMedia('(prefers-color-scheme: dark)');
  on(dark, 'change', updatePalette);
  scrub.max = String(recording.duration_ms);
  scrub.step = String(recording.dt_ms);
  function dispose() {
    if (disposed) return;
    disposed = true;
    cancelAnimationFrame(frame);
    visibility.disconnect(); resize.disconnect(); theme.disconnect();
    labels.remove();
    app.destroy(true, { children: true });
  }
  // Aborting also cleans up if a late layout/render error prevents returning.
  signal.addEventListener('abort', dispose, { once: true });
  layout();
  status.hidden = true;
  for (const control of [play, replay, speed, scrub]) control.disabled = false;
  syncPlayback();
  return dispose;
}

class NeuronReplay extends HTMLElement {
  private controller?: AbortController;
  private observer?: IntersectionObserver;
  private dispose?: () => void;
  connectedCallback() {
    if (this.controller) return;
    const controller = new AbortController();
    this.controller = controller;
    let started = false;
    const start = async () => {
      if (started || controller.signal.aborted) return;
      started = true;
      this.observer?.disconnect();
      try {
        const dispose = await mountReplay(this, controller.signal);
        if (controller.signal.aborted || !this.isConnected) dispose();
        else this.dispose = dispose;
      } catch (error) {
        if (controller.signal.aborted) return;
        controller.abort();
        console.error('Neuron recording could not be displayed', error);
        const status = this.querySelector<HTMLElement>('.replay-status');
        if (status) {
          status.hidden = false;
          status.textContent = 'The recording could not be displayed. Current pulses raise voltage; reaching threshold emits a spike and resets the neuron. Try reloading this page to retry.';
        }
      }
    };
    if (typeof IntersectionObserver === 'undefined') void start();
    else {
      this.observer = new IntersectionObserver(entries => {
        if (entries.some(entry => entry.isIntersecting)) void start();
      }, { rootMargin: '200px' });
      this.observer.observe(this);
    }
  }
  disconnectedCallback() {
    this.observer?.disconnect();
    this.controller?.abort();
    this.dispose?.();
    this.dispose = undefined;
    this.controller = undefined;
    const status = this.querySelector<HTMLElement>('.replay-status');
    if (status) { status.hidden = false; status.textContent = 'Loading recorded neuron…'; }
    this.querySelectorAll<HTMLButtonElement | HTMLInputElement | HTMLSelectElement>('button, input, select').forEach(control => { control.disabled = true; });
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('snnlab-neuron-replay')) customElements.define('snnlab-neuron-replay', NeuronReplay);
