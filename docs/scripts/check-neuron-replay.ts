import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// Exercise pure playback logic without starting a browser or GPU renderer.
Object.defineProperty(globalThis, 'HTMLElement', { value: class {}, configurable: true });
const { lastAtOrBefore, validateRecording } = await import('../src/components/neuron-replay');
const recording = JSON.parse(readFileSync(new URL('../public/animated-neuron/recording.json', import.meta.url), 'utf8'));
const original = JSON.stringify(recording);
validateRecording(recording);
let checks = 0;
for (const values of [recording.time_ms, recording.spike_times_ms, recording.input_departures_ms, recording.input_arrivals_ms] as number[][]) {
  const times = [-1, 0, recording.duration_ms, recording.duration_ms + 1,
    ...values.flatMap(t => [t - 0.001, t, t + 0.001])];
  // Repeated reverse seeking must give exactly the same states as forward play.
  for (const t of [...times, ...times.toReversed()]) {
    let expected = -1;
    for (let i = 0; i < values.length && values[i]! <= t; i++) expected = i;
    assert.equal(lastAtOrBefore(values, t), expected);
    checks++;
  }
}
assert.equal(lastAtOrBefore([], 10), -1);
const malformed = [
  { ...recording, schema: 'unknown' },
  { ...recording, dt_ms: 0 },
  { ...recording, voltage_mv: [NaN] },
  { ...recording, time_ms: [0, 1, 0] },
  { ...recording, spike_times_ms: [0.123] },
  { ...recording, input_arrivals_ms: [] },
  { ...recording, threshold_mv: recording.rest_mv },
];
for (const value of malformed) assert.throws(() => validateRecording(value));
assert.equal(JSON.stringify(recording), original);
console.log(`Replay verified: ${checks} seek lookups; ${malformed.length} malformed recordings rejected; source unchanged.`);
