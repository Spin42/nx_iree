// CPU arithmetic oracle only; this does not execute or validate GPU code.
const fs = require('node:fs');

function aggregate(rows, p1, p2, mode) {
  let previous = rows[0].map(() => 0);
  return rows.map(cost => {
    const minimum = Math.min(...previous);
    const next = cost.map((value, d) => {
      if (mode === 'safe-int32') {
        const own = (previous[d] - minimum) | 0;
        const lower = d > 0 ? (Math.min((previous[d - 1] - minimum) | 0, p2) + p1) | 0 : p2;
        const upper = d + 1 < previous.length ? (Math.min((previous[d + 1] - minimum) | 0, p2) + p1) | 0 : p2;
        return (value + Math.min(own, lower, upper, p2)) | 0;
      }
      const add = mode === 'original-int32' ? (a, b) => (a + b) | 0 : (a, b) => a + b;
      const choices = [previous[d], add(minimum, p2)];
      if (d > 0) choices.push(add(previous[d - 1], p1));
      else if (mode !== 'reference') choices.push(add(1000000000, p1));
      if (d + 1 < previous.length) choices.push(add(previous[d + 1], p1));
      else if (mode !== 'reference') choices.push(add(1000000000, p1));
      const transition = Math.min(...choices);
      return mode === 'original-int32' ? (add(value, transition) - minimum) | 0 : value + transition - minimum;
    });
    previous = next;
    return next;
  });
}

const cases = [
  {name: 'overflow before minimum', shape: [3, 1, 2], rows: [[0, 999999999], [999999999, 999999999], [0, 0]], p1: 999999999, p2: 999999999},
  {name: 'finite missing-neighbor sentinel', shape: [3, 1, 4], rows: [[0, 999999999, 999999999, 999999999], [999999999, 999999999, 999999999, 999999999], [0, 0, 0, 0]], p1: 0, p2: 999999999}
];
const results = cases.map(test => ({...test, reference: aggregate(test.rows, test.p1, test.p2, 'reference'), originalUnbounded: aggregate(test.rows, test.p1, test.p2, 'original-unbounded'), originalInt32: aggregate(test.rows, test.p1, test.p2, 'original-int32'), safeInt32: aggregate(test.rows, test.p1, test.p2, 'safe-int32')}));
for (const result of results) {
  if (JSON.stringify(result.reference) === JSON.stringify(result.originalInt32)) throw Error('Original formula did not reproduce: ' + result.name);
  if (JSON.stringify(result.reference) !== JSON.stringify(result.safeInt32)) throw Error('Normalized formula differs: ' + result.name);
}

let seed = 0x57e2c18b;
const random = () => {seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed;};
const edges = [0, 1, 2, 31, 32, 99999999, 499999999, 500000000, 999999998, 999999999];
const value = () => random() % 3 ? edges[random() % edges.length] : random() % 1000000000;
const dimensions = [1, 2, 3, 4, 31, 32, 33, 47, 63, 64, 65, 95, 96, 97, 127, 128, 129];
let p1GreaterP2 = 0;
let valuesCompared = 0;
for (let test = 0; test < 2000; ++test) {
  const disparities = dimensions[test % dimensions.length];
  const steps = 1 + random() % 20;
  const p1 = value(), p2 = value();
  if (p1 > p2) p1GreaterP2++;
  const rows = Array.from({length: steps}, () => Array.from({length: disparities}, value));
  const reference = aggregate(rows, p1, p2, 'reference');
  const safe = aggregate(rows, p1, p2, 'safe-int32');
  for (let step = 0; step < steps; ++step) for (let d = 0; d < disparities; ++d) {
    valuesCompared++;
    if (reference[step][d] !== safe[step][d]) throw Error(JSON.stringify({test, disparities, steps, p1, p2, rows, step, d, expected: reference[step][d], actual: safe[step][d]}));
    if (safe[step][d] < 0 || safe[step][d] >= 2000000000) throw Error('Range invariant violated');
  }
}
const report = {evidence: 'CPU exact reference and explicitly wrapping int32 simulation; no GPU execution', cases: results, normalizedRandomOracle: {seed: '0x57e2c18b', tests: 2000, p1GreaterP2, valuesCompared, dimensions, result: 'PASS'}};
fs.writeFileSync('/tmp/stereo-opt-arithmetic-red.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
