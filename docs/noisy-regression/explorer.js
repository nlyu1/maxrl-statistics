(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const value = (id) => Number(byId(id).value);
  const fixed = (x, digits = 3) => Number(x).toFixed(digits);
  const compact = (x) => Math.abs(x) !== 0 && (Math.abs(x) < 0.001 || Math.abs(x) >= 10000)
    ? x.toExponential(2) : Number(x.toPrecision(4)).toString();
  const ink = "#18334c", teal = "#087f8c", orange = "#ba5511";
  const canonical = Object.freeze({ sigma: 0.25, range: 4, levels: 256 });
  const delta = 2 * canonical.range / (canonical.levels - 1);

  function randomSource(seed) {
    let state = seed >>> 0;
    const uniform = () => {
      state += 0x6D2B79F5;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
    return { uniform, normal };
  }

  function taskDraw(seed, count) {
    const rng = randomSource(seed);
    const w = [rng.normal() / Math.sqrt(2), rng.normal() / Math.sqrt(2)];
    const point = () => ({ x: [rng.normal(), rng.normal()], z: rng.normal() });
    return { w, rows: Array.from({ length: count }, point), query: point() };
  }

  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  function quantize(z, range = canonical.range, levels = canonical.levels) {
    if (!Number.isFinite(z)) throw new Error("Cannot quantize a nonfinite scalar");
    const spacing = 2 * range / (levels - 1);
    let lo = 0, hi = levels - 1;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (z >= -range + (mid + 0.5) * spacing) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  const decode = (z) => -canonical.range + quantize(z) * delta;
  function digits(z) {
    const q = quantize(z);
    return [Math.floor(q / 16).toString(16).toUpperCase(), (q % 16).toString(16).toUpperCase()];
  }

  function normalSurvival(z) {
    if (z === Infinity) return 0;
    if (z === -Infinity) return 1;
    if (z < 0) return 1 - normalSurvival(-z);
    const t = 1 / (1 + 0.2316419 * z);
    const polynomial = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI) * polynomial;
  }
  function normalMass(lo, hi) {
    if (lo >= 0) return Math.max(0, normalSurvival(lo) - normalSurvival(hi));
    if (hi <= 0) return Math.max(0, normalSurvival(-hi) - normalSurvival(-lo));
    return Math.max(0, 1 - normalSurvival(-lo) - normalSurvival(hi));
  }

  function svgFrame(id, width, height, title, body) {
    byId(id).innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${id}-title"><title id="${id}-title">${title}</title>${body}</svg>`;
  }
  const text = (x, y, label, attrs = "") => `<text x="${x}" y="${y}" ${attrs}>${label}</text>`;
  const line = (x1, y1, x2, y2, attrs = "") => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
  function path(points, attrs = "") {
    return `<path d="${points.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ")}" ${attrs}/>`;
  }
  function marker(x, y, kind, color, title, radius = 6) {
    const points = kind === "star"
      ? Array.from({ length: 10 }, (_, i) => {
        const a = i * Math.PI / 5 - Math.PI / 2, r = i % 2 ? radius * 0.43 : radius;
        return `${x + r * Math.cos(a)},${y + r * Math.sin(a)}`;
      }).join(" ")
      : `${x},${y - radius} ${x + radius},${y} ${x},${y + radius} ${x - radius},${y}`;
    return `<polygon points="${points}" fill="${color}" stroke="white" stroke-width="1"><title>${title}</title></polygon>`;
  }
  function squareAxes(xmin, xmax, ymin, ymax, xLabel, yLabel) {
    const left = 88, top = 22, size = 314;
    const X = (x) => left + (x - xmin) / (xmax - xmin) * size;
    const Y = (y) => top + (ymax - y) / (ymax - ymin) * size;
    let axes = "";
    for (let i = 0; i <= 4; i++) {
      const x = xmin + (xmax - xmin) * i / 4, y = ymin + (ymax - ymin) * i / 4;
      axes += line(X(x), top, X(x), top + size, 'stroke="#e3eaf0"');
      axes += line(left, Y(y), left + size, Y(y), 'stroke="#e3eaf0"');
      const tick = (v, range) => Number(v.toFixed(Math.min(8, Math.max(0, Math.ceil(-Math.log10(range / 4)) + 1)))).toString();
      axes += text(X(x), 356, tick(x, xmax - xmin), 'text-anchor="middle" style="font-size:11px"');
      axes += text(left - 8, Y(y) + 4, tick(y, ymax - ymin), 'text-anchor="end" style="font-size:11px"');
    }
    axes += `<rect x="${left}" y="${top}" width="${size}" height="${size}" fill="none" stroke="#a5b7c2"/>`;
    axes += text(left + size / 2, 378, xLabel, 'text-anchor="middle" class="nr-axis-label"');
    axes += text(14, top + size / 2, yLabel, `text-anchor="middle" class="nr-axis-label" transform="rotate(-90 14 ${top + size / 2})"`);
    return { X, Y, axes, left, top, size };
  }
  function signalColor(s) {
    const t = Math.min(1, Math.abs(s) / 3);
    const start = [247, 249, 246], end = s < 0 ? [39, 103, 159] : [191, 82, 28];
    return `rgb(${start.map((c, i) => Math.round(c + t * (end[i] - c))).join(",")})`;
  }

  function updatePopulation() {
    const dimension = value("nr-dimension"), conditional = byId("nr-pop-mode").value === "conditional";
    byId("nr-dimension").disabled = conditional;
    const rng = randomSource(73911), count = 12000, bins = 80, lo = -5, hi = 5, spacing = (hi - lo) / bins;
    const histogram = Array(bins).fill(0);
    let sum = 0, sum2 = 0, outside = 0;
    for (let i = 0; i < count; i++) {
      let s = 0;
      if (conditional) s = rng.normal();
      else for (let j = 0; j < dimension; j++) s += rng.normal() * rng.normal() / Math.sqrt(dimension);
      sum += s; sum2 += s * s;
      const bin = Math.floor((s - lo) / spacing);
      if (bin >= 0 && bin < bins) histogram[bin]++;
      else outside++;
    }
    const canvasWidth = Math.max(300, Math.min(790, byId("nr-pop-chart").clientWidth));
    const narrow = canvasWidth < 500;
    const left = 55, top = 18, width = canvasWidth - 85, height = 220;
    const ymax = Math.max(0.8, ...histogram.map((v) => v / (count * spacing))) * 1.08;
    const X = (x) => left + (x - lo) / (hi - lo) * width;
    const Y = (y) => top + height * (1 - y / ymax);
    let body = "";
    for (let i = 0; i <= 4; i++) {
      const y = ymax * i / 4;
      body += line(left, Y(y), left + width, Y(y), 'stroke="#e4ebef"');
      body += text(left - 7, Y(y) + 4, fixed(y, 2), 'text-anchor="end"');
    }
    histogram.forEach((v, i) => {
      const density = v / (count * spacing);
      body += `<rect x="${X(lo + i * spacing)}" y="${Y(density)}" width="${width / bins - 0.7}" height="${height * density / ymax}" fill="${teal}" opacity="0.7"><title>Signal in [${fixed(lo + i * spacing, 2)}, ${fixed(lo + (i + 1) * spacing, 2)}): ${v} samples</title></rect>`;
    });
    const normal = [], laplace = [];
    for (let i = 0; i <= 400; i++) {
      const x = lo + (hi - lo) * i / 400;
      normal.push([X(x), Y(Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI))]);
      laplace.push([X(x), Y(Math.exp(-Math.sqrt(2) * Math.abs(x)) / Math.sqrt(2))]);
    }
    body += path(normal, `fill="none" stroke="${ink}" stroke-width="2.5" stroke-dasharray="6 4"`);
    if (!conditional && dimension === 2) body += path(laplace, `fill="none" stroke="${orange}" stroke-width="2"`);
    for (let x = narrow ? -4 : -5; x <= 5; x += narrow ? 2 : 1) body += text(X(x), 258, x, 'text-anchor="middle"');
    body += text(left + width / 2, 281, "Clean signal s", 'text-anchor="middle" class="nr-axis-label"');
    body += text(16, 125, "Density", 'text-anchor="middle" transform="rotate(-90 16 125)"');
    body += text(narrow ? 25 : 55, 308, `Teal: samples · Dashed: N(0, 1)${!narrow && !conditional && dimension === 2 ? " · Orange: exact d = 2 Laplace law" : ""}`, 'class="nr-legend"');
    if (narrow && !conditional && dimension === 2) body += text(25, 326, "Orange: exact d = 2 Laplace law", 'class="nr-legend"');
    svgFrame("nr-pop-chart", canvasWidth, narrow ? 342 : 325, "Signal histogram and Gaussian comparison", body);
    const variance = sum2 / count - (sum / count) ** 2;
    byId("nr-pop-readout").textContent = `${conditional ? "Fixed unit-norm rule: exactly Gaussian for every d." : `Across tasks, d = ${dimension}: theoretical excess kurtosis = ${compact(6 / dimension)} (Gaussian: 0).`} Population mean = 0; variance = 1. This draw: mean ${fixed(sum / count)}, variance ${fixed(variance)}. ${outside} of ${count.toLocaleString("en-US")} signals lie outside the plotted [−5, 5]; histogram normalization includes them.`;
  }

  function updateQuantization() {
    const sigma = 10 ** value("nr-q-sigma"), phase = value("nr-phase"), range = value("nr-range"), levels = value("nr-levels");
    const spacing = 2 * range / (levels - 1), base = Math.floor(levels / 2), center = (k) => -range + k * spacing;
    const signal = center(base) + phase * spacing;
    const probabilities = Array.from({ length: levels }, (_, k) => {
      const lower = k === 0 ? -Infinity : center(k) - spacing / 2;
      const upper = k === levels - 1 ? Infinity : center(k) + spacing / 2;
      return normalMass((lower - signal) / sigma, (upper - signal) / sigma);
    });
    const targetBin = quantize(signal, range, levels), span = Math.max(2.2 * spacing, 3.8 * sigma);
    const canvasWidth = Math.max(300, Math.min(790, byId("nr-q-chart").clientWidth));
    const narrow = canvasWidth < 500;
    const xmin = signal - span, xmax = signal + span, left = 50, width = canvasWidth - 80;
    const X = (x) => left + (x - xmin) / (xmax - xmin) * width;
    const topBase = 157, bottomBase = 322;
    let body = "";
    for (const k of [0, 0.5, 1]) {
      body += line(left, bottomBase - 118 * k, left + width, bottomBase - 118 * k, 'stroke="#e2eaf0"');
      body += text(left - 8, bottomBase - 118 * k + 4, k, 'text-anchor="end"');
    }
    const visibleStep = Math.max(1, Math.ceil(Math.min(levels, 2 * span / spacing) / 50));
    for (let k = 0; k < levels; k++) {
      const lower = k === 0 ? xmin : center(k) - spacing / 2;
      const upper = k === levels - 1 ? xmax : center(k) + spacing / 2;
      if (upper < xmin || lower > xmax) continue;
      const x1 = X(Math.max(xmin, lower)), x2 = X(Math.min(xmax, upper));
      body += `<rect x="${x1 + 0.25}" y="${bottomBase - 118 * probabilities[k]}" width="${Math.max(0.1, x2 - x1 - 0.5)}" height="${118 * probabilities[k]}" fill="${k === targetBin ? orange : teal}" opacity="0.85"><title>Bin ${k}; center ${fixed(center(k), 6)}; probability ${probabilities[k].toPrecision(6)}${k === 0 || k === levels - 1 ? "; includes the entire endpoint tail" : ""}</title></rect>`;
      if (k > 0 && lower >= xmin && lower <= xmax && k % visibleStep === 0) body += line(X(lower), 22, X(lower), topBase, 'stroke="#adbcc5" stroke-dasharray="3 4"');
    }
    const peak = 1 / (sigma * Math.sqrt(2 * Math.PI));
    const density = Array.from({ length: 601 }, (_, i) => {
      const x = xmin + (xmax - xmin) * i / 600;
      return [X(x), topBase - 118 * Math.exp(-0.5 * ((x - signal) / sigma) ** 2)];
    });
    body += path(density, `fill="none" stroke="${teal}" stroke-width="2.5"`);
    body += line(left, topBase, left + width, topBase, 'stroke="#adbcc5"');
    body += line(X(signal), 20, X(signal), bottomBase, `stroke="${orange}" stroke-width="1.5" stroke-dasharray="5 4"`);
    body += text(left + 7, 18, `Gaussian density (peak ${compact(peak)})`, 'class="nr-legend"');
    body += text(left + 7, 191, "Probability per output bin", 'class="nr-legend"');
    const ticks = narrow ? 2 : 4;
    for (let i = 0; i <= ticks; i++) {
      const x = xmin + (xmax - xmin) * i / ticks;
      body += text(X(x), 345, compact(x), 'text-anchor="middle"');
    }
    body += text(canvasWidth / 2, 367, "Continuous noisy value y = s + ε", 'text-anchor="middle" class="nr-axis-label"');
    body += text(narrow ? 20 : 50, 393, narrow ? "Orange line: clean s · Orange bar: q(s)" : "Orange line: clean s · Orange bar: q(s) · Gray lines: bin boundaries", 'class="nr-legend"');
    if (narrow) body += text(20, 413, "Gray lines: bin boundaries", 'class="nr-legend"');
    svgFrame("nr-q-chart", canvasWidth, narrow ? 431 : 410, "Gaussian noise density and output-bin probabilities", body);
    byId("nr-q-sigma-value").textContent = compact(sigma);
    byId("nr-phase-value").textContent = fixed(phase, 2);
    byId("nr-q-readout").textContent = `Δ = ${compact(spacing)}; Δ/σ = ${compact(spacing / sigma)}; rounding scale Δ/√12 = ${compact(spacing / Math.sqrt(12))}. Clean s = ${fixed(signal, 6)}. Probability of retaining its noiseless bin: ${(100 * probabilities[targetBin]).toFixed(3)}%. ${levels === canonical.levels && range === canonical.range ? "Canonical scalar codec." : "Counterfactual codec: the training vocabulary uses 256 levels on [−4, 4]."}`;
    byId("nr-q-range-note").textContent = `Population tails outside ±${range}: input coordinate ${(200 * normalSurvival(range)).toFixed(3)}%; d = 2 clean signal ${(100 * Math.exp(-Math.sqrt(2) * range)).toFixed(3)}%. These are range exceedances, not endpoint-bin probabilities. Probability calculations use the Gaussian CDF formula (numerical approximation); tallies can round to 100%.`;
    byId("nr-quantization").dataset.sameBinProbability = probabilities[targetBin];
    byId("nr-quantization").dataset.probabilitySum = probabilities.reduce((a, b) => a + b, 0);
  }

  function posterior(rows, sigma) {
    let a = 2 * sigma * sigma, b = 0, c = a, u = 0, v = 0;
    for (const row of rows) {
      a += row.x[0] ** 2; b += row.x[0] * row.x[1]; c += row.x[1] ** 2;
      u += row.x[0] * row.y; v += row.x[1] * row.y;
    }
    const determinant = a * c - b * b;
    if (!(determinant > 0)) throw new Error("Posterior precision must be positive definite");
    return {
      mean: [(c * u - b * v) / determinant, (a * v - b * u) / determinant],
      covariance: [sigma * sigma * c / determinant, -sigma * sigma * b / determinant, sigma * sigma * a / determinant],
    };
  }
  function contour(w, level, bound) {
    const candidates = [];
    if (Math.abs(w[1]) > 1e-12) for (const x of [-bound, bound]) {
      const y = (level - w[0] * x) / w[1];
      if (Math.abs(y) <= bound) candidates.push([x, y]);
    }
    if (Math.abs(w[0]) > 1e-12) for (const y of [-bound, bound]) {
      const x = (level - w[1] * y) / w[0];
      if (Math.abs(x) <= bound) candidates.push([x, y]);
    }
    return candidates;
  }

  let geometrySeed = 61739;
  let geometryTask = taskDraw(geometrySeed, 128);
  function setRuleControls(w) {
    byId("nr-angle").step = "any";
    byId("nr-norm").step = "any";
    byId("nr-angle").value = Math.atan2(w[1], w[0]) * 180 / Math.PI;
    const norm = Math.hypot(...w);
    byId("nr-norm").min = Math.min(0.05, norm);
    byId("nr-norm").max = Math.max(3, norm);
    byId("nr-norm").value = norm;
  }
  function updateGeometry() {
    const n = value("nr-n"), sigma = 10 ** value("nr-g-sigma"), angle = value("nr-angle") * Math.PI / 180, norm = value("nr-norm");
    const w = [norm * Math.cos(angle), norm * Math.sin(angle)], showDecoded = byId("nr-show-quantized").checked;
    const rows = geometryTask.rows.slice(0, n).map((r) => ({ x: r.x, s: dot(w, r.x), y: dot(w, r.x) + sigma * r.z }));
    const query = geometryTask.query, querySignal = dot(w, query.x), queryY = querySignal + sigma * query.z;
    const continuous = posterior(rows, sigma), decodedRows = rows.map((r) => ({ x: r.x.map(decode), y: decode(r.y) }));
    const decoded = posterior(decodedRows, sigma);
    const bound = Math.ceil(Math.max(3, ...geometryTask.rows.flatMap((r) => r.x.map(Math.abs)), ...query.x.map(Math.abs)) * 2) / 2;
    const plane = squareAxes(-bound, bound, -bound, bound, "x₁", "x₂");
    let body = "";
    const tiles = 35, tile = 2 * bound / tiles;
    for (let i = 0; i < tiles; i++) for (let j = 0; j < tiles; j++) {
      const x = -bound + i * tile, y = -bound + j * tile;
      body += `<rect x="${plane.X(x)}" y="${plane.Y(y + tile)}" width="${plane.size / tiles + 0.3}" height="${plane.size / tiles + 0.3}" fill="${signalColor(dot(w, [x + tile / 2, y + tile / 2]))}" opacity="0.55"/>`;
    }
    body += plane.axes;
    for (const level of [-2, -1, 0, 1, 2]) {
      const points = contour(w, level, bound);
      if (points.length >= 2) body += line(plane.X(points[0][0]), plane.Y(points[0][1]), plane.X(points[1][0]), plane.Y(points[1][1]), `stroke="${ink}" opacity="${level === 0 ? 0.6 : 0.22}" stroke-width="${level === 0 ? 1.5 : 1}"`);
    }
    rows.forEach((row, i) => {
      const x = plane.X(row.x[0]), y = plane.Y(row.x[1]);
      body += `<circle cx="${x}" cy="${y}" r="4" fill="${signalColor(row.y)}" stroke="${ink}" stroke-width="0.7"><title>Observation ${i + 1}: x = (${fixed(row.x[0], 5)}, ${fixed(row.x[1], 5)}); clean s = ${fixed(row.s, 5)}; noisy y = ${fixed(row.y, 5)}</title></circle>`;
      if (showDecoded) {
        const dx = plane.X(decodedRows[i].x[0]), dy = plane.Y(decodedRows[i].x[1]);
        body += line(x, y, dx, dy, `stroke="${orange}" stroke-width="1"`);
        body += `<rect x="${dx - 3}" y="${dy - 3}" width="6" height="6" fill="${signalColor(decodedRows[i].y)}" stroke="${orange}" stroke-width="1"><title>Decoded observation ${i + 1}: x = (${fixed(decodedRows[i].x[0], 5)}, ${fixed(decodedRows[i].x[1], 5)}); y = ${fixed(decodedRows[i].y, 5)}</title></rect>`;
      }
    });
    const arrowLength = Math.min(1.4, 0.5 * bound), end = w.map((a) => a / norm * arrowLength);
    body += `<defs><marker id="nr-rule-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="${ink}"/></marker></defs>`;
    body += line(plane.X(0), plane.Y(0), plane.X(end[0]), plane.Y(end[1]), `stroke="${ink}" stroke-width="2" marker-end="url(#nr-rule-arrow)"`);
    body += text(plane.X(end[0]) + 8, plane.Y(end[1]) - 8, "w direction");
    body += marker(plane.X(query.x[0]), plane.Y(query.x[1]), "star", ink, `Query x = (${fixed(query.x[0], 5)}, ${fixed(query.x[1], 5)}); clean s = ${fixed(querySignal, 5)}`, 9);
    if (showDecoded) body += marker(plane.X(decode(query.x[0])), plane.Y(decode(query.x[1])), "diamond", orange, "Decoded query input", 4);
    body += text(48, 402, "Circles: noisy observations · Star: query", 'class="nr-legend"');
    body += text(48, 422, showDecoded ? "Squares: decoded x and y · Blue −3 → orange +3" : "Background: clean s · Blue −3 → orange +3", 'class="nr-legend"');
    svgFrame("nr-input-chart", 440, 440, "Two-dimensional input plane, observations, query and hidden rule", body);

    const [a, b, c] = continuous.covariance;
    const disc = Math.hypot(a - c, 2 * b), eig1 = (a + c + disc) / 2;
    const eig2 = Math.max(0, (a * c - b * b) / eig1);
    const theta = 0.5 * Math.atan2(2 * b, a - c), radius = Math.sqrt(-2 * Math.log(0.05));
    const r1 = radius * Math.sqrt(eig1), r2 = radius * Math.sqrt(eig2);
    const ellipse = Array.from({ length: 181 }, (_, i) => {
      const t = i / 180 * 2 * Math.PI, u = r1 * Math.cos(t), v = r2 * Math.sin(t);
      return [continuous.mean[0] + u * Math.cos(theta) - v * Math.sin(theta), continuous.mean[1] + u * Math.sin(theta) + v * Math.cos(theta)];
    });
    const anchors = [...ellipse, w, ...(showDecoded ? [decoded.mean] : [])];
    const xmin = Math.min(...anchors.map((p) => p[0])), xmax = Math.max(...anchors.map((p) => p[0]));
    const ymin = Math.min(...anchors.map((p) => p[1])), ymax = Math.max(...anchors.map((p) => p[1]));
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2, half = Math.max(xmax - xmin, ymax - ymin, 1e-6) * 0.66;
    const coefficients = squareAxes(cx - half, cx + half, cy - half, cy + half, "w₁ (auto zoom)", "w₂");
    let posteriorBody = coefficients.axes;
    posteriorBody += path(ellipse.map((p) => [coefficients.X(p[0]), coefficients.Y(p[1])]), `fill="${teal}" fill-opacity="0.16" stroke="${teal}" stroke-width="2"`);
    posteriorBody += `<circle cx="${coefficients.X(continuous.mean[0])}" cy="${coefficients.Y(continuous.mean[1])}" r="4" fill="${teal}"><title>Continuous posterior mean (${fixed(continuous.mean[0], 7)}, ${fixed(continuous.mean[1], 7)})</title></circle>`;
    posteriorBody += marker(coefficients.X(w[0]), coefficients.Y(w[1]), "star", ink, `Hidden w = (${fixed(w[0], 7)}, ${fixed(w[1], 7)})`, 8);
    if (showDecoded) posteriorBody += marker(coefficients.X(decoded.mean[0]), coefficients.Y(decoded.mean[1]), "diamond", orange, `Decoded-input ridge mean (${fixed(decoded.mean[0], 7)}, ${fixed(decoded.mean[1], 7)})`, 6);
    posteriorBody += text(48, 402, "Teal: continuous posterior (95%) · Star: true w", 'class="nr-legend"');
    posteriorBody += text(48, 422, showDecoded ? "Orange diamond: decoded-input ridge mean" : `Ellipse semi-axes: ${compact(r1)}, ${compact(r2)}`, 'class="nr-legend"');
    svgFrame("nr-coefficient-chart", 440, 440, "Continuous Bayesian posterior over the two coefficients", posteriorBody);

    const predictiveVariance = a * query.x[0] ** 2 + 2 * b * query.x[0] * query.x[1] + c * query.x[1] ** 2;
    byId("nr-n-value").textContent = n;
    byId("nr-g-sigma-value").textContent = compact(sigma);
    byId("nr-angle-value").textContent = `${fixed(value("nr-angle"), 1)}°`;
    byId("nr-norm-value").textContent = compact(norm);
    byId("nr-g-readout").textContent = `w = (${fixed(w[0])}, ${fixed(w[1])}); this rule's signal variance = ${compact(norm * norm)}; population signal variance = 1. Query clean s = ${fixed(querySignal, 5)}, noisy y = ${fixed(queryY, 5)}. Continuous posterior mean prediction = ${fixed(dot(continuous.mean, query.x), 5)}; clean predictive SD = ${compact(Math.sqrt(predictiveVariance))}, noisy predictive SD = ${compact(Math.sqrt(predictiveVariance + sigma * sigma))}.`;
    const scalars = [...rows.flatMap((r) => [...r.x, r.y]), ...query.x, queryY];
    const clipped = scalars.filter((z) => Math.abs(z) > canonical.range).length;
    byId("nr-g-detail").textContent = `Population SNR = ${compact(1 / (sigma * sigma))}; this task's SNR = ${compact(norm * norm / (sigma * sigma))}. ${clipped}/${scalars.length} displayed raw scalars exceed the codec range ±${canonical.range}. ${showDecoded ? `Decoded-input ridge query prediction = ${fixed(dot(decoded.mean, query.x.map(decode)), 5)}. The posterior panel expands to include it; no calibrated uncertainty for quantized data is implied.` : "Enable decoded centers to reveal input rounding and clipping; most movements are small on the input-plane scale."}`;
    byId("nr-geometry").dataset.posteriorTrace = a + c;
    byId("nr-geometry").dataset.observations = n;
  }

  const promptTask = taskDraw(93271, 64);
  const example = {
    w: promptTask.w,
    sigma: canonical.sigma,
    rows: promptTask.rows.map((r) => ({ x: r.x, signal: dot(promptTask.w, r.x), noise: canonical.sigma * r.z, y: dot(promptTask.w, r.x) + canonical.sigma * r.z })),
    query: { x: promptTask.query.x, signal: dot(promptTask.w, promptTask.query.x), noise: canonical.sigma * promptTask.query.z, y: dot(promptTask.w, promptTask.query.x) + canonical.sigma * promptTask.query.z },
  };
  function updatePrompts() {
    const modern = byId("nr-prompt-format").value === "new";
    const formatX = (x) => modern ? `${digits(x[0]).join(" ")} [SEP] ${digits(x[1]).join(" ")}` : x.flatMap(digits).join(" ");
    const lines = ["[BOS]", ...example.rows.map((r) => `[X] ${formatX(r.x)} [Y] ${digits(r.y).join(" ")}${modern ? " [EOO]" : ""}`)];
    lines.push(`${modern ? "[QUERY] " : ""}[X] ${formatX(example.query.x)} [Y]`);
    const prompt = lines.join("\n"), answer = digits(example.query.y).join(" ") + (modern ? " [EOS]" : "");
    const promptLength = prompt.split(/\s+/).length, answerLength = answer.split(/\s+/).length;
    byId("nr-prompt-text").textContent = prompt;
    byId("nr-answer-text").textContent = answer;
    byId("nr-prompt-summary").textContent = `64 observations · d = 2 · σ = ${example.sigma} · ${promptLength} prompt tokens + ${answerLength} supervised tokens = ${promptLength + answerLength} total. Same continuous example in both formats.`;
    byId("nr-latent-description").textContent = `Hidden w = (${fixed(example.w[0], 8)}, ${fixed(example.w[1], 8)}). Values below are rounded for display only; tokens are computed from the full-precision draw. The hidden w, clean signals and noises are shown here for explanation, never supplied to the model.`;
    byId("nr-example-rows").innerHTML = [...example.rows, example.query].map((r, i) => `<tr><td>${i < 64 ? i + 1 : "Query"}</td>${[...r.x, r.signal, r.noise, r.y].map((v) => `<td>${fixed(v, 7)}</td>`).join("")}</tr>`).join("");
    byId("nr-prompts").dataset.promptLength = promptLength;
  }

  for (const id of ["nr-pop-mode", "nr-dimension"]) byId(id).addEventListener("change", updatePopulation);
  for (const id of ["nr-q-sigma", "nr-phase", "nr-range", "nr-levels"]) byId(id).addEventListener("input", updateQuantization);
  for (const id of ["nr-n", "nr-g-sigma", "nr-angle", "nr-norm", "nr-show-quantized"]) byId(id).addEventListener("input", updateGeometry);
  byId("nr-q-default").addEventListener("click", () => {
    byId("nr-q-sigma").value = Math.log10(canonical.sigma); byId("nr-phase").value = 0; byId("nr-range").value = canonical.range; byId("nr-levels").value = canonical.levels; updateQuantization();
  });
  byId("nr-q-boundary").addEventListener("click", () => { byId("nr-phase").value = 0.5; updateQuantization(); });
  byId("nr-q-hidden").addEventListener("click", () => { byId("nr-q-sigma").value = -3; updateQuantization(); });
  byId("nr-g-default").addEventListener("click", () => {
    geometrySeed = 61739; geometryTask = taskDraw(geometrySeed, 128); setRuleControls(geometryTask.w);
    byId("nr-n").value = 64; byId("nr-g-sigma").value = Math.log10(canonical.sigma); byId("nr-show-quantized").checked = false; updateGeometry();
  });
  byId("nr-new-task").addEventListener("click", () => { geometryTask = taskDraw(++geometrySeed, 128); setRuleControls(geometryTask.w); updateGeometry(); });
  byId("nr-prompt-format").addEventListener("change", updatePrompts);
  let resizeFrame;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => { updatePopulation(); updateQuantization(); });
  });

  const data = document.createElement("script");
  data.id = "nr-example-data"; data.type = "application/json"; data.textContent = JSON.stringify(example);
  byId("nr-prompts").appendChild(data);
  setRuleControls(geometryTask.w);
  updatePopulation(); updateQuantization(); updateGeometry(); updatePrompts();
})();
