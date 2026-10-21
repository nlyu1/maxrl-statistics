(() => {
  const bundle = window.UTKFACE_DATA;
  const metrics = {
    mae_mean: "Mean-age MAE (years) ↓",
    rmse_mean: "Mean-age RMSE (years) ↓",
    median_spearman: "Median Spearman ↑",
    label_nll: "Label NLL (nats) ↓",
    mae_mode: "Mode MAE (years) ↓",
  };
  const colors = (hex, opacity) => {
    const rgb = [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16));
    return `rgba(${rgb.join(",")},${opacity})`;
  };
  const axisName = (kind, index) => `${kind}axis${index ? index + 1 : ""}`;
  const axisRef = (kind, index) => `${kind}${index ? index + 1 : ""}`;
  const label = (group) => group.label
    .replace(", w=2", "")
    .replace(" (population)", " · exact")
    .replace(" (64 samples)", " · 64");

  function curve(group, metric, windowSize, seed) {
    const runs = group.ids.map((id) => bundle.runs[id])
      .filter((run) => seed === "all" || run.seed === Number(seed));
    if (!runs.length) throw new Error(`No runs for ${group.key}, seed ${seed}`);
    const column = bundle.curveColumns.indexOf(metric);
    if (column < 0) throw new Error(`Unknown metric: ${metric}`);
    const values = runs.map((run) => run.curves.map((point) => point[column]));
    if (values.every((row) => row.every((value) => value === null))) return null;
    if (values.some((row) => row.some((value) => !Number.isFinite(value)))) {
      throw new Error(`Undefined values in ${group.key}: ${metric}`);
    }
    const smoothed = values.map((row) => row.slice(windowSize - 1).map((_, index) =>
      row.slice(index, index + windowSize).reduce((sum, value) => sum + value, 0) / windowSize));
    const x = smoothed[0].map((_, index) =>
      (runs[0].curves[index][0] + runs[0].curves[index + windowSize - 1][0]) / 2);
    const mean = x.map((_, index) =>
      smoothed.reduce((sum, row) => sum + row[index], 0) / runs.length);
    const sd = runs.length > 1 ? x.map((_, index) => Math.sqrt(
      smoothed.reduce((sum, row) => sum + (row[index] - mean[index]) ** 2, 0) / (runs.length - 1))) : [];
    return { x, mean, sd, runs };
  }

  function yRange(curves, metric, horizon, full) {
    const values = curves.flatMap((series) => series.x.flatMap((epoch, index) => {
      if (!full && epoch < horizon * 0.4) return [];
      const sd = series.sd[index] || 0;
      return [series.mean[index] - sd, series.mean[index] + sd];
    }));
    const low = Math.min(...values);
    const high = Math.max(...values);
    const pad = Math.max((high - low) * 0.1, metric === "median_spearman" ? 0.003 : 0.1);
    return [Math.max(metric === "median_spearman" ? -1 : 0, low - pad),
      metric === "median_spearman" ? Math.min(1, high + pad) : high + pad];
  }

  function figure(spec, view, state, width) {
    const sourcePanels = view === "summary"
      ? spec.panels.filter((panel) => !state.noise || panel.title === state.noise)
      : spec.panels.filter((panel) => panel.metric === "mae_mean")
        .map((panel) => ({ ...panel, metric: "mae_mode" }));
    const cols = sourcePanels.length === 1 || width < 460 ? 1 : 2;
    const rows = Math.ceil(sourcePanels.length / cols);
    const groups = new Set(sourcePanels.flatMap((panel) => panel.groups.map((group) => group.key)));
    const legendCols = width < 480 ? 1 : 2;
    const legendHeight = Math.ceil(groups.size / legendCols) * 23;
    const plotHeight = rows * (cols === 1 ? 230 : 215);
    const gapX = 0.14;
    const gapY = rows > 1 ? 0.21 / (rows - 1) : 0;
    const cellWidth = (1 - gapX * (cols - 1)) / cols;
    const cellHeight = (1 - gapY * (rows - 1)) / rows;
    const data = [];
    const annotations = [];
    const seen = new Set();
    const layout = {
      height: plotHeight + legendHeight + 145,
      margin: { l: 52, r: 22, t: 45, b: legendHeight + 95 },
      font: { family: "Segoe UI, Arial, sans-serif", size: 11, color: "#334652" },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      hovermode: "closest",
      dragmode: "zoom",
      uirevision: `${spec.id}:${view}:${state.noise}:${state.seed}:${state.revision}`,
      legend: {
        orientation: "h", x: 0, y: -60 / plotHeight, xanchor: "left", yanchor: "top",
        entrywidth: Math.max(100, (width - 90) / legendCols - 45),
        entrywidthmode: "pixels", tracegroupgap: 0, font: { size: 10 },
        groupclick: "togglegroup", itemclick: "toggle", itemdoubleclick: "toggleothers",
      },
      annotations,
    };
    sourcePanels.forEach((panel, index) => {
      const row = Math.floor(index / cols);
      const col = index % cols;
      const xd = [col * (cellWidth + gapX), col * (cellWidth + gapX) + cellWidth];
      const yd = [1 - row * (cellHeight + gapY) - cellHeight, 1 - row * (cellHeight + gapY)];
      const windowSize = state.smoothing === "auto" ? (panel.horizon === 200 ? 10 : 5) : 1;
      const ranges = [];
      const counts = new Set();
      for (const group of panel.groups) {
        const series = curve(group, panel.metric, windowSize, state.seed);
        if (!series) continue;
        const raw = windowSize === 1 ? series : curve(group, panel.metric, 1, state.seed);
        ranges.push(series);
        if (state.full) ranges.push(raw);
        counts.add(series.runs.length);
        const common = {
          type: "scatter", mode: "lines", name: label(group), legendgroup: group.key,
          xaxis: axisRef("x", index), yaxis: axisRef("y", index),
        };
        if (series.sd.length) {
          data.push({ ...common, x: series.x, y: series.mean.map((value, i) => value - series.sd[i]),
            line: { width: 0, color: group.color }, showlegend: false, hoverinfo: "skip" });
          data.push({ ...common, x: series.x, y: series.mean.map((value, i) => value + series.sd[i]),
            line: { width: 0, color: group.color }, fill: "tonexty", fillcolor: colors(group.color, 0.1),
            showlegend: false, hoverinfo: "skip" });
        }
        if (windowSize > 1) {
          data.push({ ...common, x: raw.x, y: raw.mean,
            line: { color: colors(group.color, 0.2), width: 0.8, dash: group.dash },
            showlegend: false, hoverinfo: "skip" });
        }
        const seeds = series.runs.map((run) => run.seed).join(", ");
        const rates = [...new Set(series.runs.map((run) => run.lr))].join(", ");
        data.push({ ...common, x: series.x, y: series.mean,
          line: { color: group.color, width: 2.1, dash: group.dash },
          showlegend: !seen.has(group.key),
          meta: { kind: "mean", metric: panel.metric, ids: series.runs.map((run) => run.id) },
          customdata: series.x.map((_, i) => series.sd[i] ?? null),
          hovertemplate: `${group.label}<br>Epoch %{x:.1f}<br>${metrics[panel.metric]}: %{y:.4f}`
            + (series.sd.length ? "<br>Seed SD: %{customdata:.4f}" : "")
            + `<br>n=${series.runs.length} · seeds ${seeds}<br>Peak LR ${rates}<extra></extra>`,
        });
        seen.add(group.key);
      }
      const commonAxis = { showline: true, linecolor: "#cad2d8", gridcolor: "#edf0f2",
        zeroline: false, automargin: true, tickfont: { size: 10 }, nticks: 5 };
      layout[axisName("x", index)] = {
        ...commonAxis, domain: xd, anchor: axisRef("y", index),
        title: { text: "Epoch", font: { size: 10 }, standoff: 5 },
        range: state.full ? [1, panel.horizon] : panel.defaultX,
      };
      layout[axisName("y", index)] = {
        ...commonAxis, domain: yd, anchor: axisRef("x", index),
        range: yRange(ranges, panel.metric, panel.horizon, state.full),
      };
      const condition = panel.title.startsWith("SD ") ? panel.title.split(" · SNR")[0] : "";
      annotations.push({ text: (condition ? `${condition}<br>` : "") + metrics[panel.metric],
        x: (xd[0] + xd[1]) / 2, y: yd[1] + 0.025, xref: "paper", yref: "paper",
        showarrow: false, xanchor: "center", yanchor: "bottom", font: { size: 11 } });
      if (panel.metric === "label_nll" && panel.groups.some((group) =>
        bundle.runs[group.ids[0]].head === "scalar")) {
        annotations.push({ text: "Scalar NLL undefined", x: xd[0], y: yd[1] - 0.025,
          xref: "paper", yref: "paper", showarrow: false, xanchor: "left",
          font: { size: 9, color: "#687985" } });
      }
      if (counts.size !== 1) throw new Error("Mixed seed coverage within a panel");
    });
    return { data, layout };
  }

  function selectControl(toolbar, text, options, value, change) {
    const label = document.createElement("label");
    label.append(document.createTextNode(text));
    const select = document.createElement("select");
    for (const [key, text] of options) select.add(new Option(text, key));
    select.value = value;
    select.addEventListener("change", () => change(select.value));
    label.append(select);
    toolbar.append(label);
  }

  function initialize(block, spec) {
    const view = block.dataset.view;
    const conditions = [...new Set(spec.panels.map((panel) => panel.title))];
    const firstGroup = spec.panels[0].groups[0];
    const seeds = firstGroup.ids.map((id) => bundle.runs[id].seed);
    const state = { smoothing: "auto", seed: "all", full: false, revision: 0,
      noise: view === "summary" && conditions.length > 1 ? conditions[0] : "" };
    const toolbar = document.createElement("div");
    toolbar.className = "utk-controls";
    toolbar.setAttribute("role", "group");
    toolbar.setAttribute("aria-label", `${spec.title}: ${view} plot controls`);
    const graph = document.createElement("div");
    graph.className = "utk-plot";
    graph.setAttribute("role", "img");
    graph.setAttribute("aria-label", `${spec.title}: ${view === "mode" ? "mode MAE" : "four validation metrics"}`);
    const note = document.createElement("p");
    note.className = "utk-plot-note";
    note.textContent = `n=${seeds.length} · seeds ${seeds.join(", ")}. Faint: raw; bold: moving mean. `
      + (seeds.length > 1 ? "Bands: seed SD. " : "")
      + "Click a legend entry to hide; double-click to isolate. Drag to zoom.";
    const error = document.createElement("p");
    error.className = "utk-plot-error";
    error.setAttribute("role", "alert");
    error.hidden = true;
    block.append(toolbar, graph, note, error);
    let frame = 0;
    let previousWidth = 0;
    const render = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(async () => {
        if (!block.open) return;
        try {
          if (!bundle || !window.Plotly) throw new Error("The plot data or bundled Plotly library did not load.");
          const width = graph.getBoundingClientRect().width;
          const plot = figure(spec, view, state, width);
          await Plotly.react(graph, plot.data, plot.layout, {
            responsive: true, displaylogo: false, scrollZoom: false,
            toImageButtonOptions: { format: "png", filename: `${spec.id}-${view}`, scale: 2 },
          });
          error.hidden = true;
          block.dataset.rendered = "true";
          note.textContent = `n=${state.seed === "all" ? seeds.length : 1} · seeds ${state.seed === "all" ? seeds.join(", ") : state.seed}. `
            + (state.smoothing === "auto" ? "Faint: raw; bold: moving mean. " : "Raw epochs. ")
            + (state.seed === "all" && seeds.length > 1 ? "Bands: seed SD. " : "")
            + "Click a legend entry to hide; double-click to isolate. Drag to zoom.";
        } catch (reason) {
          error.textContent = `Unable to draw this plot: ${reason.message}`;
          error.hidden = false;
          console.error(reason);
        }
      });
    };
    selectControl(toolbar, "Curves", [["auto", "Moving mean + raw"], ["raw", "Raw epochs"]], state.smoothing,
      (value) => { state.smoothing = value; render(); });
    if (seeds.length > 1) selectControl(toolbar, "Seeds",
      [["all", "All three"], ...seeds.map((seed) => [String(seed), String(seed)])], state.seed,
      (value) => { state.seed = value; state.revision++; render(); });
    if (state.noise) selectControl(toolbar, "Training noise SD",
      conditions.map((title) => [title, `${title.split(" · ")[0]} years`]), state.noise,
      (value) => { state.noise = value; render(); });
    for (const [text, full] of [["Full range", true], ["Reset axes", false]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = text;
      button.addEventListener("click", () => { state.full = full; state.revision++; render(); });
      toolbar.append(button);
    }
    block.addEventListener("toggle", render);
    const observer = new ResizeObserver(() => {
      const width = graph.getBoundingClientRect().width;
      if (block.open && width > 0 && Math.abs(width - previousWidth) > 1) {
        previousWidth = width;
        render();
      }
    });
    observer.observe(block);
    render();
  }

  for (const pair of document.querySelectorAll(".utk-plots")) {
    const spec = bundle?.figures.find((figure) => figure.id === pair.dataset.figure);
    for (const block of pair.querySelectorAll("details")) {
      block.addEventListener("toggle", () => {
        if (!block.open || block.dataset.initialized) return;
        block.dataset.initialized = "true";
        if (!spec) {
          const error = document.createElement("p");
          error.className = "utk-plot-error";
          error.textContent = "The plot data did not load. Reload this page to try again.";
          error.setAttribute("role", "alert");
          block.append(error);
          return;
        }
        initialize(block, spec);
      });
    }
  }
})();
