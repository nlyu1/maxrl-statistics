import argparse
import hashlib
import json
import math
import statistics
import subprocess
import textwrap
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from plotly.offline import get_plotlyjs
from pydantic.dataclasses import dataclass

from src.experiments.utkface.final_report import METRICS


@dataclass(kw_only=True, frozen=True)
class Method:
    key: str
    label: str
    method: str
    head: str
    width: float
    color: str
    dash: str


METHODS = {
    "scalar_mae": Method(
        key="scalar_mae",
        label="Scalar MAE",
        method="mae",
        head="scalar",
        width=2,
        color="#7b3294",
        dash="solid",
    ),
    "scalar_mse": Method(
        key="scalar_mse",
        label="Scalar MSE",
        method="mse",
        head="scalar",
        width=2,
        color="#b05fab",
        dash="solid",
    ),
    "mae": Method(
        key="mae",
        label="Categorical MAE",
        method="mae",
        head="categorical",
        width=2,
        color="#ca6e26",
        dash="solid",
    ),
    "mse": Method(
        key="mse",
        label="Categorical MSE",
        method="mse",
        head="categorical",
        width=2,
        color="#c83b45",
        dash="solid",
    ),
    "tailrl_population": Method(
        key="tailrl_population",
        label="TailRL (population)",
        method="tailrl_population",
        head="categorical",
        width=2,
        color="#0077a1",
        dash="solid",
    ),
    "tailrl": Method(
        key="tailrl",
        label="TailRL (64 samples)",
        method="tailrl",
        head="categorical",
        width=2,
        color="#009a85",
        dash="solid",
    ),
    "ce": Method(
        key="ce",
        label="CE",
        method="ce",
        head="categorical",
        width=2,
        color="#464646",
        dash="solid",
    ),
}

METRIC_LABELS = {
    "mae_mean": "Validation mean-age MAE (years)",
    "rmse_mean": "Validation mean-age RMSE (years)",
    "median_spearman": "Validation median Spearman (higher is better)",
    "label_nll": "Validation label NLL (nats)",
}


def statistical(*, kind: str, width: int, population: bool) -> Method:
    suffix = "_population" if population else ""
    method = f"stat_maxrl_{kind}{suffix}"
    estimator = "population" if population else "64 samples"
    return Method(
        key=f"{method}_s{width}",
        label=f"Stat {kind.upper()}, w={width} ({estimator})",
        method=method,
        head="categorical",
        width=width,
        color="#4e8c29" if kind == "mae" else "#939925",
        dash="solid" if width == 2 else "dash",
    )


def matches(*, run: dict, method: Method) -> bool:
    actual = run["method"]
    if method.method == "ce" and run["size"] == 18462:
        actual = "ce" if actual == "binary_maxrl_population" else actual
    return (
        actual == method.method
        and run["head"] == method.head
        and (not actual.startswith("stat_maxrl") or run["width"] == method.width)
    )


def select(
    *,
    runs: list[dict],
    method: Method,
    stages: set[str],
    size: int,
    epochs: int,
    mode: str,
    std: int,
) -> list[dict]:
    chosen = sorted(
        [
            r
            for r in runs
            if r["stage"] in stages
            and r["size"] == size
            and r["epochs"] == epochs
            and r["noise"] == mode
            and r["std"] == std
            and matches(run=r, method=method)
        ],
        key=lambda r: r["seed"],
    )
    expected = [42, 43, 44] if size == 18462 and method.head != "digits" else [42]
    if [r["seed"] for r in chosen] != expected:
        raise ValueError((
            method.key,
            size,
            epochs,
            mode,
            std,
            expected,
            [r["id"] for r in chosen],
        ))
    if len({r["trainingFingerprint"] for r in chosen}) != 1:
        raise ValueError("Mixed training subsets")
    return chosen


def best_family(*, runs: list[dict], family: str, size: int) -> tuple[Method, list]:
    groups = defaultdict(list)
    stages = {"main"} if size == 18462 else {"subset", "subset_replication"}
    for run in runs:
        if (
            run["stage"] in stages
            and run["size"] == size
            and run["method"].startswith(family + "_")
        ):
            groups[run["method"]].append(run)
    ranking = sorted(
        [
            {
                "method": key,
                "mean_best_validation_mae": statistics.mean(
                    r["best"]["mae_mean"] for r in values
                ),
                "seeds": sorted(r["seed"] for r in values),
            }
            for key, values in groups.items()
        ],
        key=lambda x: (x["mean_best_validation_mae"], x["method"]),
    )
    if not ranking or any(r["seeds"] != [42, 43, 44] for r in ranking):
        raise ValueError((family, size, ranking))
    method = ranking[0]["method"]
    kind = "MAE" if "_mae" in method else "MSE"
    estimator = "population" if method.endswith("_population") else "64 samples"
    return Method(
        key=method,
        label=f"{family.upper()} {kind} ({estimator})",
        method=method,
        head="categorical",
        width=2,
        color="#2462ad" if family == "grpo" else "#c43e83",
        dash="solid",
    ), ranking


def prepare(*, payload: dict, source_hash: str) -> dict:
    runs = [r for r in payload["runs"] if r["status"] == "completed"]
    if (
        len({r["cohort"] for r in runs}) != 1
        or len({r["fingerprint"] for r in runs}) != 1
    ):
        raise ValueError("Bundle must use one training implementation and split")
    for run in runs:
        if len(run["curves"]) != run["epochs"]:
            raise ValueError((run["id"], "incomplete history"))
    used = {}
    decisions = {}
    grpo, decisions["full_grpo"] = best_family(runs=runs, family="grpo", size=18462)
    rloo, decisions["full_rloo"] = best_family(runs=runs, family="rloo", size=18462)
    noise_rl, decisions["noise_rl"] = best_family(runs=runs, family="grpo", size=4096)
    decisions["noise_rl_coverage"] = "RLOO noise arms were not run; use available GRPO."
    stat_mae = statistical(kind="mae", width=2, population=True)
    stat_mse = statistical(kind="mse", width=2, population=True)
    figures = []

    def group(
        *,
        method: Method,
        stages: set[str],
        size: int,
        epochs: int,
        mode: str = "none",
        std: int = 0,
    ) -> dict:
        chosen = select(
            runs=runs,
            method=method,
            stages=stages,
            size=size,
            epochs=epochs,
            mode=mode,
            std=std,
        )
        for r in chosen:
            used[r["id"]] = {
                key: r[key]
                for key in (
                    "id",
                    "stage",
                    "method",
                    "head",
                    "width",
                    "size",
                    "epochs",
                    "seed",
                    "lr",
                    "snr",
                    "noiseRmse",
                    "noise",
                    "std",
                    "cohort",
                    "fingerprint",
                    "trainingFingerprint",
                    "curves",
                    "initial",
                )
            }
        return {
            "key": method.key,
            "label": method.label,
            "legendKey": method.key,
            "legendLabel": method.label,
            "color": method.color,
            "dash": method.dash,
            "ids": [r["id"] for r in chosen],
        }

    def clean(*, method: Method, size: int, epochs: int, digits: bool = False) -> dict:
        stages = (
            {"digits"}
            if digits
            else {"main", "scalar"}
            if size == 18462
            else {"matched_steps"}
        )
        return group(method=method, stages=stages, size=size, epochs=epochs)

    def add(
        *, slug: str, title: str, note: str, rows: list, four_grid: bool = False
    ) -> None:
        panels = []
        for row in rows:
            for metric in METRICS:
                panels.append({
                    "title": row["title"],
                    "metric": metric,
                    "groups": row["groups"],
                    "horizon": row["horizon"],
                })
        figures.append({
            "id": slug,
            "title": title,
            "note": note,
            "rows": 2 if four_grid else len(rows),
            "cols": 2 if four_grid else 4,
            "panels": panels,
        })

    main_methods = [
        METHODS["scalar_mae"],
        METHODS["tailrl_population"],
        METHODS["tailrl"],
        METHODS["ce"],
        stat_mae,
        stat_mse,
        grpo,
        rloo,
    ]
    add(
        slug="01-main-full",
        title="Full data: eight-method comparison",
        note="18,462 images × 50 epochs; three matched seeds. GRPO/RLOO configurations were selected by mean best validation MAE, without test outcomes.",
        rows=[
            {
                "title": "Full data · n=3",
                "horizon": 50,
                "groups": [
                    clean(method=m, size=18462, epochs=50) for m in main_methods
                ],
            }
        ],
        four_grid=True,
    )
    add(
        slug="02-sl-heads",
        title="Full data: scalar and categorical SL",
        note="Three matched seeds; all four settings use peak LR 0.001. Scalar label NLL is undefined and is omitted from that panel.",
        rows=[
            {
                "title": "Full data · n=3",
                "horizon": 50,
                "groups": [
                    clean(method=METHODS[k], size=18462, epochs=50)
                    for k in ("scalar_mae", "scalar_mse", "mae", "mse")
                ],
            }
        ],
        four_grid=True,
    )
    reduced_grpo = Method(
        key="grpo_mse_population",
        label="GRPO MSE (population)",
        method="grpo_mse_population",
        head="categorical",
        width=2,
        color="#c43e83",
        dash="dash",
    )
    reduced_methods = [
        METHODS["mae"],
        METHODS["mse"],
        METHODS["tailrl_population"],
        METHODS["ce"],
        stat_mae,
        stat_mse,
        noise_rl,
        reduced_grpo,
    ]
    add(
        slug="03-main-1024",
        title="1024 images × 200 epochs: eight tested methods",
        note="Seed 42. Adapted to the eight completed categorical settings: categorical MAE/MSE and both population GRPO rewards replace the unrun scalar, sampled-TailRL and RLOO settings.",
        rows=[
            {
                "title": "1024 × 200 · n=1",
                "horizon": 200,
                "groups": [
                    clean(method=m, size=1024, epochs=200) for m in reduced_methods
                ],
            }
        ],
        four_grid=True,
    )
    width_rows = []
    for population in (True, False):
        methods = [
            statistical(kind=kind, width=width, population=population)
            for kind in ("mae", "mse")
            for width in (2, 5)
        ]
        groups = []
        for method in [*methods, METHODS["ce"]]:
            selected = clean(method=method, size=18462, epochs=50)
            selected["legendKey"] = method.key.replace("_population", "")
            selected["legendLabel"] = method.label.split(" (")[0]
            groups.append(selected)
        width_rows.append({
            "title": "Population" if population else "64 samples",
            "horizon": 50,
            "groups": groups,
        })
    add(
        slug="04-stat-widths",
        title="Statistical MaxRL: widths and estimators",
        note="Full data, three seeds per setting. Five shared legend entries control both estimator rows. Solid: width 2; dashed: width 5. The same exact CE reference appears in both rows. Rates are independently validation-selected; this is not a matched-LR intervention.",
        rows=width_rows,
    )
    focus_rows = [
        {
            "title": "Full data × 50 · n=3",
            "horizon": 50,
            "groups": [
                clean(method=METHODS[k], size=18462, epochs=50)
                for k in ("tailrl", "tailrl_population", "scalar_mae", "ce")
            ],
        },
    ]
    add(
        slug="05-tail-focus",
        title="TailRL and supervised baselines",
        note="Full data × 50 epochs, three matched seeds: sampled TailRL, population TailRL, scalar MAE and CE. TailRL peak LR: population 0.0001; 64 samples 0.001. Rates are independently validation-selected.",
        rows=focus_rows,
        four_grid=True,
    )
    noise_rows = []
    for std in (1, 3, 5, 10):
        groups = [
            group(
                method=m, stages={"noise"}, size=4096, epochs=50, mode="fixed", std=std
            )
            for m in (
                METHODS["scalar_mae"],
                METHODS["tailrl_population"],
                METHODS["ce"],
                stat_mae,
                noise_rl,
            )
        ]
        record = used[groups[0]["ids"][0]]
        ratio = 10 ** (-record["snr"] / 20)
        noise_rows.append({
            "title": f"SD {std} · η={ratio:.3f} · SNR {record['snr']:.1f} dB",
            "horizon": 50,
            "groups": groups,
        })
    add(
        slug="06-fixed-noise",
        title="Fixed training-label noise: four levels",
        note="4096 images × 50 epochs; optimization seed 42, noise seed 2026; held-out labels are clean. η = noise RMS / clean-age SD (about 0.05 at SD 1); conventional power SNR = 1/η². SD 3 replaces the unrun SD 2 by request. GRPO MAE is the best available clean-validation RL configuration; RLOO noise arms were not run.",
        rows=noise_rows,
    )
    digit_methods = [
        Method(
            key=m.key,
            label=m.label,
            method=m.method,
            head="digits",
            width=m.width,
            color=m.color,
            dash=m.dash,
        )
        for m in reduced_methods
    ]
    digit_rows = [
        {
            "title": "Full data × 50 · n=1",
            "horizon": 50,
            "groups": [
                clean(method=m, size=18462, epochs=50, digits=True)
                for m in digit_methods
            ],
        }
    ]
    add(
        slug="07-digits",
        title="Conditional two-digit policies: eight objectives",
        note="Seed 42. Eight conditional 10×10 objectives on full data × 50 epochs. No 1024 × 200 digit runs are available. All digit rates were selected on 4096 images; full-data digit runs reuse them. These panels contain digit policies only.",
        rows=digit_rows,
        four_grid=True,
    )
    return {
        "generated": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "snapshot": payload["generated"],
        "sourceSha256": source_hash,
        "sourceCohort": runs[0]["cohort"],
        "dataFingerprint": runs[0]["fingerprint"],
        "curveColumns": payload["curveColumns"],
        "metrics": METRIC_LABELS,
        "selection": decisions,
        "figures": figures,
        "runs": used,
        "adaptations": [
            "Use SD 3 instead of unavailable SD 2.",
            "Use the eight tested categorical 1024×200 methods.",
            "Show full data only in the TailRL-versus-supervised figure.",
            "Show full data only for digit policies; no 1024×200 digit runs exist.",
        ],
    }


def series(*, runs: list[dict], metric: str, window: int) -> dict:
    column = METRICS.index(metric) + 1
    epochs = runs[0]["epochs"]
    values = [[point[column] for point in r["curves"]] for r in runs]
    if all(v is None for row in values for v in row):
        return {"x": [], "mean": [], "sd": []}
    if any(v is None or not math.isfinite(v) for row in values for v in row):
        raise ValueError("Partially undefined or nonfinite series")
    kernel = np.ones(window) / window
    smoothed = np.array([np.convolve(row, kernel, mode="valid") for row in values])
    initial = np.array([r["initial"]["validation"][metric] for r in runs])
    if not np.isfinite(initial).all():
        raise ValueError("Nonfinite epoch-zero metric")
    plotted = np.column_stack((initial, smoothed))
    return {
        "x": [0, *(np.arange(epochs - window + 1) + (window + 1) / 2).tolist()],
        "mean": plotted.mean(0).tolist(),
        "sd": plotted.std(0, ddof=1).tolist() if len(runs) > 1 else [],
    }


def panel_range(*, curves: list[dict], metric: str, horizon: int) -> list[float]:
    values = [
        y for curve in curves for x, y in zip(curve["x"], curve["mean"], strict=True)
    ]
    low, high = min(values), max(values)
    pad = max((high - low) * 0.1, 0.003 if metric == "median_spearman" else 0.1)
    return [
        max(-1 if metric == "median_spearman" else 0, low - pad),
        min(1, high + pad) if metric == "median_spearman" else high + pad,
    ]


def render_static(*, bundle: dict, output_dir: Path) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 170,
    })
    with PdfPages(output_dir / "all-seven.pdf") as book:
        for spec in bundle["figures"]:
            width = 14 if spec["cols"] == 2 else 20
            foot = (
                spec["note"]
                + " Moving means: 5 epochs (10 for 200-epoch runs); bands: between-seed sample SD where n=3. Use the interactive viewer to zoom into later training."
            )
            caption = "\n".join(textwrap.wrap(foot, width=145 if width == 14 else 210))
            caption_height = (len(caption.splitlines()) * 12 + 8) / 72
            legend_columns = 3 if width == 14 else 4
            keys = {g["legendKey"] for p in spec["panels"] for g in p["groups"]}
            legend_height = (math.ceil(len(keys) / legend_columns) * 16 + 12) / 72
            footer_height = caption_height + legend_height + 0.35
            height = 3.6 * spec["rows"] + footer_height + 0.7
            fig, axes = plt.subplots(
                spec["rows"],
                spec["cols"],
                figsize=(width, height),
            )
            legend = {}
            for ax, panel in zip(axes.flat, spec["panels"], strict=True):
                metric = panel["metric"]
                window = 10 if panel["horizon"] == 200 else 5
                curves = []
                for g in panel["groups"]:
                    chosen = [bundle["runs"][key] for key in g["ids"]]
                    curve = series(runs=chosen, metric=metric, window=window)
                    if not curve["x"]:
                        continue
                    curves.append(curve)
                    mean = np.array(curve["mean"])
                    line = ax.plot(
                        curve["x"],
                        mean,
                        color=g["color"],
                        ls="--" if g["dash"] == "dash" else "-",
                        lw=1.9,
                        label=g["label"],
                    )[0]
                    legend.setdefault(g["legendKey"], (line, g["legendLabel"]))
                    if curve["sd"]:
                        sd = np.array(curve["sd"])
                        ax.fill_between(
                            curve["x"],
                            mean - sd,
                            mean + sd,
                            color=g["color"],
                            alpha=0.09,
                        )
                ax.set_title(
                    (panel["title"] + "\n" if spec["cols"] == 4 else "")
                    + METRIC_LABELS[metric],
                    fontsize=10,
                )
                ax.set_xlabel("Epoch")
                ax.set_xlim(0, panel["horizon"])
                ax.set_ylim(
                    panel_range(curves=curves, metric=metric, horizon=panel["horizon"])
                )
                ax.grid(alpha=0.16)
                panel["defaultX"] = list(ax.get_xlim())
                panel["defaultY"] = list(ax.get_ylim())
                if metric == "label_nll" and any(
                    bundle["runs"][g["ids"][0]]["head"] == "scalar"
                    for g in panel["groups"]
                ):
                    ax.text(
                        0.02,
                        0.98,
                        "Scalar NLL undefined",
                        transform=ax.transAxes,
                        va="top",
                        color="#666666",
                        fontsize=9,
                    )
            fig.suptitle(spec["title"], fontsize=15)
            handles, labels = zip(*legend.values(), strict=True)
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, (caption_height + 0.22) / height),
                ncol=legend_columns,
                fontsize=9,
            )
            fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=9)
            fig.tight_layout(
                rect=(0, (footer_height + 0.1) / height, 1, 0.95),
                h_pad=2.0,
                w_pad=2.0,
            )
            fig.savefig(output_dir / f"{spec['id']}.png")
            fig.savefig(output_dir / f"{spec['id']}.pdf")
            book.savefig(fig)
            plt.close(fig)


def build(*, data: Path, output_dir: Path, epoch_zero_path: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    contents = data.read_bytes()
    source_hash = hashlib.sha256(contents).hexdigest()
    payload = json.loads(contents)
    if epoch_zero_path:
        initial = json.loads(Path(epoch_zero_path).read_text())
        if initial["sourceDataSha256"] != source_hash:
            raise ValueError("Epoch-zero metrics belong to another report snapshot")
        for run in payload["runs"]:
            if run["id"] in initial["runs"]:
                record = initial["records"][initial["runs"][run["id"]]]
                if (
                    record["epoch"] != 0
                    or record["head"] != run["head"]
                    or record["seed"] != run["seed"]
                    or record["data_fingerprint"] != run["fingerprint"]
                ):
                    raise ValueError(f"Epoch-zero provenance mismatch: {run['id']}")
                run["initial"] = record
        write_initial = json.dumps(initial, indent=2, allow_nan=False) + "\n"
        (output_dir / "epoch-zero.json").write_text(write_initial)
    bundle = prepare(payload=payload, source_hash=source_hash)
    render_static(bundle=bundle, output_dir=output_dir)
    serialized = json.dumps(bundle, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    (output_dir / "bundle-data.json").write_text(
        json.dumps(bundle, indent=2, allow_nan=False) + "\n"
    )
    template = Path(__file__).with_suffix(".html").read_text()
    html = template.replace("__PLOTLY_RUNTIME__", get_plotlyjs()).replace(
        "__BUNDLE_DATA__", serialized
    )
    (output_dir / "viewer.html").write_text(html)
    (output_dir / "README.txt").write_text(
        "UTKFace seven-figure bundle\n\nOpen viewer.html in a browser. It is self-contained and needs no network connection. "
        "Use the seven page links, Plotly legend clicks/double-clicks and native zoom/pan tools. "
        "The full-data curves aggregate seeds 42/43/44. Other panels state their actual coverage. "
        "No new training or test-driven selection was performed.\n\n"
        "Default moving means are five epochs, or ten for 200-epoch runs. Choose Raw to inspect unsmoothed curves. "
        "Bands are sample SD across optimization seeds, not confidence intervals. "
        "Epoch zero is unsmoothed and connected directly to the training curves. "
        "Use native zoom/pan to inspect later training. Scalar label NLL is undefined. "
        "epoch-zero.json retains reconstruction provenance; historical training and checkpoint selections are unchanged.\n\n"
        "Approved coverage adaptations: SD 1/3/5/10; eight available categorical 1024x200 settings; "
        "TailRL-versus-supervised and digit figures show full data only. No 1024x200 digit runs exist. "
        "The Statistical MaxRL figure uses five shared legend entries across its two estimator rows. "
        "Eta is noise RMS / signal SD; conventional SNR = 1/eta^2.\n\n"
        "Each figure is supplied as PNG and vector PDF. all-seven.pdf contains all seven pages. "
        "bundle-data.json records source IDs, LR values, seeds, raw curves and validation-only selection decisions.\n"
    )
    manifest = {
        "generated": bundle["generated"],
        "dataSnapshot": bundle["snapshot"],
        "sourceSha256": bundle["sourceSha256"],
        "sourceCohort": bundle["sourceCohort"],
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "figureCount": len(bundle["figures"]),
        "runCount": len(bundle["runs"]),
        "newTrainingRuns": 0,
        "epochZeroRuns": len(bundle["runs"]),
        "adaptations": bundle["adaptations"],
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output_dir.iterdir())
            if p.is_file()
            and p.name != "manifest.json"
            and p.suffix in {".png", ".pdf", ".html", ".json", ".txt"}
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(
        output_dir / "utkface-seven-figures.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.suffix != ".zip":
                archive.write(path, arcname=path.name)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch-zero", default="")
    args = parser.parse_args()
    build(data=args.data, output_dir=args.output_dir, epoch_zero_path=args.epoch_zero)


if __name__ == "__main__":
    main()
