import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from src.experiments.utkface.policy_centering_analysis import CENTERS, COLORS
from src.experiments.utkface.policy_examples import sha256

NOISE_LEVELS = (1, 3, 5, 10)
ORIGINAL_COLOR = "#7451a8"
NOISY_COLOR = "#c16b16"


def load_points(*, root: Path) -> tuple[dict[int, pl.DataFrame], dict]:
    audit = json.loads((root / "inference-audit.json").read_text())
    frame = pl.read_csv(root / "per-image-statistics.csv").filter(
        (pl.col("method") == "tailrl_population") & (pl.col("split") == "train")
    )
    points = {}
    reference = None
    for std in NOISE_LEVELS:
        data = frame.filter(pl.col("std") == std).sort("index")
        labels = data.select("index", "original_label")
        if (
            data.height != audit["train_count"]
            or data["index"].n_unique() != data.height
        ):
            raise ValueError(f"Incomplete or duplicated training cohort at SD {std}")
        if reference is not None and not labels.equals(reference):
            raise ValueError("Noise levels must use the same faces and original labels")
        reference = labels
        for column in ("original_label", "corrupted_training_label", *CENTERS):
            values = data[column].to_numpy()
            if (
                not np.isfinite(values).all()
                or not ((0 <= values) & (values <= 99)).all()
            ):
                raise ValueError(f"Invalid {column} at SD {std}")
        delta = data["corrupted_training_label"] - data["original_label"]
        if not delta.equals(data["label_error_or_test_probe_error"].rename(delta.name)):
            raise ValueError("The plotted noise must be the actual training corruption")
        if not data["median"].equals(data["q50"].rename("median")):
            raise ValueError("Median must agree with the saved policy CDF quantile")
        points[std] = data
    return points, audit


def checkpoint_epoch(*, audit: dict, std: int) -> int:
    return next(
        run["epoch"]
        for run in audit["runs"]
        if run["method"] == "tailrl_population" and run["std"] == std
    )


def residual_scatter(
    *, root: Path, points: dict[int, pl.DataFrame], audit: dict
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(13, 14.5), layout="constrained")
    fig.suptitle(
        "TailRL: individual policy centers versus actual label corruption", fontsize=18
    )
    all_x = np.concatenate([
        (data["corrupted_training_label"] - data["original_label"]).to_numpy()
        for data in points.values()
    ])
    x_limit = 5 * np.ceil((np.abs(all_x).max() + 1) / 5)
    all_y = np.concatenate([
        data[center].to_numpy() - data["original_label"].to_numpy()
        for data in points.values()
        for center in CENTERS
    ])
    y_limit = 5 * np.ceil((np.abs(all_y).max() + 1) / 5)
    guides = np.linspace(-x_limit, x_limit, 200)
    for row, (std, data) in enumerate(points.items()):
        original = data["original_label"].to_numpy()
        delta = data["corrupted_training_label"].to_numpy() - original
        for column, center in enumerate(CENTERS):
            ax = axes[row, column]
            error = data[center].to_numpy() - original
            ax.fill_between(
                guides,
                np.minimum(0, guides),
                np.maximum(0, guides),
                color="#e5e9ee",
                zorder=0,
            )
            ax.axhline(0, color=ORIGINAL_COLOR, lw=1.4, zorder=2)
            ax.plot(guides, guides, color=NOISY_COLOR, lw=1.4, ls="--", zorder=2)
            dots = ax.scatter(
                delta,
                error,
                s=6,
                alpha=0.17,
                linewidths=0,
                color=COLORS["tailrl_population"],
                zorder=3,
            )
            if len(dots.get_offsets()) != data.height:
                raise ValueError("Every face must contribute exactly one scatter point")
            ax.set(xlim=(-x_limit, x_limit), ylim=(-y_limit, y_limit))
            ax.set_title(
                f"SD {std} · {center.capitalize()} · epoch {checkpoint_epoch(audit=audit, std=std)}",
                fontsize=11,
            )
            if column == 0:
                ax.set_ylabel("Policy center − original label (years)")
            if row == len(points) - 1:
                ax.set_xlabel("Corrupted label − original label (years)")
            ax.grid(alpha=0.12)
            ax.set_axisbelow(True)
    handles = [
        Line2D(
            [],
            [],
            color=ORIGINAL_COLOR,
            lw=1.6,
            label="Center equals original label: y = 0",
        ),
        Line2D(
            [],
            [],
            color=NOISY_COLOR,
            lw=1.6,
            ls="--",
            label="Center equals corrupted label: y = x",
        ),
        Line2D(
            [], [], color="#b5bdc7", lw=7, label="Shaded region: center between labels"
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False)
    fig.savefig(root / "raw-residual-scatter.png", dpi=150, facecolor="white")
    plt.close(fig)


def label_scatter(*, root: Path, std: int, data: pl.DataFrame, audit: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.8), layout="constrained")
    fig.suptitle(
        f"TailRL at fixed-noise SD {std}: raw labels versus policy centers\n"
        f"All {data.height:,} training faces · checkpoint epoch {checkpoint_epoch(audit=audit, std=std)}",
        fontsize=17,
    )
    original = data["original_label"].to_numpy()
    noisy = data["corrupted_training_label"].to_numpy()
    delta = noisy - original
    colors = np.where(delta < 0, "#3574ad", np.where(delta > 0, "#ce691d", "#77818e"))
    order = np.random.default_rng(0).permutation(data.height)
    for row, (label, x) in enumerate((
        ("Original label", original),
        ("Corrupted label", noisy),
    )):
        for column, center in enumerate(CENTERS):
            ax = axes[row, column]
            ax.plot([0, 99], [0, 99], color="#5d6570", ls="--", lw=1.1, zorder=1)
            dots = ax.scatter(
                x[order],
                data[center].to_numpy()[order],
                c=colors[order],
                s=7,
                alpha=0.27,
                linewidths=0,
                zorder=2,
            )
            if len(dots.get_offsets()) != data.height:
                raise ValueError("Every face must contribute exactly one scatter point")
            ax.set(
                title=f"Policy {center}",
                xlabel=f"{label} (years)",
                ylabel=f"Policy {center} (years)",
                xlim=(-2, 101),
                ylim=(-2, 101),
                xticks=(0, 20, 40, 60, 80, 99),
                yticks=(0, 20, 40, 60, 80, 99),
                aspect="equal",
            )
            ax.grid(alpha=0.12)
            ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", ls="", color=color, label=label, markersize=5)
        for color, label in (
            ("#3574ad", "Corruption lowered the label"),
            ("#ce691d", "Corruption raised the label"),
            ("#77818e", "Rounded/clipped label unchanged"),
        )
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=9
    )
    fig.savefig(root / f"raw-label-scatter-s{std}.png", dpi=150, facecolor="white")
    plt.close(fig)


def image_html(*, root: Path, filename: str, caption: str) -> str:
    encoded = base64.b64encode((root / filename).read_bytes()).decode()
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def report_html(*, root: Path, points: dict[int, pl.DataFrame], audit: dict) -> None:
    residual = image_html(
        root=root,
        filename="raw-residual-scatter.png",
        caption="All 4,096 training faces in every panel; all noise levels use identical axes. "
        "The vertical coordinate is each policy center minus the original annotation. "
        "The horizontal coordinate is the actual rounded/clipped training corruption. "
        "Purple: original-label center; orange: corrupted-label center; gray band: between. "
        "These lines are hypothesis references. No fitted regression, binning, subsampling, "
        "jitter, or age adjustment is applied. Coincident integer-valued points overlap.",
    )
    tabs = []
    css = []
    for std in NOISE_LEVELS:
        checked = " checked" if std == 5 else ""
        tabs.append(f'<input type="radio" id="sd{std}" name="noise"{checked}>')
        css.append(f"#sd{std}:checked ~ .views #view{std}{{display:block}}")
        css.append(
            f'#sd{std}:checked ~ .choices label[for="sd{std}"]{{background:#146d69;color:white}}'
        )
    choices = "".join(f'<label for="sd{std}">SD {std}</label>' for std in NOISE_LEVELS)
    views = []
    for std in NOISE_LEVELS:
        plot = image_html(
            root=root,
            filename=f"raw-label-scatter-s{std}.png",
            caption="Top row: original annotation versus policy center. Bottom row: actual "
            "corrupted training label versus the same policy center. Dashed lines indicate "
            "equality. Colors show the direction of corruption. Every panel uses all faces "
            "and the full 0–99-year label range.",
        )
        views.append(f'<section class="view" id="view{std}">{plot}</section>')
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TailRL raw per-image scatter plots</title><style>
body{margin:0;background:#f4f6f8;color:#1d2938;font:16px/1.6 system-ui,sans-serif}
main{max-width:1300px;margin:auto;padding:24px}h1{font-size:30px;line-height:1.25}code{overflow-wrap:anywhere}
h2{font-size:22px;margin-top:30px}p{max-width:1100px}figure{background:white;margin:20px 0}
img{display:block;width:100%;height:auto}figcaption{padding:12px 16px;color:#4b5c6f;font-size:14px}
input[type=radio]{position:absolute;opacity:0}.choices{display:flex;gap:8px;flex-wrap:wrap}
.choices label{cursor:pointer;padding:8px 20px;background:#e0e6ec;border-radius:6px}
.view{display:none}input:focus-visible ~ .choices{outline:2px solid #146d69}
__CSS__
@media print{body{background:white}main{padding:0}.choices{display:none}.view{display:block}figure{break-inside:avoid}}
</style></head><body><main>
<h1>TailRL: the individual observations</h1>
<p>Mean, median, and mode from the saved fixed-noise checkpoints. Each point is one
training face; each panel contains all 4,096 faces. The median is the first integer age
whose policy CDF reaches 0.5; the mode is the highest-probability age.</p>
<h2>Actual corruption versus prediction error</h2>
<p>A center at the original label would sit on the horizontal purple line. A center
at the corrupted label would sit on the orange diagonal. A center between the labels
would fall inside the gray band. The broad clouds show why an aggregate association
does not describe every image.</p>
__RESIDUAL__
<h2>Original and corrupted labels versus the policy centers</h2>
<p>Select a noise level to inspect the raw age coordinates. The same prediction is
plotted against the original label above and the corrupted label below.</p>
__TABS__<div class="choices">__CHOICES__</div><div class="views">__VIEWS__</div>
<p>These are unadjusted observations from one optimization seed and one corruption field.
Checkpoints were selected using clean validation mean MAE (epochs 23, 22, 19, and 12).
Original UTKFace annotations are reference ages, not independently verified true ages.
Median and mode are discrete: overlap can hide many coincident points, and cloud
appearance alone cannot establish a Gaussian shape, bimodality, or a causal mechanism.
No held-out synthetic noise probes are included in these plots.</p>
<p>Source training fingerprint: <code>__FINGERPRINT__</code>.</p>
</main></body></html>"""
    for key, value in (
        ("__CSS__", "".join(css)),
        ("__RESIDUAL__", residual),
        ("__TABS__", "".join(tabs)),
        ("__CHOICES__", choices),
        ("__VIEWS__", "".join(views)),
        ("__FINGERPRINT__", html.escape(audit["training_fingerprint"])),
    ):
        document = document.replace(key, value)
    (root / "scatter-report.html").write_text(document)
    validation = {
        "source_csv_sha256": sha256(path=root / "per-image-statistics.csv"),
        "training_fingerprint": audit["training_fingerprint"],
        "method": "tailrl_population",
        "split": "train",
        "points_per_panel": {str(std): data.height for std, data in points.items()},
        "centers": list(CENTERS),
        "same_image_ids_across_noise_levels": True,
        "actual_corrupted_training_labels": True,
        "fitted_regressions": False,
        "jitter": False,
        "subsampling": False,
        "figures": ["raw-residual-scatter.png"]
        + [f"raw-label-scatter-s{std}.png" for std in NOISE_LEVELS],
    }
    (root / "scatter-validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    points, audit = load_points(root=args.root)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    residual_scatter(root=args.root, points=points, audit=audit)
    for std, data in points.items():
        label_scatter(root=args.root, std=std, data=data, audit=audit)
    report_html(root=args.root, points=points, audit=audit)


if __name__ == "__main__":
    main()
