import argparse
import ast
import hashlib
import json
import subprocess
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import torchvision

from src.experiments.utils import set_seeds
from src.experiments.utkface.config import RunConfig
from src.experiments.utkface.data import read_manifest
from src.experiments.utkface.distributed import DistributedContext
from src.experiments.utkface.model import build_model
from src.experiments.utkface.run import initial_validation, make_loader, write_json


def verify_sources(*, revision: str) -> dict:
    verified = {}
    for path in (
        "src/experiments/utils.py",
        "src/experiments/utkface/data.py",
        "src/experiments/utkface/distributed.py",
        "src/experiments/utkface/losses.py",
        "src/experiments/utkface/metrics.py",
        "src/experiments/utkface/model.py",
    ):
        historical = subprocess.check_output(["git", "show", f"{revision}:{path}"])
        current = Path(path).read_bytes()
        if historical != current:
            raise ValueError(f"Reconstruction source differs: {path}")
        verified[path] = hashlib.sha256(current).hexdigest()
    path = "src/experiments/utkface/run.py"
    historical = ast.parse(
        subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
    )
    current = ast.parse(Path(path).read_text())
    for name in ("evaluate", "make_loader", "autocast_context"):
        nodes = [
            next(node for node in tree.body if getattr(node, "name", "") == name)
            for tree in (historical, current)
        ]
        if ast.dump(nodes[0]) != ast.dump(nodes[1]):
            raise ValueError(f"Reconstruction function differs: {name}")
        verified[f"{path}:{name}"] = hashlib.sha256(
            ast.dump(nodes[1]).encode()
        ).hexdigest()
    return verified


def reconstruct(
    *, data: Path, plans: tuple[Path, ...], output: Path, device: str
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    evaluation_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    contents = data.read_bytes()
    payload = json.loads(contents)
    selected = {
        row["id"]: row
        for row in payload["runs"]
        if row["status"] == "completed" and row["stage"] != "smoke"
    }
    results = {}
    for path in plans:
        state = json.loads((path.parent / "state.json").read_text())
        for identifier, record in state["jobs"].items():
            if identifier in selected:
                results[identifier] = json.loads(
                    Path(record["result_path"]).read_text()
                )
    if results.keys() != selected.keys():
        raise ValueError("Missing executed configs for epoch-zero reconstruction")
    versions = {
        (result["environment"]["torch"], result["environment"]["torchvision"])
        for result in results.values()
    }
    if versions != {(torch.__version__, torchvision.__version__)}:
        raise ValueError(
            f"Historical and current framework versions differ: {versions}"
        )
    revisions = sorted({
        result["environment"]["git_commit"] for result in results.values()
    })
    verified = {revision: verify_sources(revision=revision) for revision in revisions}
    groups = defaultdict(list)
    configs = {}
    for identifier, result in results.items():
        if result["world_size"] != 1:
            raise ValueError("Reconstruction requires the original single-rank setting")
        config = RunConfig.model_validate(result["config"])
        row = selected[identifier]
        if config.seed != row["seed"] or config.head != row["head"]:
            raise ValueError(f"Executed config mismatch: {identifier}")
        signature = {
            "head": config.head,
            "seed": config.seed,
            "pretrained": config.pretrained,
            "precision": config.precision,
            "eval_batch_size": config.eval_batch_size,
            "eval_limit": config.eval_limit,
            "data_fingerprint": result["data_fingerprint"],
            "training_source_cohort": row["cohort"],
        }
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        groups[key].append(identifier)
        configs[key] = config
    context = DistributedContext(rank=0, world_size=1, device=torch.device(device))
    records, mapping = {}, {}
    cuda = (
        torch.cuda.device(context.device)
        if context.device.type == "cuda"
        else nullcontext()
    )
    with cuda:
        for key, identifiers in groups.items():
            config = configs[key]
            torch.set_num_threads(config.cpu_threads)
            torch.use_deterministic_algorithms(config.deterministic)
            torch.backends.cudnn.benchmark = not config.deterministic
            set_seeds(config.seed)
            manifest = read_manifest(path=config.manifest)
            if manifest["fingerprint"] != selected[identifiers[0]]["fingerprint"]:
                raise ValueError("Changed validation split")
            make_loader(config=config, context=context, payload=manifest, split="train")
            loader = make_loader(
                config=config, context=context, payload=manifest, split="val"
            )
            with torch.device("cpu"):
                model = build_model(pretrained=config.pretrained, head=config.head)
            model = model.to(context.device)
            record = initial_validation(
                model=model,
                loader=loader,
                config=config,
                context=context,
                data_fingerprint=manifest["fingerprint"],
                origin="reconstructed",
            )
            records[key] = record
            mapping.update({identifier: key for identifier in identifiers})
            print(
                json.dumps({
                    "head": config.head,
                    "seed": config.seed,
                    "validation": record["validation"],
                    "seconds": record["seconds"],
                }),
                flush=True,
            )
            del model, loader
    report = {
        "generated": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "sourceDataSha256": hashlib.sha256(contents).hexdigest(),
        "sourceSnapshot": payload["generated"],
        "evaluationRevision": evaluation_revision,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "verifiedHistoricalSources": verified,
        "records": records,
        "runs": mapping,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(path=output, payload=report)
    print(
        json.dumps({
            "initializations": len(records),
            "runs": len(mapping),
            "output": str(output),
        })
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--plans", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    reconstruct(
        data=args.data, plans=tuple(args.plans), output=args.output, device=args.device
    )


if __name__ == "__main__":
    main()
