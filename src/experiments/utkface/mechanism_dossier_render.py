import argparse
import csv
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from src.experiments.utkface.adaptation_train import file_sha256
from src.experiments.utkface.mechanism_dossier import (
    MAX_FILE_BYTES,
    load,
    require,
    write_checksums,
)
from src.experiments.utkface.run import write_json

RENDERER_SOURCE = Path("src/experiments/utkface/mechanism_dossier_render.py")


class HtmlInventory(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.math_elements = 0
        self.math_errors = 0
        self.figures = 0
        self.identifiers = set()
        self.references = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        attributes = dict(attrs)
        self.math_elements += tag == "math"
        self.math_errors += tag == "merror"
        self.figures += tag == "figure"
        if identifier := attributes.get("id"):
            self.identifiers.add(identifier)
        field = {"a": "href", "link": "href", "img": "src", "script": "src"}.get(tag)
        if field and (reference := attributes.get(field)):
            self.references.append((tag, reference))


def inspect_html(*, directory: Path, path: Path) -> dict:
    directory = directory.resolve()
    text = path.read_text(encoding="utf-8")
    document = HtmlInventory()
    document.feed(text)
    document.close()
    require(
        condition=text.rstrip().endswith("</html>")
        and document.math_elements > 0
        and document.math_errors == 0
        and document.figures > 0,
        path=path,
        expectation="complete HTML with mathematical expressions, figures and zero MathML error nodes",
    )
    local_files, anchors = set(), set()
    for tag, reference in document.references:
        uri = urlsplit(reference)
        if uri.scheme or uri.netloc:
            require(
                condition=uri.scheme not in {"file"}
                and len(uri.scheme) != 1
                and (tag == "a" or uri.scheme == "data"),
                path=path,
                expectation=f"embedded rendering resources: {reference[:120]}",
            )
            continue
        if uri.path:
            target = directory / unquote(uri.path)
            require(
                condition=target.resolve().is_relative_to(directory)
                and target.is_file()
                and not target.is_symlink(),
                path=path,
                expectation=f"a retained local data, note or zoom-image target: {uri.path}",
            )
            local_files.add(target.resolve().relative_to(directory).as_posix())
        elif uri.fragment:
            identifier = unquote(uri.fragment)
            require(
                condition=identifier in document.identifiers,
                path=path,
                expectation=f"an available internal anchor: {identifier}",
            )
            anchors.add(identifier)
    return {
        "math_elements": document.math_elements,
        "math_errors": document.math_errors,
        "figures": document.figures,
        "local_targets": sorted(local_files),
        "internal_anchors_checked": len(anchors),
        "rendering_resources_embedded": True,
    }


def verify_files(*, directory: Path, files: list[dict]) -> None:
    require(
        condition=len({row["path"] for row in files}) == len(files),
        path=directory,
        expectation="unique inventory paths",
    )
    for row in files:
        path = directory / row["path"]
        require(
            condition=path.resolve().is_relative_to(directory)
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == int(row["bytes"])
            and file_sha256(path=path) == row["sha256"],
            path=path,
            expectation="unchanged retained bytes and SHA-256",
        )


def render(*, dossier_dir: Path, quarto: Path) -> dict:
    directory = dossier_dir.resolve()
    manifest_path = directory / "dossier-manifest.json"
    manifest = load(path=manifest_path)
    require(
        condition=manifest["schema"] == "utkface-mechanism-dossier-v1"
        and manifest["scientific_complete"] is True
        and manifest["rendered"] is False,
        path=manifest_path,
        expectation="a complete scientific dossier awaiting its first verified render",
    )
    with (directory / "SHA256SUMS.tsv").open(encoding="utf-8", newline="") as stream:
        checksums = list(csv.DictReader(stream, delimiter="\t"))
    require(
        condition={row["path"] for row in checksums}
        == {row["path"] for row in manifest["files"]} | {manifest_path.name},
        path=directory,
        expectation="a checksum index covering every assembled file and the manifest",
    )
    verify_files(directory=directory, files=checksums)
    verify_files(directory=directory, files=manifest["files"])
    source = (
        directory
        / "source"
        / manifest["source_snapshot"]["git_commit"]
        / RENDERER_SOURCE
    )
    require(
        condition=source.is_file()
        and file_sha256(path=source) == file_sha256(path=Path(__file__)),
        path=source,
        expectation="the executing renderer preserved in the frozen source snapshot",
    )
    require(
        condition=quarto.is_file(),
        path=quarto,
        expectation="the requested Quarto executable",
    )
    version = subprocess.check_output([str(quarto), "--version"], text=True).strip()
    command = [
        str(quarto),
        "render",
        "report.qmd",
        "--to",
        "html",
        "--output",
        "report.html",
        "--no-execute",
    ]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    result = subprocess.run(
        command,
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    log = result.stdout + result.stderr
    (directory / "render.log").write_text(log, encoding="utf-8")
    require(
        condition=result.returncode == 0
        and re.search(r"\bwarning\b", log, flags=re.IGNORECASE) is None,
        path=directory / "render.log",
        expectation="successful Quarto rendering with zero warnings",
    )
    html = directory / "report.html"
    validation = {
        "status": "passed",
        "created_utc": datetime.now(UTC).isoformat(),
        "quarto_version": version,
        "command": command,
        "assembly_manifest_sha256": file_sha256(path=manifest_path),
        "renderer_sha256": file_sha256(path=Path(__file__)),
        "html_sha256": file_sha256(path=html),
        "html_checks": inspect_html(directory=directory, path=html),
        "code_execution": False,
        "gpu_execution": False,
    }
    verify_files(directory=directory, files=manifest["files"])
    write_json(path=directory / "render-validation.json", payload=validation)
    paths = {row["path"] for row in manifest["files"]}
    for path in sorted(directory.rglob("*")):
        if path.is_dir() or path in {manifest_path, directory / "SHA256SUMS.tsv"}:
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in paths:
            continue
        require(
            condition=not path.is_symlink() and path.stat().st_size <= MAX_FILE_BYTES,
            path=path,
            expectation="a bounded regular rendering artifact",
        )
        manifest["files"].append({
            "source": "quarto render and structural verification",
            "path": relative,
            "kind": "render",
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path=path),
        })
    manifest.update(
        rendered=True,
        rendering=validation,
        hash_scope="SHA256SUMS.tsv covers every retained dossier file except itself.",
    )
    write_json(path=manifest_path, payload=manifest)
    write_checksums(directory=directory, files=manifest["files"])
    verify_files(directory=directory, files=manifest["files"])
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dossier-dir", type=Path, required=True)
    parser.add_argument("--quarto", type=Path, required=True)
    args = parser.parse_args()
    result = render(dossier_dir=args.dossier_dir, quarto=args.quarto)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
