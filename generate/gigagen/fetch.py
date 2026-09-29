"""Builds the teacher-model mirror on the box from public URLs (the app's catalogs), never the laptop.

Paths in the voice lists are bucket paths; the app catalogs give each bucket path (`mirrorPath`) a
public URL: fp32 ONNX and sidecars from source_catalog.json (Hugging Face, offline-translator), the
app's MNN files from its index. Engine teachers render from fp32 conversions of the ONNX, so their
int8 MNN files are not fetched; Piper voices need the MNN because piper-rs phonemizes through it.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse

from .catalog import (
    CoquiSource, GlowTtsSource, KokoroJaSource, KokoroSource, Mimic3Source, MmsSource, Tier, parse_kokoro_list,
    parse_other_list, parse_table, parse_tiers, parse_voice_list,
)
from .config import Config

MNN_HOST = "offline-translator.davidv.dev"


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Remote:
    # tried in order; the first answering with the expected size is used
    urls: tuple[str, ...]
    size: int | None  # None where no catalog records it


@dataclass(frozen=True)
class Needed:
    path: Path
    remote: Remote | None  # None: no public URL known


def url_index(source_catalog: dict, mnn_catalog: dict, extra: list[dict[str, str]]) -> dict[str, Remote]:
    """Bucket-relative path -> URLs. The app's own mirror serves every file at its bucket path
    with the bytes the app uses, so it is tried first; the catalog's upstream URL (Hugging Face,
    GitHub) covers files the mirror lacks. Upstream files can drift from the catalog size, and
    the size check rejects them."""
    found: dict[str, tuple[list[str], int | None]] = {}

    def add(path: str, url: str, size: int | None) -> None:
        urls, known = found.setdefault(path, ([], None))
        if url not in urls:
            urls.append(url)
        found[path] = (urls, known or size)

    for pack in mnn_catalog["packs"].values():
        for f in pack.get("files", []):
            u = urlparse(f["url"])
            if u.netloc == MNN_HOST:
                add(u.path.lstrip("/"), f["url"], f.get("sizeBytes") or None)
    for pack in source_catalog["packs"].values():
        for f in pack.get("files", []):
            if "mirrorPath" in f:
                add(f["mirrorPath"], f"https://{MNN_HOST}/{f['mirrorPath']}", f.get("sizeBytes") or None)
                add(f["mirrorPath"], f["url"], f.get("sizeBytes") or None)
    for row in extra:
        add(row["path"], row["url"], int(row["size"]))
    return {path: Remote(tuple(urls), size) for path, (urls, size) in found.items()}


def _siblings(model: Path, bucket: Path, index: dict[str, Remote]) -> list[Path]:
    """Every catalogued file in an engine model's directory except MNN graphs (tokens, configs,
    speaker and language ID maps)."""
    directory = model.parent.relative_to(bucket)
    return [
        bucket / p for p in index
        if Path(p).parent == directory and not p.endswith((".mnn", ".mnn.weight"))
    ]


def required_paths(
    config: Config, voices: set[str], piper: dict[str, Path], engines: dict, index: dict[str, Remote]
) -> list[Path]:
    bucket = config.paths.bucket
    out: list[Path] = []
    for voice in sorted(voices):
        if voice in piper:
            cfg = piper[voice]
            stem = cfg.name.removesuffix(".onnx.json")
            out += [cfg, cfg.with_name(stem + ".onnx"), cfg.with_name(stem + ".mnn")]
            continue
        source, _ = engines[voice]
        match source:
            case KokoroSource():
                out.append(config.paths.kokoro_voices)
            case KokoroJaSource():
                out += [config.paths.kokoro_voices, source.dict_path]
            case MmsSource() | CoquiSource():
                out += [source.model.with_suffix(".onnx"), *_siblings(source.model, bucket, index)]
            case Mimic3Source():
                out += [source.mnn.with_suffix(".onnx"), *_siblings(source.mnn, bucket, index)]
            case GlowTtsSource():
                out += [source.model.with_suffix(".onnx"), source.vocoder.with_suffix(".onnx"), source.lexicon]
    return sorted(set(out))


def plan_fetch(config: Config) -> list[Needed]:
    p = config.paths
    index = url_index(
        json.loads(p.source_catalog.read_text(encoding="utf-8")),
        json.loads(p.mnn_catalog.read_text(encoding="utf-8")),
        parse_table(p.extra_sources.read_text(encoding="utf-8")),
    )
    rows = parse_tiers(p.tiers.read_text(encoding="utf-8"))
    sel = config.selection
    voices = {
        r.voice for r in rows
        if r.tier is not Tier.EXCLUDED and r.voice not in sel.exclude_voices and r.espeak not in sel.exclude_espeak
        and (not sel.voices or r.voice in sel.voices)
    }
    piper = parse_voice_list(p.voice_list.read_text(encoding="utf-8"), p.bucket_in_lists, p.bucket)
    engines = dict(parse_kokoro_list(p.kokoro_list.read_text(encoding="utf-8")))
    engines.update(parse_other_list(p.other_list.read_text(encoding="utf-8"), p.bucket_in_lists, p.bucket))
    rendered = {v for v in voices if v in piper or v in engines}
    # catalog.load parses the config of every listed Piper voice, rendered or not
    paths = set(required_paths(config, rendered, piper, engines, index)) | set(piper.values())
    needed = [Needed(path, index.get(str(path.relative_to(p.bucket)))) for path in sorted(paths)]
    if any(isinstance(engines.get(v, (None,))[0], (KokoroSource, KokoroJaSource)) for v in rendered):
        needed.append(Needed(p.kokoro_source_onnx, Remote((config.kokoro_url,), None)))
    return needed


def _ascii(url: str) -> str:
    # voice names such as pt_PT-tugão appear verbatim in catalog URLs
    return quote(url, safe=":/?=&%")


def _head(url: str, size: int | None) -> str | None:
    """None if the URL answers with the expected size, else what went wrong."""
    request = urllib.request.Request(_ascii(url), method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as r:
            length = r.headers.get("Content-Length")
    except urllib.error.URLError as e:
        return str(e)
    if size is not None and (length is None or int(length) != size):
        return f"size {length} != catalog {size}"
    return None


def resolve(remote: Remote) -> str:
    errors = []
    for url in remote.urls:
        error = _head(url, remote.size)
        if error is None:
            return url
        errors.append(f"{url}: {error}")
    raise FetchError("; ".join(errors))


def _try_resolve(n: Needed) -> str | None:
    assert n.remote is not None
    try:
        resolve(n.remote)
    except FetchError as e:
        return str(e)
    return None


def check(config: Config) -> None:
    """HEAD every needed URL; downloads nothing."""
    needed = plan_fetch(config)
    missing = [n.path for n in needed if n.remote is None]
    with ThreadPoolExecutor(16) as pool:
        problems = {
            n.path: e for n, e in zip(needed, pool.map(lambda n: _try_resolve(n) if n.remote else None, needed)) if e
        }
    print(f"{len(needed)} files needed, {len(missing)} without a public URL, {len(problems)} URLs failing")
    for path in missing:
        print(f"  no URL: {path}")
    for path, e in problems.items():
        print(f"  failing: {path}: {e}")
    if missing or problems:
        raise FetchError("mirror is incomplete")


def _download(n: Needed) -> Path:
    assert n.remote is not None
    if n.path.exists() and (n.remote.size is None or n.path.stat().st_size == n.remote.size):
        return n.path
    n.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = n.path.with_name(n.path.name + f".part{os.getpid()}")
    url = resolve(n.remote)
    with urllib.request.urlopen(_ascii(url), timeout=600) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if n.remote.size is not None and tmp.stat().st_size != n.remote.size:
        raise FetchError(f"{url}: got {tmp.stat().st_size} bytes, catalog says {n.remote.size}")
    tmp.rename(n.path)
    return n.path


def fetch(config: Config) -> None:
    needed = plan_fetch(config)
    missing = [n.path for n in needed if n.remote is None]
    if missing:
        raise FetchError(f"no public URL for {[str(m) for m in missing]}")
    with ThreadPoolExecutor(8) as pool:
        for path in pool.map(_download, needed):
            print(f"have {path}", flush=True)
