"""Stage-B extension: add speakers, languages and conditions to a trained back-end checkpoint.

IDs are only appended (ids.build_id_maps), so every existing row, front end and optimizer moment
keeps its meaning. The embedding tables and the classifier outputs grow by rows; new rows keep
their fresh initialisation, and Adam starts them from zero moments. A new language copies the
front end and language-embedding row of a named existing language, and optionally replaces the
text encoder and inference duration-predictor weights with a Piper voice's.
"""
from __future__ import annotations

import re

import numpy as np
import torch

from .ids import IdMaps
from .init_from_piper import FromVoice, source_for

GROWABLE = frozenset({
    "model.cond.emb_g.weight",
    "model.cond.emb_lang.weight",
    "model.cond.emb_cond.weight",
    "model.adversary.speaker.2.weight",
    "model.adversary.speaker.2.bias",
    "model.adversary.condition.2.weight",
    "model.adversary.condition.2.bias",
})


class ExtendError(ValueError):
    pass


def grow_rows(old: torch.Tensor, fresh: torch.Tensor, key: str) -> torch.Tensor:
    if old.shape[1:] != fresh.shape[1:] or old.shape[0] > fresh.shape[0]:
        raise ExtendError(f"{key}: cannot grow {tuple(old.shape)} into {tuple(fresh.shape)}")
    grown = fresh.clone()
    grown[: old.shape[0]] = old
    return grown


def _front_end_key(key: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"model\.front_ends\.([^.]+)\.(.+)", key)
    return (match.group(1), match.group(2)) if match else None


def check_extension(old: IdMaps, new: IdMaps, like: dict[str, str]) -> None:
    for field in ("speakers", "languages", "conditions"):
        before, after = getattr(old, field), getattr(new, field)
        if after[: len(before)] != before:
            raise ExtendError(f"{field} were renumbered")
    added = set(new.languages[len(old.languages):])
    if set(like) != added:
        raise ExtendError(f"new languages {sorted(added)} need exactly one closest existing language each, got {like}")
    for language, source in like.items():
        if source not in old.languages:
            raise ExtendError(f"{language}: {source!r} is not an existing language")


def grown_state(
    old_state: dict[str, torch.Tensor],
    fresh_state: dict[str, torch.Tensor],
    old_ids: IdMaps,
    new_ids: IdMaps,
    like: dict[str, str],
    voices: dict[str, dict[str, np.ndarray]],
) -> dict[str, torch.Tensor]:
    check_extension(old_ids, new_ids, like)
    missing = set(old_state) - set(fresh_state)
    if missing:
        raise ExtendError(f"extended model lacks {sorted(missing)[:5]}")
    out = {}
    for key, fresh in fresh_state.items():
        front_end = _front_end_key(key)
        if front_end and front_end[0] in like:
            language, name = front_end
            value = old_state[f"model.front_ends.{like[language]}.{name}"]
            source = source_for(key, 0)
            if language in voices and isinstance(source, FromVoice):
                if name not in voices[language]:
                    raise ExtendError(f"voice for {language!r} has no {name}")
                value = torch.from_numpy(np.array(voices[language][name]))
        elif key not in old_state:
            raise ExtendError(f"{key} is new but is not part of a new front end")
        elif key in GROWABLE:
            value = grow_rows(old_state[key], fresh, key)
        else:
            value = old_state[key]
        if value.shape != fresh.shape:
            raise ExtendError(f"{key}: shape {tuple(value.shape)}, extended model has {tuple(fresh.shape)}")
        out[key] = value.to(fresh.dtype).clone()
    rows = out["model.cond.emb_lang.weight"]
    for language, source in like.items():
        rows[new_ids.lid(language)] = rows[old_ids.lid(source)]
    return out


def grown_optimizer_state(
    old: dict, old_names: list[str], new_names: list[str], new_shapes: dict[str, torch.Size]
) -> dict:
    """Re-index an Adam(W) state dict (one param group) from old to new parameter order.

    Grown parameters get zero moments for their new rows; new parameters get no state, which Adam
    initialises on their first step.
    """
    if len(old["param_groups"]) != 1:
        raise ExtendError("expected a single parameter group")
    if set(old_names) - set(new_names):
        raise ExtendError("parameters disappeared")
    old_index = {name: i for i, name in enumerate(old_names)}
    state = {}
    for i, name in enumerate(new_names):
        if name not in old_index or old_index[name] not in old["state"]:
            continue
        moments = dict(old["state"][old_index[name]])
        for moment in ("exp_avg", "exp_avg_sq"):
            if moments[moment].shape != new_shapes[name]:
                if name not in GROWABLE:
                    raise ExtendError(f"{name}: optimizer moment changed shape")
                moments[moment] = grow_rows(moments[moment], torch.zeros(new_shapes[name], dtype=moments[moment].dtype), name)
        state[i] = moments
    group = dict(old["param_groups"][0]) | {"params": list(range(len(new_names)))}
    return {"state": state, "param_groups": [group]}
