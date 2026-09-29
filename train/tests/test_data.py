import logging
from pathlib import Path

import torch

from gigatrain.config import AudioConfig, DataConfig
from gigatrain.data import Example, Item, collate, select_items
from gigatrain.records import Condition, Record


def item(language: str, seconds: float) -> Item:
    record = Record("x", Path("/a.flac"), language, language, "s", Condition.CLEAN, (1, 0, 2), None, "", "t", 1.0)
    return Item(record, int(seconds * 22050))


def test_long_items_are_dropped_and_reported_per_language(caplog):
    data = DataConfig(shards=(), token_table="", max_utterance_s=12.0)
    items = [item("en", 3.0), item("en", 12.5), item("es", 20.0), item("es", 11.9)]
    with caplog.at_level(logging.INFO):
        kept = select_items(items, data, AudioConfig())
    assert [i.num_samples for i in kept] == [items[0].num_samples, items[3].num_samples]
    assert "en: dropped 1 items" in caplog.text and "es: dropped 1 items" in caplog.text


def test_collate_precomputes_rows_on_cpu():
    def example(lid: int, durations: bool) -> Example:
        return Example(torch.tensor([1, 0, 2]), torch.tensor([1, 1, 1]) if durations else None, torch.zeros(800), 0, lid, 0)

    batch = collate([example(3, True), example(1, False), example(3, False)])
    assert batch.teacher_rows.tolist() == [0]
    assert batch.searched_rows.tolist() == [1, 2]
    assert [(lid, rows.tolist()) for lid, rows in batch.language_rows] == [(1, [1]), (3, [0, 2])]
