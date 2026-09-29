"""train/configs/run1.json adapted to the box: stage A shards, calibrated loader and batching."""
import json
import sys
from pathlib import Path

shards = sorted(str(p) for p in Path("/workspace/teacher-stage-a/shards").glob("shard-*"))
c = json.loads(Path("/workspace/gigapiper/train/configs/run1.json").read_text())
c["data"].update(
    shards=shards,
    token_table="/workspace/gigapiper/tokens/table.json",
    # the calibration found loader workers make no difference beyond a few (training is bound by the main process)
    num_workers=10,
)
c["keep_every_n_epochs"] = 5
Path(sys.argv[1]).write_text(json.dumps(c, indent=2))
print(f"{len(shards)} shards -> {sys.argv[1]}")
