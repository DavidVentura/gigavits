"""Which Piper voices were fine-tuned from lessac: cosine between each voice ONNX's text-encoder layer
weights and lessac's. Prints voice, quality, cosine (TSV)."""
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

lessac = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
ref = {k.removeprefix("model_g."): v.float().numpy() for k, v in lessac.items() if ".enc_p.encoder." in "." + k.removeprefix("model_g.")}
for path in sorted(Path(sys.argv[2]).rglob("*.onnx")):
    if path.name.startswith("kokoro") or "/tts/1/" not in str(path):
        continue
    try:
        inits = {t.name: t for t in onnx.load(str(path), load_external_data=False).graph.initializer}
    except Exception as e:
        print(f"{path.stem}\t?\terror {e}")
        continue
    a, b = [], []
    for name, t in inits.items():
        if name in ref:
            w = numpy_helper.to_array(t)
            if w.shape == ref[name].shape:
                a.append(w.ravel()); b.append(ref[name].ravel())
    if not a:
        print(f"{path.stem}\t{path.parent.name}\tnone")
        continue
    a, b = np.concatenate(a), np.concatenate(b)
    print(f"{path.stem}\t{path.parent.name}\t{float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)):.3f}\t{len(a)}")
