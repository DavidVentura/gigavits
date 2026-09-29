"""Step time of the back-end generator loss (forward + backward + AdamW step) against the number of
distinct languages in a batch, on synthetic fixed-duration items.

    python prof_langs.py <train_config.json> [--batch 32] [--langs 1,5,16,32] [--steps 20]
"""
import argparse
import dataclasses
import time
from pathlib import Path

import torch

from gigatrain.config import load_config, to_jsonable
from gigatrain.data import Batch
from gigatrain.records import Vocab
from gigatrain.training import BackEndJob, Phase


def ids_json(n_languages: int) -> dict:
    languages = [f"l{i}" for i in range(n_languages)]
    speakers = [f"s{i}" for i in range(2 * n_languages)]
    return {
        "speakers": speakers,
        "languages": languages,
        "conditions": ["clean", "narrow_band", "degraded"],
        "language_espeak": {lang: "en" for lang in languages},
        "language_speakers": {lang: [f"s{2 * i}", f"s{2 * i + 1}"] for i, lang in enumerate(languages)},
    }


def batch(size: int, n_languages: int, n_vocab: int, device: str) -> Batch:
    g = torch.Generator().manual_seed(0)
    t_x = torch.randint(80, 240, (size,), generator=g)
    durations = torch.zeros(size, int(t_x.max()), dtype=torch.long)
    for i, n in enumerate(t_x.tolist()):
        durations[i, :n] = torch.randint(1, 8, (n,), generator=g)
    frames = durations.sum(1)
    audio = torch.randn(size, 1, int(frames.max()) * 256, generator=g) * 0.1
    ids = torch.randint(3, n_vocab, (size, int(t_x.max())), generator=g)
    lid = torch.arange(size) % n_languages
    made = Batch(
        phoneme_ids=ids, phoneme_lengths=t_x, durations=durations, has_durations=torch.ones(size, dtype=torch.bool),
        audio=audio, audio_lengths=frames * 256, sid=lid * 2, lid=lid, cid=torch.zeros(size, dtype=torch.long),
    )
    return Batch(**{f.name: getattr(made, f.name).to(device) for f in dataclasses.fields(Batch)})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--langs", default="1,5,16,32", help="distinct languages per batch")
    p.add_argument("--model-langs", type=int, help="front ends in the model (default: as many as in the batch)")
    p.add_argument("--fused", action="store_true", help="fused AdamW")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--trace", action="store_true", help="print the torch profiler's top ops for the last step")
    args = p.parse_args()
    config = load_config(args.config)
    vocab = Vocab.load(Path(config.data.token_table))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    for n in [int(x) for x in args.langs.split(",")]:
        job = BackEndJob(to_jsonable(config), ids_json(max(n, args.model_langs or n)), vocab.size).cuda()
        opt = torch.optim.AdamW(job.model.parameters(), lr=1e-4, fused=args.fused or None)
        b = batch(args.batch, n, vocab.size, "cuda")
        times = []
        for step in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = job._generator_losses(b, Phase.RECONSTRUCTION, 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        if args.trace:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _, _ = job._generator_losses(b, Phase.RECONSTRUCTION, 0.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        steady = sorted(times[5:])[len(times[5:]) // 2]
        print(f"languages {n:3d}: median step {steady * 1000:.0f} ms", flush=True)
        del job, opt


if __name__ == "__main__":
    main()
