"""One job: render a part of one speaker's utterances, filter them, write FLAC + items (shell)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from remap import TeacherInput, teacher_input
from tokenizer import Language, Table, piper_input, tokenize

from . import TOKENS_DIR
from .align import HOP, TARGET_RATE, AlignmentError, rescale_durations
from .audio import fit_frames, peak_normalize, to_target_rate, write_flac
from .catalog import (
    CoquiSource, GlowTtsSource, KokoroJaSource, KokoroSource, Mimic3Source, MmsSource, PhonemeType, PiperSource,
    Speaker,
)
from .config import NoisePolicy, Paths, QualityConfig
from .plan import Job, Utterance, int_hash, part_utterances, utterance_id
from .quality import Drop, TeacherTiming, duration_outliers, longest_internal_silence
from .rt import Phonemized, TeacherRuntime
from .sentences import phonetic_ids

PIPER_DEFAULT_NOISE = 0.667
PIPER_DEFAULT_NOISE_W = 0.8


class TeacherInputMismatch(AssertionError):
    pass


@dataclass(frozen=True)
class RenderContext:
    paths: Paths
    noise: NoisePolicy
    quality: QualityConfig
    threads: int


@dataclass(frozen=True)
class Take:
    """One render before filtering."""

    utterance: Utterance
    phonemes: str
    phoneme_ids: list[int]
    durations: list[int] | None
    audio: np.ndarray  # TARGET_RATE
    timing: TeacherTiming | None


@dataclass(frozen=True)
class JobStats:
    job_id: str
    teacher: str
    speaker: str
    language: str
    rendered: int
    rendered_seconds: float
    kept: int
    kept_seconds: float
    dropped: dict[str, int]
    unaligned: int


def job_dir(out: Path, job_id: str) -> Path:
    return out / "work" / job_id


def model_path(out: Path, voice: str) -> Path:
    return out / "models" / f"{voice}.durations.onnx"


def prepare_model(source: PiperSource, out: Path, voice: str) -> Path:
    """fp32 teacher graph with the `durations` output (durations/expose_durations.py)."""
    import onnx
    from expose_durations import expose_durations

    target = model_path(out, voice)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".tmp{os.getpid()}")
    onnx.save(expose_durations(onnx.load(source.onnx)), tmp)
    tmp.rename(target)
    return target


def _fp32(out: Path, mnn: Path, name: str) -> tuple[Path, Path]:
    """The app's MNN teachers hold int8 weights; each is rebuilt from the fp32 ONNX beside it."""
    return mnn.with_suffix(".onnx"), out / "models" / f"{name}.fp32.mnn"


def engine_conversions(speaker: Speaker, paths: Paths) -> dict[str, tuple[Path, Path]]:
    """Role -> (fp32 ONNX, fp32 MNN to build) for the MNN graphs an engine teacher runs."""
    s, out, v = speaker.source, paths.out, speaker.voice
    match s:
        case KokoroSource() | KokoroJaSource():
            return {"model": (paths.kokoro_onnx, out / "models" / "kokoro.fp32.mnn")}
        case MmsSource() | CoquiSource():
            return {"model": _fp32(out, s.model, v)}
        case Mimic3Source():
            return {"mnn": _fp32(out, s.mnn, v)}
        case GlowTtsSource():
            return {"model": _fp32(out, s.model, v + ".glowtts"), "vocoder": _fp32(out, s.vocoder, v + ".hifigan")}
    raise TypeError(f"{speaker.key}: {type(s).__name__} is not an engine teacher")


def patch_kokoro(source: Path, target: Path) -> Path:
    """Upstream Kokoro ONNX -> the graph MNN converts faithfully (see kokoro_patch.py)."""
    from .kokoro_patch import patch

    if target.exists():
        return target
    tmp = target.with_name(target.name + f".tmp{os.getpid()}")
    patch(source, tmp)
    tmp.rename(target)
    return target


def convert_model(teacher_rt: Path, onnx: Path, mnn: Path) -> Path:
    if mnn.exists():
        return mnn
    if not onnx.exists():
        raise FileNotFoundError(f"fp32 source {onnx} is missing; int8 app models are not used as teachers")
    mnn.parent.mkdir(parents=True, exist_ok=True)
    tmp = mnn.with_name(mnn.name + f".tmp{os.getpid()}")
    subprocess.run([str(teacher_rt), "convert", str(onnx), str(tmp)], check=True, capture_output=True)
    tmp.rename(mnn)
    return mnn


def runtime_spec(speaker: Speaker, paths: Paths) -> dict:
    s = speaker.source
    e = speaker.espeak
    if isinstance(s, PiperSource):
        return {"engine": "piper", "config": str(s.config_path), "mnn": str(s.mnn)}
    m = {role: str(mnn) for role, (_, mnn) in engine_conversions(speaker, paths).items()}
    match s:
        case KokoroSource():
            return {"engine": "kokoro", "model": m["model"], "voices": str(paths.kokoro_voices), "voice": s.voice, "espeak": e}
        case KokoroJaSource():
            return {
                "engine": "kokoro_ja", "model": m["model"], "voices": str(paths.kokoro_voices),
                "voice": s.voice, "dict": str(s.dict_path), "espeak": e,
            }
        case MmsSource():
            return {"engine": "mms", "model": m["model"], "tokens": str(s.tokens), "espeak": e}
        case CoquiSource():
            return {"engine": "coqui", "model": m["model"], "config": str(s.config), "espeak": e}
        case Mimic3Source():
            return {"engine": "mimic3", "config": str(s.config), "mnn": m["mnn"], "espeak": e}
        case GlowTtsSource():
            return {"engine": "glowtts", "model": m["model"], "vocoder": m["vocoder"], "lexicon": str(s.lexicon), "espeak": e}
    raise TypeError(f"{speaker.key}: {type(s).__name__} is not rendered by this generator")


class PiperTeacher:
    """Renders with onnxruntime from the duration-exposing graph, feeding exactly the IDs piper-rs
    would feed (checked against piper-rs for every sentence)."""

    def __init__(self, job: Job, ctx: RenderContext, rt: TeacherRuntime, table: Table, language: Language):
        import onnxruntime as ort

        source = job.speaker.source
        assert isinstance(source, PiperSource)
        self.config = source.config
        # onnxruntime seeds RandomNormalLike when a session is created; one seed per job part, so
        # the second render of a sentence (always in another part) draws different noise.
        ort.set_seed(int_hash(job.seed, job.job_id))
        options = ort.SessionOptions()
        options.intra_op_num_threads = ctx.threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path(ctx.paths.out, job.speaker.voice)), options, providers=["CPUExecutionProvider"])
        match ctx.noise:
            case NoisePolicy.PIPER_DEFAULTS:
                noise, noise_w = PIPER_DEFAULT_NOISE, PIPER_DEFAULT_NOISE_W
            case NoisePolicy.VOICE_CONFIG:
                noise, noise_w = self.config.noise_scale, self.config.noise_w
        # articulation rates were measured at the voice's configured length scale
        self.scales = np.array([noise, self.config.length_scale * job.tempo, noise_w], dtype=np.float32)
        self.sid = job.speaker.sid
        self.rt, self.table, self.language = rt, table, language
        self.phonemized: dict[str, Phonemized] = {}
        self.unaligned = 0

    def _positions_match(self, ph: Phonemized, teacher: TeacherInput) -> bool:
        """Teacher durations carry over only when the teacher read the student's espeak string one
        character per position (PLAN 2.3); `text` voices and lossy old maps are aligned in training."""
        if self.config.phoneme_type is PhonemeType.TEXT:
            return False
        assert ph.teacher is not None and ph.teacher.fed == piper_input(ph.phonemes)
        return not teacher.dropped and not teacher.whitespace_split

    def take(self, u: Utterance, scratch: Path) -> Take:
        if u.text not in self.phonemized:
            self.phonemized[u.text] = self.rt.phonemize(u.text)
        ph = self.phonemized[u.text]
        assert ph.teacher is not None
        teacher = teacher_input(ph.teacher.fed, self.config.phoneme_id_map)
        if tuple(teacher.ids[2:-1:2]) != ph.teacher.piper_ids:
            raise TeacherInputMismatch(f"{u.text!r}: remap.teacher_input differs from piper-rs")
        feed = {
            "input": np.array([teacher.ids], dtype=np.int64),
            "input_lengths": np.array([len(teacher.ids)], dtype=np.int64),
            "scales": self.scales,
        }
        if self.sid is not None:
            feed["sid"] = np.array([self.sid], dtype=np.int64)
        audio, durations = self.session.run(["output", "durations"], feed)
        audio = audio.reshape(-1)
        teacher_durations = [int(d) for d in durations.reshape(-1)]
        if len(teacher_durations) != len(teacher.ids):
            raise AlignmentError(f"{len(teacher_durations)} durations for {len(teacher.ids)} IDs")
        if sum(teacher_durations) * HOP != audio.size:
            raise AlignmentError(f"sum(durations) * {HOP} = {sum(teacher_durations) * HOP} != {audio.size} samples")
        audio = peak_normalize(audio)
        student_ids = tokenize(piper_input(ph.phonemes), self.language, self.table)
        durations_out: list[int] | None = None
        if self._positions_match(ph, teacher):
            if len(student_ids) != len(teacher_durations):
                raise AlignmentError(f"{len(student_ids)} student IDs for {len(teacher_durations)} teacher positions")
            durations_out = rescale_durations(teacher_durations, self.config.sample_rate)
            audio = fit_frames(to_target_rate(audio, self.config.sample_rate), sum(durations_out))
        else:
            self.unaligned += 1
            audio = to_target_rate(audio, self.config.sample_rate)
        return Take(
            utterance=u,
            phonemes=ph.phonemes,
            phoneme_ids=student_ids,
            durations=durations_out,
            audio=audio,
            timing=TeacherTiming(tuple(teacher.ids), tuple(teacher_durations)),
        )


class EngineTeacher:
    """Kokoro, MMS, Coqui, mimic3, GlowTTS through piper-rs: audio only; the student input is the
    text's espeak phonemes, aligned in training (PLAN 2.3b)."""

    def __init__(self, job: Job, rt: TeacherRuntime, table: Table, language: Language):
        self.speed = 1.0 / job.tempo
        self.rt, self.table, self.language = rt, table, language
        self.unaligned = 0

    def take(self, u: Utterance, scratch: Path) -> Take:
        r = self.rt.render(u.text, self.speed, scratch)
        if r.audio.size == 0:
            raise AlignmentError(f"{u.text!r}: teacher rendered no audio")
        return Take(
            utterance=u,
            phonemes=r.phonemes,
            phoneme_ids=tokenize(piper_input(r.phonemes), self.language, self.table),
            durations=None,
            audio=to_target_rate(r.audio, r.sample_rate),
            timing=None,
        )


def _record(job: Job, take: Take, audio_rel: str) -> dict:
    s = job.speaker
    return {
        "id": utterance_id(s.key, take.utterance),
        "audio": audio_rel,
        "language": s.language,
        "espeak": s.espeak,
        "speaker": s.key,
        "condition": s.condition.value,
        "phoneme_ids": take.phoneme_ids,
        "durations": take.durations,
        "text": take.utterance.text,
        "teacher": s.voice,
        "weight": s.weight,
    }


def _takes(teacher: PiperTeacher | EngineTeacher, job: Job, scratch: Path) -> Iterator[Take]:
    rendered = 0.0
    for u in part_utterances(job):
        if rendered >= job.budget_seconds:
            return
        take = teacher.take(u, scratch)
        rendered += take.audio.size / TARGET_RATE
        yield take


def drops_for(takes: list[Take], phonemes: frozenset[int], q: QualityConfig) -> dict[int, tuple[Drop, float]]:
    """Index -> (reason, measured value: largest z, or silence seconds)."""
    drops: dict[int, tuple[Drop, float]] = {}
    timed = [(i, t.timing) for i, t in enumerate(takes) if t.timing is not None]
    for j, z in duration_outliers([t for _, t in timed], phonemes, q.duration_z, q.min_symbol_count).items():
        drops[timed[j][0]] = (Drop.DURATION_OUTLIER, z)
    for i, t in enumerate(takes):
        if i in drops:
            continue
        silence = longest_internal_silence(t.audio, TARGET_RATE, q.silence_frame_s, q.silence_db)
        if silence > q.max_internal_silence_s:
            drops[i] = (Drop.INTERNAL_SILENCE, silence)
    return drops


def run_job(job: Job, ctx: RenderContext) -> JobStats:
    d = job_dir(ctx.paths.out, job.job_id)
    done = d / "done.json"
    if done.exists():
        return JobStats(**json.loads(done.read_text(encoding="utf-8")))
    if d.exists():
        shutil.rmtree(d)
    (d / "audio").mkdir(parents=True)
    table = Table.load(TOKENS_DIR / "table.json")
    language = table.language(job.speaker.espeak)
    spec = runtime_spec(job.speaker, ctx.paths)
    scratch = d / "scratch.f32"
    with TeacherRuntime(ctx.paths.teacher_rt, ctx.paths.espeak_data, spec, d / "teacher-rt.log") as rt:
        teacher: PiperTeacher | EngineTeacher
        if isinstance(job.speaker.source, PiperSource):
            teacher = PiperTeacher(job, ctx, rt, table, language)
        else:
            teacher = EngineTeacher(job, rt, table, language)
        takes = list(_takes(teacher, job, scratch))
    drops = drops_for(takes, phonetic_ids(table), ctx.quality)
    kept_seconds = 0.0
    with (d / "items.jsonl").open("w", encoding="utf-8") as items, (d / "drops.jsonl").open("w", encoding="utf-8") as dropped:
        for i, take in enumerate(takes):
            rel = f"audio/{utterance_id(job.speaker.key, take.utterance)}.flac"
            record = _record(job, take, rel)
            if i in drops:
                reason, value = drops[i]
                dropped.write(json.dumps({"id": record["id"], "reason": reason.value, "value": round(value, 3), "text": record["text"]}, ensure_ascii=False) + "\n")
                continue
            write_flac(d / rel, take.audio)
            kept_seconds += take.audio.size / TARGET_RATE
            items.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = JobStats(
        job_id=job.job_id,
        teacher=job.speaker.voice,
        speaker=job.speaker.key,
        language=job.speaker.language,
        rendered=len(takes),
        rendered_seconds=sum(t.audio.size for t in takes) / TARGET_RATE,
        kept=len(takes) - len(drops),
        kept_seconds=kept_seconds,
        dropped=dict(Counter(r.value for r, _ in drops.values())),
        unaligned=teacher.unaligned,
    )
    done.write_text(json.dumps(stats.__dict__, indent=1) + "\n", encoding="utf-8")
    return stats
