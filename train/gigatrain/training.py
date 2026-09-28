"""Lightning modules for the three kinds of job: back end, front end per language, decoder per voice."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import lightning as L
import torch
from torch import autocast
from torch.nn import functional as F

from .config import AudioConfig, ScheduleConfig, TrainConfig, parse_dataclass
from .data import Batch
from .ids import IdMaps
from .model import Latent, MultilingualVits, language_speaker_mask, make_decoder, masked_mean, speaker_adversary_loss
from .vits.commons import rand_slice_segments, slice_segments
from .vits.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .vits.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from .vits.models import MultiPeriodDiscriminator



class Phase(Enum):
    RECONSTRUCTION = "reconstruction"
    ADVERSARIAL = "adversarial"


def phase_at(step: int, schedule: ScheduleConfig) -> Phase:
    return Phase.RECONSTRUCTION if step < schedule.gan_start_step else Phase.ADVERSARIAL


def adversarial_scale(step: int, schedule: ScheduleConfig) -> float:
    if schedule.adversarial_ramp_steps == 0:
        return schedule.adversarial_weight
    return schedule.adversarial_weight * min(1.0, step / schedule.adversarial_ramp_steps)


def lr_gamma(config: TrainConfig) -> float:
    """Per-step decay that lands on lr_final_ratio at max_steps."""
    return config.optim.lr_final_ratio ** (1.0 / config.max_steps)


def step_seed(seed: int, step: int, rank: int) -> int:
    return (seed * 1_000_003 + step) * 64 + rank


def spectrogram(audio: torch.Tensor, audio_lengths: torch.Tensor, cfg: AudioConfig):
    """Linear spectrogram on the training device, in fp32 (cuFFT has no bf16)."""
    with autocast(audio.device.type, enabled=False):
        spec = spectrogram_torch(audio.squeeze(1).float(), cfg.filter_length, cfg.sample_rate, cfg.hop_length, cfg.win_length)
    return spec, audio_lengths // cfg.hop_length


def mel_of_audio(audio: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    with autocast(audio.device.type, enabled=False):
        return mel_spectrogram_torch(
            audio.squeeze(1).float(), cfg.filter_length, cfg.mel_channels, cfg.sample_rate,
            cfg.hop_length, cfg.win_length, cfg.mel_fmin, cfg.mel_fmax,
        )


def mel_of_spec(spec: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    with autocast(spec.device.type, enabled=False):
        return spec_to_mel_torch(spec.float(), cfg.filter_length, cfg.mel_channels, cfg.sample_rate, cfg.mel_fmin, cfg.mel_fmax)


@dataclass
class Reconstruction:
    y: torch.Tensor
    y_hat: torch.Tensor
    mel_l1: torch.Tensor


class _Job(L.LightningModule):
    """Shared construction, per-step seeding and a step counter that counts batches, not optimizer
    steps (Lightning's global_step doubles once the discriminator optimizer starts stepping)."""

    def __init__(self, config: dict, ids: dict, n_vocab: int):
        super().__init__()
        self.save_hyperparameters()
        self.config: TrainConfig = parse_dataclass(TrainConfig, config)
        self.ids = IdMaps.from_json(ids)
        audio = self.config.audio
        self.model = MultilingualVits(
            self.config.model,
            n_vocab,
            self.ids.languages,
            len(self.ids.speakers),
            len(self.ids.conditions),
            audio.filter_length // 2 + 1,
        )
        self.automatic_optimization = False
        self.step_count = 0

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["gigapiper_step"] = self.step_count

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.step_count = checkpoint["gigapiper_step"]

    def on_validation_epoch_start(self) -> None:
        # Same noise and segments at every validation, so val metrics compare across checkpoints;
        # training reseeds every step, so this does not leak into training.
        torch.manual_seed(self.config.seed)

    def _begin_step(self) -> None:
        torch.manual_seed(step_seed(self.config.seed, self.step_count, self.global_rank))

    def _end_step(self) -> None:
        for scheduler in self._schedulers():
            scheduler.step()
        self.step_count += 1
        if self.step_count >= self.config.max_steps:
            self.trainer.should_stop = True

    def _schedulers(self) -> list:
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return []
        return schedulers if isinstance(schedulers, list) else [schedulers]

    def _step_optimizer(self, optimizer, loss: torch.Tensor) -> None:
        optimizer.zero_grad()
        self.manual_backward(loss)
        if self.config.optim.grad_clip is not None:
            self.clip_gradients(optimizer, gradient_clip_val=self.config.optim.grad_clip, gradient_clip_algorithm="norm")
        optimizer.step()

    def _log_metrics(self, prefix: str, metrics: dict[str, torch.Tensor], batch_size: int) -> None:
        self.log_dict({f"{prefix}_{k}": v.detach().float() for k, v in metrics.items()}, batch_size=batch_size, sync_dist=prefix == "val")

    def _latent(self, batch: Batch) -> tuple[Latent, torch.Tensor, torch.Tensor]:
        spec, spec_lengths = spectrogram(batch.audio, batch.audio_lengths, self.config.audio)
        latent = self.model.latent(
            batch.phoneme_ids, batch.phoneme_lengths, spec, spec_lengths,
            batch.sid, batch.lid, batch.cid, batch.durations, batch.has_durations,
        )
        return latent, spec, spec_lengths

    def _adamw(self, params, lr: float, betas) -> torch.optim.AdamW:
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=self.config.optim.eps)

    def _exponential(self, optimizer) -> torch.optim.lr_scheduler.ExponentialLR:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma(self.config))


def reconstruct(decoder, z: torch.Tensor, g: torch.Tensor, spec: torch.Tensor, spec_lengths: torch.Tensor, audio: torch.Tensor, cfg: AudioConfig) -> Reconstruction:
    z_slice, ids_slice = rand_slice_segments(z, spec_lengths, cfg.segment_frames)
    y_hat = decoder(z_slice, g=g)
    y_mel = slice_segments(mel_of_spec(spec, cfg), ids_slice, cfg.segment_frames)
    y_hat_mel = mel_of_audio(y_hat, cfg)
    y = slice_segments(audio, ids_slice * cfg.hop_length, cfg.segment_size)
    return Reconstruction(y=y, y_hat=y_hat[..., : y.shape[-1]], mel_l1=F.l1_loss(y_mel, y_hat_mel))


def adversarial_generator_terms(disc, rec: Reconstruction) -> tuple[torch.Tensor, torch.Tensor]:
    _, y_d_hat_g, fmap_r, fmap_g = disc(rec.y, rec.y_hat)
    with autocast(rec.y.device.type, enabled=False):
        loss_gen, _ = generator_loss(y_d_hat_g)
        return loss_gen, feature_loss(fmap_r, fmap_g)


def discriminator_step_loss(disc, rec: Reconstruction) -> torch.Tensor:
    y_d_hat_r, y_d_hat_g, _, _ = disc(rec.y, rec.y_hat.detach())
    with autocast(rec.y.device.type, enabled=False):
        loss, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        return loss


class BackEndJob(_Job):
    """Everything trains: reconstruction losses first, GAN and feature matching from gan_start_step."""

    def __init__(self, config: dict, ids: dict, n_vocab: int):
        super().__init__(config, ids, n_vocab)
        self.disc = MultiPeriodDiscriminator()
        speakers = [[self.ids.sid(s) for s in self.ids.speakers_of(lang)] for lang in self.ids.languages]
        self.register_buffer("speaker_mask", language_speaker_mask(speakers, len(self.ids.speakers)), persistent=False)

    def _generator_losses(self, batch: Batch, phase: Phase, reversal: float):
        cfg = self.config
        latent, spec, spec_lengths = self._latent(batch)
        rec = reconstruct(self.model.dec, latent.z, latent.g, spec, spec_lengths, batch.audio, cfg.audio)
        pooled = masked_mean(latent.text.x, latent.text.x_mask)
        cond_logits, spk_logits = self.model.adversary(pooled.float(), reversal)
        with autocast(self.device.type, enabled=False):
            kl = kl_loss(latent.z_p, latent.logs_q, latent.m_p, latent.logs_p, latent.y_mask)
            cond_ce = F.cross_entropy(cond_logits.float(), batch.cid)
            spk_ce = speaker_adversary_loss(spk_logits, batch.sid, batch.lid, self.speaker_mask)
        total = rec.mel_l1 * cfg.optim.c_mel + kl * cfg.optim.c_kl + latent.duration_loss + cond_ce + spk_ce
        metrics = {"mel": rec.mel_l1, "kl": kl, "dur": latent.duration_loss, "cond_ce": cond_ce, "spk_ce": spk_ce}
        if phase == Phase.ADVERSARIAL:
            gen, fm = adversarial_generator_terms(self.disc, rec)
            total = total + gen + fm
            metrics |= {"gen": gen, "fm": fm}
        return total, rec, metrics

    def training_step(self, batch: Batch, batch_idx: int) -> None:
        self._begin_step()
        phase = phase_at(self.step_count, self.config.schedule)
        reversal = adversarial_scale(self.step_count, self.config.schedule)
        opt_g, opt_d = self.optimizers()
        loss_g, rec, metrics = self._generator_losses(batch, phase, reversal)
        self._step_optimizer(opt_g, loss_g)
        if phase == Phase.ADVERSARIAL:
            loss_d = discriminator_step_loss(self.disc, rec)
            self._step_optimizer(opt_d, loss_d)
            metrics["disc"] = loss_d
        metrics |= {"loss_g": loss_g, "reversal": torch.tensor(reversal)}
        self._log_metrics("train", metrics, len(batch))
        self._end_step()

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        _, _, metrics = self._generator_losses(batch, Phase.RECONSTRUCTION, 0.0)
        self._log_metrics("val", metrics, len(batch))

    def configure_optimizers(self):
        optim = self.config.optim
        opt_g = self._adamw(self.model.parameters(), optim.learning_rate, optim.betas)
        opt_d = self._adamw(self.disc.parameters(), optim.learning_rate_d, optim.betas_d)
        return [opt_g, opt_d], [self._exponential(opt_g), self._exponential(opt_d)]


def _rows_gradient_mask(rows: list[int]):
    def hook(grad: torch.Tensor) -> torch.Tensor:
        keep = torch.zeros_like(grad)
        keep[rows] = 1
        return grad * keep

    return hook


class PhonemeEmbedding(Enum):
    FROZEN = "frozen"
    TRAINED = "trained"


class FrontEndJob(_Job):
    """Front ends and their language-embedding rows against the frozen back end.

    Two uses: the warm-up before the main run (all languages, shared phoneme embedding trained)
    re-fits every front end to the shared embedding and lessac's back end; the post-freeze job
    (one language, embedding frozen) adds or replaces a language.

    Losses are KL (against the frozen posterior encoder and flow) and duration only: no decoder
    forward, no discriminators. Language rows train with weight decay off and the other rows'
    gradients masked to zero, so Adam leaves the other rows bit-identical.
    """

    def __init__(self, config: dict, ids: dict, n_vocab: int, languages: list[str], phoneme_embedding: str):
        super().__init__(config, ids, n_vocab)
        unknown = set(languages) - set(self.ids.languages)
        if unknown or not languages:
            raise ValueError(f"languages {sorted(unknown)} are not in the run's languages")
        self.languages = list(languages)
        self.lids = [self.ids.lid(lang) for lang in languages]
        self.phoneme_embedding = PhonemeEmbedding(phoneme_embedding)
        self.model.requires_grad_(False)
        for lang in languages:
            self.model.front_ends[lang].requires_grad_(True)
        self.model.emb.requires_grad_(self.phoneme_embedding is PhonemeEmbedding.TRAINED)
        self.model.cond.emb_lang.weight.requires_grad_(True)
        self.model.cond.emb_lang.weight.register_hook(_rows_gradient_mask(self.lids))

    def copy_language_row(self, target: str, source: str) -> None:
        """Start a new language's row from its closest trained language."""
        weight = self.model.cond.emb_lang.weight
        with torch.no_grad():
            weight[self.ids.lid(target)] = weight[self.ids.lid(source)]

    def _losses(self, batch: Batch) -> dict[str, torch.Tensor]:
        if bool((~torch.isin(batch.lid, torch.tensor(self.lids, device=batch.lid.device))).any()):
            raise ValueError("front-end job received items of a language it does not train")
        latent, _, _ = self._latent(batch)
        with autocast(self.device.type, enabled=False):
            kl = kl_loss(latent.z_p, latent.logs_q, latent.m_p, latent.logs_p, latent.y_mask)
        return {"kl": kl, "dur": latent.duration_loss}

    def training_step(self, batch: Batch, batch_idx: int) -> None:
        self._begin_step()
        metrics = self._losses(batch)
        loss = metrics["kl"] * self.config.optim.c_kl + metrics["dur"]
        self._step_optimizer(self.optimizers(), loss)
        self._log_metrics("train", metrics | {"loss": loss}, len(batch))
        self._end_step()

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        self._log_metrics("val", self._losses(batch), len(batch))

    def configure_optimizers(self):
        optim = self.config.optim
        groups = [{"params": [p for lang in self.languages for p in self.model.front_ends[lang].parameters()]}]
        if self.phoneme_embedding is PhonemeEmbedding.TRAINED:
            groups.append({"params": [self.model.emb.weight]})
        groups.append({"params": [self.model.cond.emb_lang.weight], "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups, lr=optim.learning_rate, betas=optim.betas, eps=optim.eps)
        return [opt], [self._exponential(opt)]


class DecoderJob(_Job):
    """Fine-tunes a copy of the shared decoder on one speaker; everything else is frozen.

    The copy starts from the shared decoder (see `start_from_shared_decoder`) and trains from the
    posterior latent of the voice's own audio with mel, GAN and feature-matching losses.
    """

    def __init__(self, config: dict, ids: dict, n_vocab: int, speaker: str):
        super().__init__(config, ids, n_vocab)
        if speaker not in self.ids.speakers:
            raise ValueError(f"{speaker!r} is not in the run's speakers")
        self.speaker = speaker
        self.voice_dec = make_decoder(self.config.model)
        self.disc = MultiPeriodDiscriminator()
        self.model.requires_grad_(False)

    def start_from_shared_decoder(self) -> None:
        self.voice_dec.load_state_dict(self.model.dec.state_dict())

    def _reconstruct(self, batch: Batch) -> Reconstruction:
        if bool((batch.sid != self.ids.sid(self.speaker)).any()):
            raise ValueError("decoder job received another speaker's items")
        spec, spec_lengths = spectrogram(batch.audio, batch.audio_lengths, self.config.audio)
        with torch.no_grad():
            g = self.model.cond(batch.sid, batch.lid, batch.cid)
            z, _, _, _ = self.model.enc_q(spec, spec_lengths, g=g)
        return reconstruct(self.voice_dec, z, g, spec, spec_lengths, batch.audio, self.config.audio)

    def training_step(self, batch: Batch, batch_idx: int) -> None:
        self._begin_step()
        opt_g, opt_d = self.optimizers()
        rec = self._reconstruct(batch)
        gen, fm = adversarial_generator_terms(self.disc, rec)
        loss_g = rec.mel_l1 * self.config.optim.c_mel + gen + fm
        self._step_optimizer(opt_g, loss_g)
        loss_d = discriminator_step_loss(self.disc, rec)
        self._step_optimizer(opt_d, loss_d)
        self._log_metrics("train", {"mel": rec.mel_l1, "gen": gen, "fm": fm, "disc": loss_d}, len(batch))
        self._end_step()

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        self._log_metrics("val", {"mel": self._reconstruct(batch).mel_l1}, len(batch))

    def configure_optimizers(self):
        optim = self.config.optim
        opt_g = self._adamw(self.voice_dec.parameters(), optim.learning_rate, optim.betas)
        opt_d = self._adamw(self.disc.parameters(), optim.learning_rate_d, optim.betas_d)
        return [opt_g, opt_d], [self._exponential(opt_g), self._exponential(opt_d)]
