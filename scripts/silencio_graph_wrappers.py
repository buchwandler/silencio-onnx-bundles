from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TextEncoderWrapper(nn.Module):
    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student

    def forward(
        self, phoneme_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        student = self.student
        predictor = student.predictor
        style = student.style
        bert_features = student.bert_encoder(
            student.bert(phoneme_ids, attention_mask=torch.ones_like(phoneme_ids))
        ).transpose(-1, -2)
        token_count = phoneme_ids.shape[1]
        expanded_style = style[:, :, None].expand(-1, -1, token_count)
        state = torch.cat((bert_features, expanded_style), dim=1)

        for layer in predictor.text_encoder.lstms:
            if isinstance(layer, nn.LSTM):
                state = layer(state.transpose(-1, -2))[0].transpose(-1, -2)
            else:
                state = torch.cat(
                    (layer(state.transpose(-1, -2), style).transpose(-1, -2), expanded_style),
                    dim=1,
                )
        duration_state = state.transpose(-1, -2)
        raw_durations = torch.sigmoid(
            predictor.duration_proj(predictor.lstm(duration_state)[0])
        ).sum(-1)

        text_encoder = student.text_encoder
        asr_state = text_encoder.embedding(phoneme_ids).transpose(1, 2)
        for layer in text_encoder.cnn:
            asr_state = layer(asr_state)
        asr_token_features = student.asr_proj(
            text_encoder.lstm(asr_state.transpose(1, 2))[0]
        )
        return asr_token_features, duration_state, raw_durations


class TextProsodyWrapper(nn.Module):
    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student

    def forward(self, duration_state_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        predictor = self.student.predictor
        return predictor.F0Ntrain(
            duration_state_frames.transpose(-1, -2), self.student.style
        )


class AdapterWrapper(nn.Module):
    def __init__(self, adapter: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter

    def forward(
        self,
        asr_features: torch.Tensor,
        pitch: torch.Tensor,
        loudness: torch.Tensor,
        bnf_length: torch.Tensor,
    ) -> torch.Tensor:
        adapter = self.adapter
        input_mask = torch.ones_like(asr_features[:, :1, :])
        features = adapter.inp(
            torch.cat((asr_features, adapter.prosody(pitch, loudness)), dim=1)
        ) * input_mask
        for block in adapter.pre:
            features = block(features, input_mask)

        features = F.interpolate(
            features, size=bnf_length[0], mode="linear", align_corners=False
        )
        output_mask = torch.ones_like(features[:, :1, :])
        for block in adapter.post:
            features = block(features, output_mask)
        content = adapter.out(adapter.norm(features.transpose(1, 2)))
        content = content * adapter.bnf_std + adapter.bnf_mean
        return content * output_mask.transpose(1, 2)


class DiTStepWrapper(nn.Module):
    def __init__(self, dit: nn.Module) -> None:
        super().__init__()
        self.dit = dit

    def forward(
        self,
        noisy_mel: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        condition: torch.Tensor,
        speaker_embedding: torch.Tensor,
        position_offset: torch.Tensor,
        cache_key_l0: torch.Tensor,
        cache_key_l1: torch.Tensor,
        cache_key_l2: torch.Tensor,
        cache_key_l3: torch.Tensor,
        cache_value_l0: torch.Tensor,
        cache_value_l1: torch.Tensor,
        cache_value_l2: torch.Tensor,
        cache_value_l3: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cache = (
            (cache_key_l0, cache_value_l0),
            (cache_key_l1, cache_value_l1),
            (cache_key_l2, cache_value_l2),
            (cache_key_l3, cache_value_l3),
        )
        velocity, updated_cache = self.dit(
            noisy_mel,
            t,
            r,
            cache=None,
            cond=condition,
            spks=speaker_embedding,
            offset=position_offset,
            is_inference=True,
            kv_cache=cache,
        )
        return (
            velocity,
            updated_cache[0][0],
            updated_cache[1][0],
            updated_cache[2][0],
            updated_cache[3][0],
            updated_cache[0][1],
            updated_cache[1][1],
            updated_cache[2][1],
            updated_cache[3][1],
        )


class VocosSpectrumWrapper(nn.Module):
    def __init__(self, vocos: nn.Module) -> None:
        super().__init__()
        self.vocos = vocos

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.vocos.backbone(mel)
        magnitude, phase = self.vocos.head.out(features).transpose(1, 2).chunk(2, dim=1)
        magnitude = torch.exp(magnitude).clamp(max=1e2)
        return magnitude, phase


