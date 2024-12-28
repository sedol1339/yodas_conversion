from typing import Literal

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, ASTForAudioClassification

SEGMENT_LENGTH = 10
SEGMENT_SHIFT = 5


def split_evenly(
    waveform: np.ndarray,
    segment_length: float = SEGMENT_LENGTH,
    segment_shift: float = SEGMENT_SHIFT,
    sampling_rate: int = 16_000
) -> np.ndarray | None:
    augio_length = len(waveform) / sampling_rate
    if augio_length < segment_length:
        return None
    segment_starts = np.arange(0, augio_length - segment_length, step=segment_shift)
    segments = [
        waveform[int(start_sec * sampling_rate):int((start_sec + segment_length) * sampling_rate)]
        for start_sec in segment_starts
    ]
    assert all([len(s) == len(segments[0]) for s in segments])
    return np.array(segments)


class AudioSpectrogramTransformer:
    def __init__(
        self,
        model_path: str = 'MIT/ast-finetuned-audioset-10-10-0.4593',
        device: str = 'cuda',
    ):
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.model = ASTForAudioClassification.from_pretrained(model_path).to(device)
        self.labels = np.array([
            self.model.config.id2label[id]
            for id in range(self.model.config.num_labels)
        ])
        self.min_frames = 240  # min waveform length to pass into feature_extractor and model
    
    def predict_on_batch(self, waveforms: np.ndarray, sampling_rate: int = 16_000) -> np.ndarray:
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=sampling_rate,
            return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            return self.model(**inputs).logits.detach().cpu().numpy()
    
    def predict_on_long_audio(
        self,
        waveform: np.ndarray,
        batch_size: int = 32,
        segment_length: float = SEGMENT_LENGTH,
        segment_shift: float = SEGMENT_SHIFT,
        sampling_rate: int = 16_000,
        min_length: float = 1,
    ) -> np.ndarray:
        # may ignore a trailing part up to `segment_shift` seconds
        waveforms = split_evenly(
            waveform,
            segment_length=segment_length,
            segment_shift=segment_shift,
            sampling_rate=sampling_rate
        )
        if waveforms is None:  # audio length < segment_length
            if (
                len(waveform) < self.min_frames  # technically can't predict
                or len(waveform) / sampling_rate < min_length  # don't want to predict, too short and considered OOD
            ):
                return np.zeros((0, len(self.labels)))
            waveforms = waveform[None]
        batches = [waveforms[i:i + batch_size] for i in range(0, len(waveforms), batch_size)]
        logits = np.concatenate(
            [self.predict_on_batch(batch, sampling_rate=sampling_rate) for batch in batches],
            axis=0
        )
        return logits

    def plot_top_classes(self, logits: np.ndarray, top_by: Literal['max', 'mean'] = 'max'):
        # logits have shape (n_segments, n_classes)
        reduction = {'max': np.max, 'mean': np.mean}[top_by]
        classes_to_display = np.argsort(reduction(logits, axis=0))[:-10:-1]
        classes_to_display
        for cls_idx in classes_to_display:
            plt.plot(logits[:, cls_idx], label=self.labels[cls_idx])
        plt.legend()
        plt.show()