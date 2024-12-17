from dataclasses import dataclass
import os

import torch
import numpy as np

from pyannote.audio.pipelines import SpeakerDiarization


@dataclass
class Segment:
    start: float
    end: float
    speaker_idx: int


@dataclass
class DiarizationResults:
    segments: list[Segment]
    speaker_embeddings: np.ndarray


class SpeakerDiarizationWrapper:
    def __init__(
        self,
        device: str = 'cuda',
        segmentation_batch_size: int = 256,
        embedding_batch_size: int = 128,
    ):
        self.pipeline = SpeakerDiarization(
            segmentation='pyannote/segmentation-3.0',
            segmentation_batch_size=segmentation_batch_size,
            embedding='speechbrain/spkrec-ecapa-voxceleb', # 'pyannote/wespeaker-voxceleb-resnet34-LM', <-- buggy?
            embedding_batch_size=embedding_batch_size,
            clustering='AgglomerativeClustering',
            embedding_exclude_overlap=True,
            use_auth_token=os.getenv('HF_TOKEN'),
        ).instantiate({
            'clustering': {
                'method': 'centroid',
                'min_cluster_size': 12,
                'threshold': 0.7045654963945799,
            },
            'segmentation': {
                'min_duration_off': 0.0,
            },
        }).to(torch.device(device))
    
    def predict_on_long_audio(
        self,
        waveform: np.ndarray,
        sampling_rate: int = 16_000
    ) -> DiarizationResults:
        annotation, speaker_embeddings = self.pipeline.apply({
            'waveform': torch.tensor(waveform)[None],
            'sample_rate': sampling_rate,
        }, return_embeddings=True)
        segments = [
            Segment(
                start=float(turn.start),
                end=float(turn.end),
                speaker_idx=int(annotation.labels().index(speaker)),
            )
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        return DiarizationResults(
            segments=segments,
            speaker_embeddings=speaker_embeddings.astype(np.float16)
        )