import copy
from dataclasses import asdict, dataclass
import os

import pandas as pd

import torch
import numpy as np


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
        # this will write in stdout
        from pyannote.audio.pipelines import SpeakerDiarization

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
            'waveform': torch.tensor(waveform, dtype=torch.float32)[None],
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
    
def reorder_speakers(
    segments: list[Segment],
    speaker_embeddings: np.ndarray,
) -> tuple[list[Segment], np.ndarray]:
    """
    Assigns index 0 to the speaker with the longest total speech duration,
    then index 1, and so on.
    1) Returns same segments with new speaker ids (not inplace).
    2) Reorders 'speaker_embeddings' field accordingly.
    """
    if len(segments) == 0:
        return segments, speaker_embeddings
    
    speaker_durations = (  # <-- old speaker ids here
        pd.DataFrame(segments)
        .set_index('speaker_idx')
        .assign(duration=lambda df: df.end - df.start)
        .groupby('speaker_idx')['duration']
        .sum()  # to series, total speech time for each speaker_idx
        .sort_index()
    )
    speakers_new_ids = (
        speaker_durations
        .rank(method='first', ascending=False)
        .astype(int)
        .values  # to numpy array
        - 1  # .rank() enumerates from 1, we need to enumerate from 0
    )
    segments = [
        Segment(**asdict(s)) for s in segments
    ]
    for s in segments:
        s.speaker_idx = speakers_new_ids[s.speaker_idx]
    
    reordering = np.argsort(speakers_new_ids)
    speaker_embeddings = speaker_embeddings[reordering]

    return segments, speaker_embeddings


def test_reorder_speakers():
    """
    speaker 0: 5 sec -> new index 2
    speaker 1: 1 sec -> new index 4
    speaker 2: 7 sec -> new index 1
    speaker 3: 20 sec -> new index 0
    speaker 4: 2 sec -> new index 3
    """
    segments = [
            Segment(start=8, end=9, speaker_idx=0),
            Segment(start=3, end=5, speaker_idx=2),
            Segment(start=15, end=16, speaker_idx=1),
            Segment(start=8, end=12, speaker_idx=2),
            Segment(start=1, end=5, speaker_idx=0),
            Segment(start=20, end=40, speaker_idx=3),
            Segment(start=41, end=43, speaker_idx=4),
            Segment(start=6, end=7, speaker_idx=2),
    ]
    speaker_embeddings = np.array([[2], [1], [2], [9], [-10]])
    segments, speaker_embeddings = reorder_speakers(segments, speaker_embeddings)
    assert [s.speaker_idx for s in segments] == [2, 1, 4, 1, 2, 0, 3, 1]
    assert (speaker_embeddings == np.array([[9], [2], [2], [-10], [1]])).all()


# test_reorder_speakers()


def get_speech_mask(
    segments: list[Segment],
    duration: float | None = None,
    tick_len: float = 1 / 100,
    for_each_speaker: bool = False,
) -> np.ndarray:
    """Returns boolean array of shape (n_speakers, n_ticks) - True if speaker speaks"""

    duration = duration if duration is not None else max([s.end for s in segments])
    total_ticks = np.ceil(duration / tick_len).astype(int)

    if len(segments) == 0:
        if for_each_speaker:
            return np.full((0, total_ticks), False)
        else:
            return np.full(total_ticks, False)

    max_speaker_idx = max([s.speaker_idx for s in segments])

    arr = np.full((max_speaker_idx + 1, total_ticks), False)
    for segment in segments:
        start_tick = round(segment.start / tick_len)
        end_tick = 1 + round(segment.end / tick_len)
        arr[segment.speaker_idx, start_tick:end_tick] = True

    if for_each_speaker:
        return arr
    else:
        return arr.max(axis=0)