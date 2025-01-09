from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Callable, Literal

import numpy as np
from datasets import Dataset
from scipy.special import expit

from .audioset_utils import Features, AudioSetOnthology
from .spectrogram_transformer import SEGMENT_SHIFT, SEGMENT_LENGTH
from .speaker_diarization import Segment, get_speech_mask, reorder_speakers


@dataclass
class YodasSearchResult:
    pass

class YodasSearch:
    """
    A class to provide results for search queries over the YODAS2 dataset. 
    """
    def __init__(
        self,
        dataset: Dataset,
        onthology: AudioSetOnthology,
        cache_file: Path | str,
        rewrite_cache: bool = False,
    ):
        """
        Arguments:
        - dataset: a YODAS2 dataset
        - cache_file: cache file for the calculated features
        """
        self.dataset = dataset
        self.onthology = onthology
        
        self.sliding_window_size = round(1 + (120 - SEGMENT_LENGTH) / SEGMENT_SHIFT)
        self.spans_sparsity_rate = 4
        
        if (cache_file := Path(cache_file)).is_file() and not rewrite_cache:
            self._features = pickle.loads(cache_file.read_bytes())
        else:
            self._features = self._calculate_features_for_dataset()
            cache_file.write_bytes(pickle.dumps(self._features))
    
    def search(
        self,
        selected_video_ids: list[str] | None = None,
    ) -> list[YodasSearchResult]:
        pass
    
    def _calculate_features_init(self):
        self.class_sets = self.onthology.get_class_sets()
    
    def _calculate_features_for_sample(
        self,
        sample: list[str, Any],
    ) -> list[str, Any]:
        features = {}

        # TODO is video_id unique over all yodas2 parts?
        features['video_id'] = sample['video_id']
        features['duration'] = sample['duration']
        features['text'] = ' '.join(sample['utterances']['text'])
        
        features['ast_probas'] = expit(np.array(sample['ast'])).astype(np.float16)
        
        tick_per_sec = 10
        segments, speaker_embeddings = reorder_speakers(
            segments=[Segment(**x) for x in sample['segments']],
            speaker_embeddings=np.array(sample['speaker_embeddings']),
        )
        if len(speaker_embeddings) == 0:
            speaker_embeddings = None

        speakers_masks = get_speech_mask(
            segments=segments,
            duration=sample['duration'],
            tick_len=1 / tick_per_sec,
            for_each_speaker=True
        )

        step_size = SEGMENT_SHIFT * self.spans_sparsity_rate * tick_per_sec
        window_size = self.sliding_window_size * SEGMENT_SHIFT * tick_per_sec

        # to ensure that AST features (after calculating span mean/max and sparsifying)
        # and diarization features are of equal length
        n_steps = max(0, len(features['ast_probas'][
            : - self.sliding_window_size + 1 : self.spans_sparsity_rate
        ]))

        features['speech_seconds'] = []
        features['voice_overlap_seconds'] = []
        features['top_speaker_ratio'] = []
        features['top_speaker_idx'] = []

        for step_idx in range(n_steps):
            # working with speakers mask for spans of length {window_size} seconds
            start_idx = step_idx * step_size
            end_idx = start_idx + window_size
            mask = speakers_masks[:, start_idx:end_idx]

            features['speech_seconds'].append((mask.sum(axis=0) > 0).sum() / tick_per_sec)
            features['voice_overlap_seconds'].append((mask.sum(axis=0) > 1).sum() / tick_per_sec)
            
            speaker_durations = mask.sum(axis=1)
            if len(speaker_durations) > 0 and sum(speaker_durations) > 0:
                top_speaker_idx = speaker_durations.argmax()
                features['top_speaker_idx'].append(top_speaker_idx)
                features['top_speaker_ratio'].append(
                    speaker_durations[top_speaker_idx] / sum(speaker_durations)
                )
            else:
                features['top_speaker_idx'].append(0)
                features['top_speaker_ratio'].append(1.0)
        
        features['speaker_embeddings'] = speaker_embeddings

        return features

    def _calculate_features_for_dataset(self) -> list[str, np.ndarray | Features]:
        self._calculate_features_init()

        features_list = self.dataset.map(
            self._calculate_features_for_sample,
            remove_columns=list(self.dataset.features),
            keep_in_memory=True,
            desc='Extracting search features'
        )

        features = {}
        features['text'] = np.array(features_list['text'], dtype=object)
        features['video_id'] = features_list.with_format('np')['video_id']
        features['duration'] = features_list.with_format('np')['duration']
        features['speaker_embeddings'] = features_list.with_format('np')['speaker_embeddings']

        ast_probas_list = features_list.with_format('np')['ast_probas']

        ast_probas = Features.from_list(
            features=ast_probas_list,
            labels=np.array(self.onthology.labels),
            delta_sec=SEGMENT_SHIFT,
            length_sec=SEGMENT_LENGTH,
        )

        features['is_music'] = (
            ast_probas
            .select_feature_ids(self.class_sets['music'])
            .reduce(
                reduction='max',
                sliding_window_size=self.sliding_window_size,
                pbar=True,
            )
            .sparsify(times=self.spans_sparsity_rate)
            .transform(lambda p: p > 0.3)
            .as_contiguous()
        )

        features['acoustic'] = (
            ast_probas
            .select_feature_ids(self.class_sets['acoustic'])
            .reduce(
                reduction='mean',
                sliding_window_size=self.sliding_window_size,
                pbar=True,
            )
            .sparsify(times=self.spans_sparsity_rate)
            .transform(lambda p: (p > 0.1) | (p > np.quantile(p, 0.999, axis=0, keepdims=True)))
            .as_contiguous()
        )

        features['warnings'] = (
            ast_probas
            .select_feature_ids(self.class_sets['warnings'] + [self.onthology.labels.index('Music')])
            .reduce(
                reduction='max',
                sliding_window_size=self.sliding_window_size,
                pbar=True,
            )
            .sparsify(times=self.spans_sparsity_rate)
            .transform(lambda p: (p > 0.1) | (p > np.quantile(p, 0.99, axis=0, keepdims=True)))
            .as_contiguous()
        )

        features['for_manual_search'] = (
            ast_probas
            .select_feature_ids([
                self.onthology.labels.index(x)
                for x in ['Child speech, kid speaking', 'Whispering', 'Conversation']
            ])
            .reduce(
                reduction='mean',
                sliding_window_size=self.sliding_window_size,
                pbar=True,
            )
            .sparsify(times=self.spans_sparsity_rate)
            .as_contiguous()
        )

        for name in [
            'speech_seconds',
            'voice_overlap_seconds',
            'top_speaker_ratio',
            'top_speaker_idx'
        ]:
            features[name] = Features.from_list(
                features=features_list.with_format('np')[name],
                labels=[name],
                delta_sec=self.spans_sparsity_rate * SEGMENT_SHIFT,
                length_sec=self.sliding_window_size * SEGMENT_SHIFT,
            ).as_contiguous()
        
        return features



'''
Features:

Filter out:
    - In principle, many sounds can be transcribed by the model, for example "laughter", "hahaha", "[MUSIC]", "Cough" etc.
    - A precisely-located ones can be annotated with a permissive annotation "{?}", however this is undesirable
    - So this is better to skip some sound types like "Shout" that are likely to be transcribed
    - Also, we may write a specific rules for a model by analyzing distrepancies when calculating WER
    - We still want to exclude music, since it is usually not precisely-located, and so cannot be annotated as {?}
    max_t[Music/*, "Sine wave"/*, "Sound effect", Humming] > T_1
    avg_t[Speech/*] < T_2

Warning:
    - Do not exclude but warn with timings
    max_t[Babbling, Shout/*, Laughter/*, Screaming, "Crying, sobbing", "Wail, moan", Groan, Whistling, Cough, Sneeze] > T_1

Undesirable but AST predictions are unreliable:
    Speech synthesizer

To promote diversity (output ordering):
    acoustic_embedding = avg_t[
        "Sounds of things"/*, "Source-ambiguous sounds"/* (inclucing Silence), "Human locomotion"/*, Digestive/*, Hands/*,
        "Heart sounds, heartbeat"/*, Cheering, Applause, Crowd, "Hubbub, speech noise, speech babble", "Children playing",
        Grunt, Sigh, Breathing/*, Sniff, Chatter, "Natural sounds"/*, Noise/*, "Sound reproduction"/*,
        Animal/*, "Acoustic environment"/*
    ]
    - We do not exclude music from these sets, since music filtering is another stage using max_t and not avg_t
    - We do not include warning classes in this set
    - Outputs will be ordered by their acoustic distance from already collected audios, and labeled with top acoustic classes
    - TODO how to search for the most non-similar embeddings, considering acoustic_embedding covariation matrix

For additional manual search queries for underrepresented recording types:
    avg_t["Child speech, kid speaking"]
    avg_t[Whispering]
    avg_t[Conversation]

Currently unused:
    "Male speech, man speaking", "Female speech, woman speaking" (we better use speaker embeddings)

Diarization queries/output:
    top_speaker_ratio
    voice_overlap_seconds

Diarization output:
    - top_speaker_embedding

AudioSet classes can also be used for subset analysis of a larger dataset, such as CommonVoice or Golos
'''

# class YodasSearcher:
#     def __init__(
#         self,
#         dataset: Dataset,
#         cache_file: str | Path,
#     ):
#         self.dataset = dataset

#         cache_file = Path(cache_file)
#         cache_file.parent.mkdir(exist_ok=True, parents=True)

#         if cache_file.is_file():
#             features = pickle.loads(cache_file.read_bytes())
#         else:
#             features = self.generate_features()
#             cache_file.write_bytes(pickle.dumps(features))

#         self.labels = AudioSpectrogramTransformer().labels
        
#         self.features_df = pd.DataFrame(features)
    
#     def get_features(self, sample: dict[str, Any]) -> dict[str, Any]:
#         """Get searchable features for a sample"""
#         sample = sample.copy()
#         reorder_speakers_inplace(sample)

#         features = {}
#         features['text'] = ' '.join(sample['utterances']['text'])
#         features['duration'] = sample['duration']
#         features['n_speakers'] = len(sample['speaker_durations'])

#         speaker_durations = np.array(sample['speaker_durations'])
#         speaker_ratios = speaker_durations / speaker_durations.sum()   # may have length 0
#         speaker_durations = np.concatenate([speaker_durations, np.full(10, np.nan)])
#         speaker_ratios = np.concatenate([speaker_ratios, np.full(10, np.nan)])
#         speaker_mask = get_speakers_mask(sample, tick_len=(tick_len := 1 / 100))

#         for i in range(3):
#             features[f'speaker{i}_duration'] = speaker_durations[i]
#             features[f'speaker{i}_ratio'] = speaker_ratios[i]
#         # simultaneous and total speech duration
#         features['sim_speech_duration'] = (speaker_mask.sum(axis=0) > 1).sum() * tick_len
#         features['speech_duration'] = (
#             speaker_mask.max(axis=0).sum() * tick_len
#             if len(speaker_mask) > 0 else 0
#         )

#         # using sigmoid for multilabel classification (not softmax),
#         # according to the AST paper https://arxiv.org/abs/2104.01778
#         ast_log_proba = expit(np.array(sample['ast'], dtype=np.float16))

#         return features
    
#     def generate_features(self) -> list[dict[str, bool | str | int | float]]:
#         """Returns dict of searchable features (with "video_id" field) for each sample in self.dataset"""
#         return [
#             {'video_id': sample['video_id'], **self.get_features(sample)}
#             for sample in self.dataset
#         ]
    
#     def search(
#         self,
#         text_query: str = '',
#         mode: Literal['and', 'or'] = 'and',
#         features_query: str = '',
#     ) -> pd.DataFrame:
#         if text_query != '':
#             mask_found_by_text_query = self.features_df['text'].str.contains(text_query, case=False, regex=False)
#             loc_found_by_text_query = self.features_df.index[mask_found_by_text_query].tolist()
#         else:
#             loc_found_by_text_query = []
        
#         if features_query != '':
#             loc_found_by_feature_query = self.features_df.query(features_query).index.tolist()
#         else:
#             loc_found_by_feature_query = []
        
#         fn = {'and': set.intersection, 'or': set.union}[mode]
#         loc_found = list(fn(
#             set(loc_found_by_text_query),
#             set(loc_found_by_feature_query)
#         ))

#         return self.features_df.loc[loc_found]

#     def display(
#         self,
#         df: pd.DataFrame,
#     ):
#         df = df.assign(text=lambda df: df.text.str.slice(0, 200)) #.str.wrap(50)
#         style = df.style.format(precision=1)

#         width = {'video_id': 100, 'text': 300}

#         for col_name in df.columns:
#             style.set_table_styles({
#                 col_name: [
#                     {
#                         'selector': 'td, th',
#                         'props': [
#                             ('text-align', 'center'),
#                             ('padding', '0em 0em'),
#                             ('border', '1px solid grey !important'),
#                         ],
#                     },
#                     {
#                         'selector': 'td',
#                         'props': [
#                             ('max-width', f'{width.get(col_name, 40)}px'),
#                             ('width', f'{width.get(col_name, 40)}px'),
#                         ],
#                     },
#                     {
#                         'selector': 'th',
#                         'props': [
#                             # ('max-width', f'{width.get(col_name, 40)}px'),
#                             ('max-width', '0px'),
#                             ('height', '170px'),
#                             ('transform', 'translateY(4em) rotate(-90deg);'),
#                         ],
#                     },
#                 ]
#             }, overwrite=False)

#         IPython.display.display(style)

# seacher = YodasSearcher(dataset, 'cache.pkl')
# result = seacher.search('привет', 'and', 'duration < 500')
# seacher.display(result)