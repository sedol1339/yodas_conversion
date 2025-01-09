from __future__ import annotations

import copy
import json
import io
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Callable
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d
from tqdm.auto import tqdm


SEGMENTS_PER_SPAN = 23


def sliding_window_mean(
    values: np.ndarray,  # 1D array
    window_size: int,
) -> np.ndarray:
    if len(values) < window_size:
        return np.array([], dtype=values.dtype)
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')


def sliding_window_max(
    values: np.ndarray,  # 1D array
    window_size: int,
):
    if len(values) < window_size:
        return np.array([], dtype=values.dtype)
    hW = (window_size - 1) // 2  # Half window size
    return maximum_filter1d(values, size=window_size)[hW:-hW]


@dataclass
class Features:
    """
    Represents concatenated feature matrices for a dataset of samples with time axis. Each feature
    represents some numerical quantity for a segment over time axis, which has start and end times
    in seconds.

    `labels` (n_features,) - feature names
    `data` (total_steps, n_features) - concatenated feature matrices
    `sample_sizes` (n_samples,) - time steps count for each sample.
        - some samples may have zero time steps
        - `sample_sizes.sum() == total_steps`
    `sample_indexer` (n_samples,) - starting index in `data` for each sample.
        - equals `np.concatenate([[0], np.cumsum(sample_sizes[:-1])])`
    `sample_ids` (total_steps,) - sample index for each step in `data`
        - equals `np.concatenate([i * np.ones(s, dtype=int) for i, s in enumerate(sample_sizes)])`
    `step_ids` (total_steps,) - step index relative to sample for each step in `data`
        - equals `np.concatenate([np.arange(len(s)) for s in sample_sizes])`
    `start_sec` (float) - starting time of the first step in each segment
    `delta_sec` (float) - delta time between start times of the consecutive steps in each segment
    `length_sec` (float) - delta between start and end time of each step in each segment

    This class implements __len__ and __getitem__, so that self[i] is a feature matrix of i-th
    sample. Features object be converted to list of sample feature matrices with list(self).
    """
    data: np.ndarray
    labels: np.ndarray
    sample_sizes: np.ndarray
    sample_indexer: np.ndarray
    sample_ids: np.ndarray
    step_ids: np.ndarray
    start_sec: float | None
    delta_sec: float | None
    length_sec: float | None

    def as_contiguous(self) -> Features:
        # storing in F-order for fast feature-wise operations
        self.data = np.ascontiguousarray(self.data.T).T
        return self

    @classmethod
    def from_list(
        cls,
        features: list[np.ndarray],
        labels: np.ndarray | list[str],
        start_sec: float | None = 0,
        delta_sec: float | None = None,
        length_sec: float | None = None,
    ) -> Features:
        sample_sizes = np.array([len(matrix) for matrix in features])
        data = np.concatenate(features, axis=0)
        return Features(
            data=data,
            labels=np.array(labels),
            sample_sizes=sample_sizes,
            sample_indexer=np.concatenate([[0], np.cumsum(sample_sizes[:-1])]),
            sample_ids=np.concatenate([i * np.ones(s, dtype=int) for i, s in enumerate(sample_sizes)]),
            step_ids=np.concatenate([np.arange(s) for s in sample_sizes]),
            start_sec=start_sec,
            delta_sec=delta_sec,
            length_sec=length_sec,
        )
    
    def get_sample(self, i: int) -> np.ndarray:
        start_idx = self.sample_indexer[i]
        return self.data[start_idx:start_idx + self.sample_sizes[i]]
    
    def __len__(self) -> int:
        return len(self.sample_sizes)
    
    def __getitem__(self, i: int) -> np.ndarray:
        return self.get_sample(i)
    
    @property
    def has_timings(self):
        return (
            self.delta_sec is not None
            and self.length_sec is not None
            and self.start_sec is not None
        )
    
    def get_step_timings(self, step_idx: int) -> tuple[float, float]:
        start_time = self.start_sec + self.delta_sec * step_idx
        return start_time, start_time + self.length_sec

    def __repr__(self) -> str:
        return (
            f'{len(self.sample_sizes)} samples'
            f', shape {(self.data.shape)}'
            f', {self.data.dtype}'
            f', {self.data.nbytes / 1024**2:.0f} MB'
            f', start {self.start_sec} sec'
            f', delta {self.delta_sec} sec'
            f', length {self.length_sec} sec'
        )
    
    def reduce(
        self,
        reduction: Literal['mean', 'max'] | Callable,
        sliding_window_size: int | None = None,
        #sliding_window_size_sec: float | None = None,
        preprocess_fn: Callable | None = None,
        pbar: bool = False,
        pbar_desc: str = 'Reducing features',
    ) -> Features:
        if sliding_window_size is not None:
            assert isinstance(reduction, str)
            reduction_fn = {
                'mean': partial(sliding_window_mean, window_size=sliding_window_size),
                'max': partial(sliding_window_max, window_size=sliding_window_size),
            }[reduction]
        elif isinstance(reduction, str):
            reduction_fn = {
                'mean': np.mean,
                'max': np.max,
            }[reduction]
        else:
            reduction_fn = reduction
        
        preprocess_fn = preprocess_fn or (lambda x: x)

        reduced_features_list: list[np.ndarray] = []
        for arr in tqdm(list(self), disable=not pbar, desc=pbar_desc):
            if len(arr) == 0:
                reduced = np.zeros((0, arr.shape[1]), dtype=arr.dtype)
            else:
                reduced = np.apply_along_axis(reduction_fn, axis=0, arr=preprocess_fn(arr))
                reduced = reduced.astype(arr.dtype)
                if reduced.ndim == 1:  # reduction_fn return a number, not an array
                    reduced = reduced[None, :]
            reduced_features_list.append(reduced)

        if sliding_window_size is not None and self.has_timings:
            new_timings = {
                'start_sec': self.start_sec,
                'delta_sec': self.delta_sec,
                'length_sec': (
                    self.length_sec + self.delta_sec * (sliding_window_size - 1)
                    if self.has_timings
                    else None
                ),
            }
        else:
            new_timings = {}
        
        return Features.from_list(
            features=reduced_features_list,
            labels=self.labels,
            **new_timings,
        )

    def sparsify(self, times: int) -> Features:
        return Features.from_list(
            features=[arr[::times] for arr in list(self)],
            labels=self.labels,
            start_sec=self.start_sec,
            length_sec=self.length_sec,
            delta_sec=(
                self.delta_sec * times
                if self.delta_sec is not None
                else None
            ),
        )

    def select_feature_ids(self, ids: list[int]) -> Features:
        result = copy.copy(self)
        result.data = result.data[:, ids]
        result.labels = result.labels[ids]
        return result
    
    def transform(
        self,
        fn: Callable,
        new_labels: list[str] | None = None,
    ) -> Features:
        result = copy.copy(self)
        result.data = fn(result.data)
        if new_labels is not None:
            result.labels = new_labels
        return result


class AudioSetOnthology:

    def __init__(
        self,
        onthology_path: Path | str = 'audioset/audioset_ontology.json',
        labels_path: Path | str = 'audioset/labels.json',
        translations_path: Path | str | None = 'audioset/ru_translations.json',
        label_correlations_path: Path | str | None = 'audioset/correlations.npy',
        label_counts_path: Path | str | None = 'audioset/counts.csv',
    ):
        self.labels: list[str] = json.loads(Path(labels_path).read_text())
        self.translations: dict[str, str] = (
            json.loads(translations_path.read_text())
            if (translations_path := Path(translations_path)).is_file()
            else {}
        )

        # building a mapping class name -> all class data (names are unique)
        self.ontology = {
            x['name']: x
            for x in json.loads(Path(onthology_path).read_text())
        }

        # building a mapping parent name -> immediate children name
        self.immediate_children = {
            name: [
                name2 for name2, data2 in self.ontology.items()
                if data2['id'] in data['child_ids']
            ]
            for name, data in self.ontology.items()
        }

        # building a mapping parent name -> distant children name (children of children etc.)
        # the ontology is not tree-like, there may be several paths to the same child
        children_levels = [self.immediate_children]
        while True:
            if len(sum(children_levels[-1].values(), [])) == 0:
                break
            children_levels.append({
                name: sum([self.immediate_children[c] for c in children], [])
                for name, children in children_levels[-1].items()
            })
        self.all_children = {
            name: list(set.union(*[set(level[name]) for level in children_levels]))
            for name in self.immediate_children
        }

        # keeping only children that exist in labels
        self.all_existent_children = {
            name: [c for c in children if c in self.labels]
            for name, children in self.all_children.items()
        }

        # getting a list of abstract classes
        self.abstract_labels = [
            name for name, children in self.all_existent_children.items()
            if name not in self.labels and len(children)
        ]

        # list of parents for each child (may be multiple inheritance)
        self.parents: dict[str, list[str]] = defaultdict(list)
        for parent, children in self.immediate_children.items():
            for child in children:
                self.parents[child].append(parent)
        self.root_nodes = set(self.immediate_children) - set(self.parents)
        for root_node in self.root_nodes:
            self.parents[root_node].append(None)

        self.correlations = (
            np.load(io.BytesIO(label_correlations_path.read_bytes()), allow_pickle=False)
            if (label_correlations_path := Path(label_correlations_path)).is_file()
            else None
        )

        self.counts = (
            pd.read_csv(label_counts_path).values
            if (label_counts_path := Path(label_counts_path)).is_file()
            else None
        )

    def get_class_name_tokens(self, child_name: str) -> list[list[str]]:
        """
        Example:
        ```
        get_full_child_name('Choir')
        >>> [['Music', 'Musical instrument', 'Choir'],
            ['Human sounds', 'Human voice', 'Singing', 'Choir']]
        ```
        """
        full_names: list[list[str]] = []
        partial_names: list[list[str]] = [[child_name]]
        while len(partial_names):
            partial_name = partial_names.pop(0)
            for parent in self.parents[partial_name[0]]:
                if parent is None:
                    full_names.append(partial_name)
                else:
                    partial_names.append([parent] + partial_name)
        return full_names
    
    def with_subclasses(self, class_name: str) -> list[str]:
        """
        Given a class name X, return a list of X and all its existent
        subclasses of any level.
        """
        return [class_name] + self.all_existent_children[class_name]
    
    def generate_report_df(self) -> pd.DataFrame:
        def _generate_report_rows(class_name_tokens: list[str]) -> list[dict]:
            class_name = class_name_tokens[-1]
            level = len(class_name_tokens)
            report = {
                f'level{level}': class_name,
                'abstract': class_name not in self.labels,
                'description': self.ontology[class_name]['description'],
            }
            
            report['also'] = ' and '.join([
                '"' + '/'.join(tokens) + '"'
                for tokens in self.get_class_name_tokens(class_name_tokens[-1])
                if tokens != class_name_tokens
            ])

            if class_name in self.labels:
                report['ru'] = self.translations.get(class_name, None)

                class_idx = self.labels.index(class_name)

                if self.counts is not None:
                    report['counts'] = ','.join([str(x) for x in self.counts[class_idx][1:]])
                
                if self.correlations is not None:
                    report['correlated'] = json.dumps({
                        self.labels[idx]: f'{corr:.2f}'
                        for idx, corr in self.get_companions(class_idx)
                        if not self.labels[idx] in self.all_existent_children[class_name]
                        and not self.labels[idx] in class_name_tokens[:-1]
                    })
                    report['anti-correlated'] = json.dumps({
                        self.labels[idx]: f'{corr:.2f}'
                        for idx, corr in self.get_antagonists(class_idx)
                    })

            report_rows = [report]
            for child in self.immediate_children[class_name]:
                if (child in self.labels) or (child in self.abstract_labels):
                    report_rows += _generate_report_rows(class_name_tokens + [child])
            
            return report_rows

        onthology_description_rows = sum([
            _generate_report_rows([class_name])
            for class_name in self.root_nodes
        ], [])
        
        return pd.DataFrame(onthology_description_rows)[[
            'level1',
            'level2',
            'level3',
            'level4',
            'level5',
            'level6',
            'ru',
            'abstract',
            'counts',
            'also',
            'correlated',
            'anti-correlated',
            'description',
        ]]

    def get_companions(self, class_idx: int, threshold: float = 0.3) -> list[tuple[int, float]]:
        assert self.correlations is not None
        return [
            (class_idx2, corr)
            for class_idx2 in np.argsort(self.correlations[class_idx])[::-1]
            if (corr := self.correlations[class_idx, class_idx2]) > threshold
            and class_idx != class_idx2
        ]

    def get_antagonists(self, class_idx: int, threshold: float = 0.3) -> list[tuple[int, float]]:
        assert self.correlations is not None
        return [
            (class_idx2, corr)
            for class_idx2 in np.argsort(self.correlations[class_idx])
            if (corr := self.correlations[class_idx, class_idx2]) < -threshold
            and class_idx != class_idx2
        ]
    
    def get_class_sets(self) -> dict[str, list[int]]:
        onthology = self

        class_sets = {
            'speech': onthology.with_subclasses('Speech'),
            'child': ['Child speech, kid speaking'],
            'whispering': ['Whispering'],
            'conversation': ['Conversation'],
            'music': [
                *onthology.with_subclasses('Music'),
                *onthology.with_subclasses('Sine wave'),
                *onthology.with_subclasses('Sound effect'),
                'Humming',
            ],
            'warnings': [
                'Babbling',
                *onthology.with_subclasses('Shout'),
                *onthology.with_subclasses('Laughter'),
                'Screaming',
                'Crying, sobbing',
                'Wail, moan',
                'Groan',
                'Whistling',
                'Cough',
                'Sneeze',
            ],
            'acoustic': [
                *onthology.with_subclasses('Natural sounds'),
                *onthology.with_subclasses('Noise'),
                *onthology.with_subclasses('Sound reproduction'),
                *onthology.with_subclasses('Animal'),
                *onthology.with_subclasses('Acoustic environment'),
                *onthology.with_subclasses('Sounds of things'),
                *onthology.with_subclasses('Source-ambiguous sounds'),  # inclucing Silence
                *onthology.with_subclasses('Human locomotion'),
                *onthology.with_subclasses('Digestive'),
                *onthology.with_subclasses('Hands'),
                *onthology.with_subclasses('Heart sounds, heartbeat'),
                *onthology.with_subclasses('Breathing'),
                'Cheering',
                'Applause',
                'Crowd',
                'Hubbub, speech noise, speech babble',
                'Children playing',
                'Grunt',
                'Sigh',
                'Sniff',
                'Chatter',
            ],
        }

        for name, labels in class_sets.items():
            class_sets[name] = [self.labels.index(l) for l in labels if l in self.labels]

        return class_sets


def pad_or_trim_to_len(arr: np.ndarray, length: int, value: float = np.nan):
    """
    Pad or trim 1D array to the specified length. Pad with the provided value.
    """
    pad_size = length - len(arr)
    if pad_size > 0:
        return np.concatenate([arr, np.full(pad_size, value)])
    else:
        return arr[:length]
    

def display_sample(
    values: np.ndarray,
    labels: list[str],
    num_to_display: int = 10,
    sec_per_tick: int = None,
    highlight_classes: list[int] | None = None,
    ax: plt.Axes | None = None,
):
    assert len(labels) == values.shape[1]
    if ax is None:
        plt.figure(figsize=(15, 4))
        ax = plt.gca()
    classes_to_display = np.argsort(values.max(axis=0))[:-num_to_display:-1]
    if len(values) > 0:
        for class_idx in classes_to_display:
            x_values = np.arange(len(values)) * (sec_per_tick or 1)
            ax.plot(
                x_values,
                values[:, class_idx],
                lw=0.9 if not class_idx in (highlight_classes or []) else 2,
                label=labels[class_idx]
            )
        ax.legend()
    ax.set_xlabel('Seconds' if sec_per_tick is not None else 'Ticks')