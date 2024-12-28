import argparse
from pathlib import Path
import pickle
import json
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm
import IPython.display
import matplotlib.pyplot as plt
from scipy.special import expit
import librosa
import soundfile as sf

from src.spectrogram_transformer import SEGMENT_SHIFT, SEGMENT_LENGTH, AudioSpectrogramTransformer
from src.audioset_utils import SEGMENTS_PER_SPAN, AudioSetOnthology, pad_or_trim_to_len, display_sample, Features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--max_samples', required=False)
    parser.add_argument('--segments_per_span', default=SEGMENTS_PER_SPAN)
    parser.add_argument('--labels_path', default='audioset/labels.json')
    parser.add_argument('--onthology_path', default='audioset/audioset_ontology.json')
    parser.add_argument('--translations_path', default='audioset/ru_translations.json')
    parser.add_argument('-o', '--output_dir', default='audioset')
    parser.add_argument('--save_features_to_dir', required=False)

    args = parser.parse_args()

    correlations_path = Path(args.output_dir) / 'correlations.npy'
    counts_path = Path(args.output_dir) / 'counts.csv'
    summary_table_path = Path(args.output_dir) / 'summary_table.csv'

    labels = json.loads(Path(args.labels_path).read_text())

    print('Loading datasets')
    assert len(args.datasets)
    dataset = concatenate_datasets([
        load_dataset(name, split='train') for name in args.datasets
    ]).with_format('np')
    if args.max_samples is not None:
        dataset = dataset.take(int(args.max_samples))

    print('Collecting audioset logits')
    audioset_logits_list = [sample['ast'].astype(np.float16) for sample in tqdm(dataset)]

    print('Concatenating logits')
    audioset_logits = Features.from_list(
        audioset_logits_list, labels, delta_sec=SEGMENT_SHIFT, length_sec=SEGMENT_LENGTH
    )
    del audioset_logits_list

    print('Calculating average probas per span')
    audioset_span_average_probas = audioset_logits.reduce(
        'mean', sliding_window_size=int(args.segments_per_span), preprocess_fn=expit, pbar=True
    )

    print('Calculating max probas per span')
    audioset_span_max_probas = audioset_logits.reduce(
        'max', sliding_window_size=int(args.segments_per_span), preprocess_fn=expit, pbar=True
    )

    print('Calculating max per sample of average probas per span')
    audioset_span_avgmax_probas = audioset_span_average_probas.reduce('max', pbar=True)

    print('Calculating and saving correlations')
    correlations = np.corrcoef(audioset_span_average_probas.data.T).astype(np.float16)
    with open(correlations_path, 'wb') as f:
        np.save(f, correlations, allow_pickle=False)
    
    print('Calculating and saving counts')
    bins_maxavg = np.linspace(0, 1, num=21, endpoint=True)
    histograms_maxavg = np.stack([
        np.histogram(audioset_span_avgmax_probas.data[:, class_idx], bins=bins_maxavg)[0]
        for class_idx in range(len(labels))
    ])
    pd.DataFrame(histograms_maxavg, columns=[
        f'{start:.2f}-{end:.2f}' for start, end in zip(bins_maxavg[:-1], bins_maxavg[1:])
    ]).to_csv(counts_path, index=False)

    print('Generating summary table')
    onthology = AudioSetOnthology(
        onthology_path=args.onthology_path,
        labels_path=args.labels_path,
        translations_path=args.translations_path,
        label_correlations_path=correlations_path,
        label_counts_path=counts_path,
    )
    onthology.generate_report_df().to_csv(summary_table_path, index=False)

    if args.save_features_to_dir is not None:
        print('Saving features as pickle')
        features_dir = Path(args.save_features_to_dir)
        features_dir.mkdir(exist_ok=True, parents=True)
        (features_dir / 'audioset_logits.pkl').write_bytes(
            pickle.dumps(audioset_logits)
        )
        (features_dir / 'audioset_span_average_probas.pkl').write_bytes(
            pickle.dumps(audioset_span_average_probas)
        )
        (features_dir / 'audioset_span_max_probas.pkl').write_bytes(
            pickle.dumps(audioset_span_max_probas)
        )
        (features_dir / 'audioset_span_avgmax_probas.pkl').write_bytes(
            pickle.dumps(audioset_span_avgmax_probas)
        )