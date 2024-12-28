import io
import argparse

import soundfile as sf
from datasets import load_dataset, Audio
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dataset')
    parser.add_argument('-n', '--n_samples', default=200_000)
    parser.add_argument('-o', '--output_figure', default='length_histogram.png')
    args = parser.parse_args()

    dataset_bytes = load_dataset(args.input_dataset, split='train').cast_column('audio', Audio(decode=False))

    lengths = []
    for sample in tqdm(dataset_bytes.take(int(args.n_samples))):
        try:
            waveform, rate = sf.read(io.BytesIO(sample['audio']['bytes']))
            lengths.append(len(waveform) / rate)
        except sf.LibsndfileError as e:
            pass

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax2 = ax.twinx()

    bins = np.logspace(-2, np.log10(30), num=500)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    hist, _ = np.histogram(lengths, bins=bins)

    ax.hist(lengths, bins=bins, log=True, color='cornflowerblue')
    ax2.plot(
        bin_centers,
        np.cumsum(hist) / np.sum(hist),
        color='C1',
        label='cumulative ratio by samples'
    )
    ax2.plot(
        bin_centers,
        np.cumsum(hist * bin_centers) / np.sum(hist * bin_centers),
        color='red',
        label='cumulative ratio by seconds'
    )

    plt.xscale('log')
    plt.title('Audio length histogram and CDF')

    ax.set_ylabel('n samples')
    ax2.set_ylabel('ratio')
    plt.legend()

    ax2.set_yticks(np.linspace(0, 1, num=21))
    ax2.grid(which='both', axis='y')
    ax.grid(which='both', axis='x')

    plt.savefig(args.output_figure)
    plt.close()