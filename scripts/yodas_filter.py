import io
import argparse
import shutil
from typing import Any
from pathlib import Path

import soundfile as sf
from datasets import load_dataset, Audio

def filter(
    sample: dict[str, Any],
    min_len: float = 1,
    max_len: float = 30,
) -> bool:
    try:
        waveform, rate = sf.read(io.BytesIO(sample['audio']['bytes']))
        assert rate == 16_000
        length_sec = len(waveform) / rate
        if length_sec < min_len or length_sec > max_len:
            return False
    except sf.LibsndfileError:
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir')
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('--min_len', default=1)
    parser.add_argument('--max_len', default=30)
    parser.add_argument('--no_shuffle', action='store_true')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    min_len = float(args.min_len)
    max_len = float(args.max_len)

    shutil.rmtree(output_dir, ignore_errors=True)

    dataset = (
        load_dataset(str(input_dir), split='train')
        .cast_column('audio', Audio(decode=False))
    )

    filtered_dataset = (
        dataset.filter(filter, fn_kwargs={'min_len': min_len, 'max_len': max_len})
    )

    if not args.no_shuffle:
        filtered_dataset = filtered_dataset.shuffle(seed=0)

    removed_ratio = 1 - len(filtered_dataset) / len(dataset)
    print(f'Filtered out {removed_ratio*100:.2f}% of {len(dataset)} samples')

    n_output_shards = int(len(filtered_dataset) / 20_000) + 1

    for shard_idx in range(n_output_shards):
        output_path = f'{output_dir}/{shard_idx:05d}-of-{n_output_shards:05d}.parquet'
        print(f'Writing {output_path}')
        filtered_dataset.shard(num_shards=n_output_shards, index=shard_idx).to_parquet(output_path)