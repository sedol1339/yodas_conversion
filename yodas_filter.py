import io
import argparse
import datetime
import time
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

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    min_len = float(args.min_len)
    max_len = float(args.max_len)

    for filepath in sorted(input_dir.glob('*.parquet')):
        output_filepath = output_dir / filepath.name
        print(
            f'[{str(datetime.datetime.now())[:-7]}]'
            f' {filepath} -> {output_filepath}'
        )

        if output_filepath.is_file():
            print('Already exists')
            continue

        
        start_time = time.time()

        dataset = (
            load_dataset(str(input_dir), data_files=[filepath.name], split='train')
            .cast_column('audio', Audio(decode=False))
        )

        filtered_dataset = (
            dataset.filter(filter, fn_kwargs={'min_len': min_len, 'max_len': max_len})
        )

        removed_ratio = 1 - len(filtered_dataset) / len(dataset)
        print(f'Filtered out {removed_ratio*100:.2f}% of {len(dataset)} samples')

        filtered_dataset.to_parquet(tmp_path := output_filepath.with_stem('tmp'))
        tmp_path.rename(output_filepath)  # to prevent truncated parquet files