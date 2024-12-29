import io
import argparse
from pathlib import Path
import sys
from typing import Any
import datetime
import time
from pathlib import Path
from itertools import product
import zipfile
import math
import random

from datasets import Dataset
from pydub import AudioSegment
from tqdm.auto import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir')
    parser.add_argument('-o', '--output_dir', default='sova')
    parser.add_argument('--n_shards', default=500)
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: read only 3 archives, use 5 shards and 10 files from each archive'
    )

    args = parser.parse_args()

    archive_paths = list(Path(args.input_dir).glob('*.zip'))

    if args.test:
        archive_paths = archive_paths[:3]

    archives = {
        str(path): zipfile.ZipFile(path, 'r')
        for path in tqdm(archive_paths, desc='opening archives')
    }

    archive_to_files_list = {
        archive_path: set(archive.namelist())
        for archive_path, archive in tqdm(archives.items(), desc='Listing files')
    }

    all_files_from_all_archives = sorted([
        (archive_path, file_path)
        for archive_path, files_list in archive_to_files_list.items()
        for file_path in files_list
        if file_path.endswith('.wav')
    ])

    random.Random(0).shuffle(all_files_from_all_archives)

    print(f'Found {len(all_files_from_all_archives)} files')

    n_shards = int(args.n_shards) if not args.test else 5

    shard_size = math.ceil(len(all_files_from_all_archives) / n_shards)
    print(f'Shard size: {shard_size}')

    for shard_idx in range(n_shards):

        filepath = Path(f'{args.output_dir}/{shard_idx:05d}-of-{n_shards:05d}')
        print(
            f'[{str(datetime.datetime.now())[:-7]}]'
            f' shard {shard_idx}/{n_shards}'
            f' -> {filepath}'
        )

        if filepath.exists():
            print('Already exists')
            continue

        start_time = time.time()

        shard = all_files_from_all_archives[
            shard_idx * shard_size : (shard_idx + 1) * shard_size
        ]

        if args.test:
            shard = shard[:10]

        collected_results = []
        for file_idx, (archive_path, file_path) in enumerate(shard):

            if file_idx % 1000 == 0:
                print(f'{file_idx / len(shard) * 100:.0f}%')
                sys.stdout.flush()

            # reading audio
            bytes = archives[archive_path].read(file_path)
            # audio = AudioSegment.from_file(io.BytesIO(bytes))
            # audio.export(buffer := io.BytesIO(), format='mp3', bitrate=args.bitrate)

            # trying to read transcription
            if (txt_path := file_path[:-4] + '.txt') in archive_to_files_list[archive_path]:
                text = archives[archive_path].read(txt_path).decode('utf-8')
            else:
                text = ''

            collected_results.append({
                'audio': {'bytes': bytes},
                'transcription': text,
                'name': file_path,  # unique across all archives
            })

        resulting_dataset = Dataset.from_list(collected_results)
        # resulting_dataset.to_parquet(tmp_path := filepath.with_stem('tmp'))
        resulting_dataset.save_to_disk(str(tmp_path := filepath.with_stem('tmp')))
        tmp_path.rename(filepath)  # to prevent truncated parquet files

        print(
            f'Elapsed {time.time() - start_time:.0f} sec'
            f', saved {filepath.stat().st_size / 1024 ** 2:.1f} MB'
        )