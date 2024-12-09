import io
import argparse
from pathlib import Path
import socket
from urllib3.connection import HTTPConnection
from typing import Any
import datetime
import time
import shutil

import datasets
from datasets import Audio, Dataset, load_dataset, IterableDataset
from pydub import AudioSegment
from tqdm.auto import tqdm


def map_to_mp3(sample: dict[str, Any], bitrate: str = '32k') -> dict[str, Any]:
    audio = AudioSegment.from_file(io.BytesIO(sample['audio']['bytes']))
    audio.export(buffer := io.BytesIO(), format='mp3', bitrate=bitrate)
    return {'audio': {'bytes': buffer.read()}}


if __name__ == '__main__':
    # fixing Huggingface "read timeout" issue
    # https://github.com/huggingface/transformers/issues/12575#issuecomment-1716743090
    HTTPConnection.default_socket_options = ( 
        HTTPConnection.default_socket_options + [
        (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000), 
        (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
        ])

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dataset', default='espnet/yodas')
    parser.add_argument('-n', '--input_name', default='ru000')
    parser.add_argument('-r', '--bitrate', default='32k')
    parser.add_argument('-o', '--output_dir', required=False)
    parser.add_argument('-s', '--audio_separately', action='store_true')
    parser.add_argument('-f', '--flush', action='store_true')

    args = parser.parse_args()

    output_dir = (
        args.output_dir
        or f'{args.input_dataset.split("/")[-1]}_{args.input_name}_{args.bitrate}'
    )

    if args.flush:
        # fixing a weird bug "No such file or directory ... 00000000.txt"
        # originating probably from the YODAS code
        shutil.rmtree(
            Path(datasets.config.HF_CACHE_HOME)
            / 'modules/datasets_modules/datasets/espnet--yodas',
            ignore_errors=True,
        )

    print(f'Loading {args.input_dataset} {args.input_name}')
    orig_dataset: IterableDataset = load_dataset(
        args.input_dataset,
        name=args.input_name,
        split='train',
        trust_remote_code=True,
        streaming=True
    )
    print(f'Probing the loaded dataset')
    next(iter(orig_dataset))  # a test

    n_shards = orig_dataset.num_shards

    for shard_idx in range(n_shards):
        filepath = Path(f'{output_dir}/{shard_idx:05d}-of-{n_shards:05d}.parquet')
        print(
            f'[{str(datetime.datetime.now())[:-7]}]'
            f' shard {shard_idx}/{n_shards}'
            f' {filepath}'
        )

        if filepath.is_file():
            print('Already exists')
            continue
        
        start_time = time.time()

        iterable_source: IterableDataset = (
            orig_dataset
            .shard(num_shards=n_shards, index=shard_idx)
            .cast_column('audio', Audio(decode=False))
            .map(map_to_mp3, fn_kwargs={'bitrate': args.bitrate})
            ._resolve_features()
        )

        collected_results = list(tqdm(iterable_source))  # should fit in RAM
        
        if args.audio_separately:
            assert args.input_dataset == 'espnet/yodas2'
            for sample_idx, sample in enumerate(collected_results):
                rel_audio_path = Path(f'audio/{sample["video_id"]}.mp3')
                audio_path = output_dir / rel_audio_path
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                with open(audio_path, 'wb') as f:
                    f.write(sample['audio']['bytes'])
                sample['audio'] = {'path': str(rel_audio_path)}
            print(f'Saved {len(collected_results)} mp3 files separately')
        

        Dataset.from_list(collected_results).to_parquet(tmp_path := filepath.with_stem('tmp'))
        tmp_path.rename(filepath)  # to prevent truncated parquet files

        print(
            f'Elapsed {time.time() - start_time:.0f} sec'
            f', saved {filepath.stat().st_size / 1024 ** 2:.1f} MB'
        )