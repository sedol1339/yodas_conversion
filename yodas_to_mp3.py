import io
import argparse
from pathlib import Path
import socket
from urllib3.connection import HTTPConnection
from typing import Any
import datetime
import time

from datasets import Audio, Dataset, load_dataset, IterableDataset
from pydub import AudioSegment
from tqdm.auto import tqdm

'''
Example:
python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 32k
'''

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
    parser.add_argument('-i','--input_dataset', default='espnet/yodas')
    parser.add_argument('-n','--input_name', default='ru000')
    parser.add_argument('-r','--bitrate', default='32k')
    parser.add_argument('-o','--output_dir', required=False)
    args = parser.parse_args()

    output_dir = (
        args.output_dir
        or f'{args.input_dataset.split("/")[-1]}_{args.input_name}_{args.bitrate}'
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

    try:
        n_shards = orig_dataset.num_shards
    except AttributeError:
        n_shards = orig_dataset.n_shards

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
        collected_results = Dataset.from_list(list(tqdm(iterable_source)))  # should fit in RAM
        collected_results.to_parquet(filepath)
        print(
            f'Elapsed {time.time() - start_time:.0f} sec'
            f', saved {filepath.stat().st_size / 1024 ** 2:.1f} MB'
        )