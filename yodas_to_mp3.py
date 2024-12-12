import io
import argparse
from pathlib import Path
import pickle
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
import librosa

from spectrogram_transformer import AudioSpectrogramTransformer


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
    parser.add_argument('--ast', action='store_true')
    parser.add_argument('--ast_batch_size', default=256)
    parser.add_argument('--ast_segment_length', default=10)
    parser.add_argument('--ast_segment_shift', default=5)
    parser.add_argument('--ast_min_length', default=1)

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
        shutil.rmtree(
            Path(datasets.config.HF_CACHE_HOME)
            / 'modules/datasets_modules/datasets/espnet--yodas2',
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

    if args.ast:
        ast = AudioSpectrogramTransformer()
        if args.input_dataset != 'espnet/yodas2':
            print('Warning: currently batching in AST is not efficient for an already segmented dataset')

    for shard_idx in range(n_shards := orig_dataset.num_shards):

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
        )

        collected_results = []
        for sample_idx, sample in enumerate(tqdm(iterable_source)):
            sample = dict(**sample)  # otherwise it does not work for some reason
            orig_bytes = sample['audio']['bytes']
            # to mp3
            audio = AudioSegment.from_file(io.BytesIO(orig_bytes))
            audio.export(buffer := io.BytesIO(), format='mp3', bitrate=args.bitrate)
            mp3_bytes = buffer.read()
            
            if args.audio_separately:
                assert args.input_dataset == 'espnet/yodas2'
                rel_audio_path = Path(f'audio/{sample["video_id"]}.mp3')
                audio_path = output_dir / rel_audio_path
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                with open(audio_path, 'wb') as f:
                    f.write(mp3_bytes)
                sample['audio'] = {'path': str(rel_audio_path)}
            else:
                sample['audio'] = {'bytes': mp3_bytes}

            if args.ast:
                # here we already have AudioSegment, but there is no documented and consistent
                # with librosa way to convert it into waveform array
                # do we need to use sr=16_000 with AST?
                waveform, sr = librosa.load(io.BytesIO(orig_bytes), sr=16_000)
                sample['ast'] = ast.predict_on_long_audio(
                    waveform,
                    batch_size=int(args.ast_batch_size),
                    segment_length=int(args.ast_segment_length),
                    segment_shift=int(args.ast_segment_shift),
                    sampling_rate=sr,
                    min_length=int(args.ast_min_length),
                )
            
            collected_results.append(sample)

        with open('collected_results.pkl', 'wb') as f:
            pickle.dump(collected_results, f)

        Dataset.from_list(collected_results).to_parquet(tmp_path := filepath.with_stem('tmp'))
        tmp_path.rename(filepath)  # to prevent truncated parquet files

        print(
            f'Elapsed {time.time() - start_time:.0f} sec'
            f', saved {filepath.stat().st_size / 1024 ** 2:.1f} MB'
        )