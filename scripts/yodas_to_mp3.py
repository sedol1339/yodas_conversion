import io
import argparse
from pathlib import Path
import socket
from urllib3.connection import HTTPConnection
import datetime
import time
import shutil
from dataclasses import asdict
from contextlib import contextmanager

import datasets
from datasets import Audio, Dataset, load_dataset, IterableDataset
from pydub import AudioSegment
from tqdm.auto import tqdm
import librosa

from src.speaker_diarization import SpeakerDiarizationWrapper
from src.spectrogram_transformer import AudioSpectrogramTransformer


@contextmanager
def catchtime(name: str, disable: bool = False):
    """Prints elapsed time on exit from the context."""
    start = time.time()
    yield lambda: time.time() - start
    if not disable:
        print(f'{name}: {time.time() - start:.1f} seconds')


if __name__ == '__main__':
    # fixing Huggingface "read timeout" issue
    # https://github.com/huggingface/transformers/issues/12575#issuecomment-1716743090
    # HTTPConnection.default_socket_options = ( 
    #     HTTPConnection.default_socket_options + [
    #     (socket.SOL_SOCKET, socket.SO_SNDBUF, 20000000), 
    #     (socket.SOL_SOCKET, socket.SO_RCVBUF, 20000000)
    #     ])

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dataset', default='espnet/yodas')
    parser.add_argument('-n', '--input_name', default='ru000')
    parser.add_argument('-r', '--bitrate', default='32k')
    parser.add_argument('-o', '--output_dir', required=False)
    parser.add_argument('-s', '--audio_separately', action='store_true')
    parser.add_argument('-f', '--flush', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ast', action='store_true')
    parser.add_argument('--ast_batch_size', default=256)
    parser.add_argument('--ast_min_length', default=1)
    parser.add_argument('--diarization', action='store_true')
    parser.add_argument('--diarization_segmentation_batch_size', default=256)
    parser.add_argument('--diarization_embedding_batch_size', default=128)

    args = parser.parse_args()

    output_dir = (
        args.output_dir
        or f'datasets/{args.input_dataset.split("/")[-1]}_{args.input_name}_{args.bitrate}'
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
    
    if args.diarization:
        speaker_diarization = SpeakerDiarizationWrapper(
            segmentation_batch_size=args.diarization_segmentation_batch_size,
            embedding_batch_size=args.diarization_embedding_batch_size
        )
    
    if ((args.ast or args.diarization) and args.input_dataset != 'espnet/yodas2'):
        print('Warning: currently batching in AST or diarization is not efficient for an already segmented dataset')

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
        for sample_idx, sample in enumerate(tqdm(iterable_source, disable=args.verbose)):
            if args.verbose:
                print(f'Sample {sample_idx}')
            sample = dict(**sample)  # otherwise it does not work for some reason
            orig_bytes = sample['audio']['bytes']
            # to mp3
            with catchtime('\tConverting to mp3 bytes', disable=not args.verbose):
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

            if args.ast or args.diarization:
                with catchtime('\tLoading wav bytes', disable=not args.verbose):
                    # here we already have AudioSegment, but there is no documented and consistent
                    # with librosa way to convert it into waveform array
                    # do we need to use sr=16_000 with AST?
                    waveform, sr = librosa.load(io.BytesIO(orig_bytes), sr=16_000)
                
                if args.verbose:
                    print(f'\tAudio length: {len(waveform) / sr:.1f} sec')

                if args.ast:
                    with catchtime('\tAST', disable=not args.verbose):
                        sample['ast'] = ast.predict_on_long_audio(
                            waveform,
                            batch_size=int(args.ast_batch_size),
                            sampling_rate=sr,
                            min_length=int(args.ast_min_length),
                        )
                
                if args.diarization:
                    with catchtime('\tDiarization', disable=not args.verbose):
                        diarization_results = asdict(speaker_diarization.predict_on_long_audio(
                            waveform, sampling_rate=sr
                        ))
                        sample['segments'] = diarization_results['segments']
                        sample['speaker_embeddings'] = diarization_results['speaker_embeddings']
            
            collected_results.append(sample)

        # with open('collected_results.pkl', 'wb') as f:
        #     pickle.dump(collected_results, f)

        with catchtime('Saving parquet', disable=not args.verbose):
            Dataset.from_list(collected_results).to_parquet(tmp_path := filepath.with_stem('tmp'))
            tmp_path.rename(filepath)  # to prevent truncated parquet files

        print(
            f'Elapsed {time.time() - start_time:.0f} sec'
            f', saved {filepath.stat().st_size / 1024 ** 2:.1f} MB'
        )