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

from datasets import Dataset
from pydub import AudioSegment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir')
    parser.add_argument('-r', '--bitrate', default='32k')
    parser.add_argument('-o', '--output_dir', required=False)

    args = parser.parse_args()

    output_dir = args.output_dir or f'sova_{args.bitrate}'

    parts = list(product(
        sorted(Path(args.input_dir).glob('*.zip')),     # archive file
        '0123456789abcdef'                              # dir first letter inside the archive
    ))

    for part_idx, (arhive_path, starting_letter) in enumerate(parts):
        filepath = Path(f'{output_dir}/{part_idx:05d}-of-{len(parts):05d}.parquet')
        print(
            f'[{str(datetime.datetime.now())[:-7]}]'
            f' shard {part_idx}/{len(parts)}'
            f': {arhive_path.name} letter "{starting_letter}"'
            f' -> {filepath}'
        )

        if filepath.is_file():
            print('Already exists')
            continue
        
        start_time = time.time()

        with zipfile.ZipFile(arhive_path) as zip:

            collected_results = []

            files = set(zip.namelist())
            wavs = [name for name in files if name.endswith('.wav')]
            
            for sample_idx, name in enumerate(wavs):

                if sample_idx % 10000 == 0:
                    print(f'{sample_idx / len(wavs) * 100:.0f}%')
                    sys.stdout.flush()
                
                name_tail = '/'.join(name.split('/')[-2:]) # such as 'fac718fc-c1e1-4daf-954a-846d8857c199/000014.wav'
                if name_tail[0] != starting_letter:
                    continue

                with zip.open(name, 'r') as f:  # acts as 'rb'
                    # waveform, rate = librosa.load(f)
                    audio = AudioSegment.from_file(f)
                    audio.export(buffer := io.BytesIO(), format='mp3', bitrate=args.bitrate)

                if (txt_name := name[:-4] + '.txt') in files:
                    with zip.open(txt_name, 'r') as f:
                        text = f.read().decode('utf-8')
                else:
                    text = ''

                collected_results.append({
                    'audio': {'bytes': buffer.read()},
                    'transcription': text,
                    'name': name_tail,
                })

            Dataset.from_list(collected_results).to_parquet(tmp_path := filepath.with_stem('tmp'))
            tmp_path.rename(filepath)  # to prevent truncated parquet files

        print(
            f'Elapsed {time.time() - start_time:.0f} sec'
            f', saved {filepath.stat().st_size / 1024 ** 2:.1f} MB'
        )