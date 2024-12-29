# YODAS initial conversion
PYTHONPATH=. screen -S yodas_ru000_128k -d -m python scripts/yodas_to_mp3.py -i espnet/yodas -n ru000 -r 128k -f
PYTHONPATH=. screen -S yodas_ru000_32k -d -m python scripts/yodas_to_mp3.py -i espnet/yodas -n ru000 -r 32k
PYTHONPATH=. screen -S yodas_ru000_16k -d -m python scripts/yodas_to_mp3.py -i espnet/yodas -n ru000 -r 16k

# SOVA initial conversion
PYTHONPATH=. screen -S sova_wav -d -m python scripts/sova_to_hf.py -i /gdv_hdd/datasets/speech/sova/RuYouTube -o sova

# or:
PYTHONPATH=. python scripts/sova_to_hf.py -i /gdv_hdd/datasets/speech/sova/RuYouTube -o sova > sova.out 2>&1 & disown

# YODAS2 initial conversion
PYTHONPATH=. screen -S yodas2 -d -m sh -c 'for name in ru000 ru001 ru100 ru101; \
do python scripts/map_dataset.py -i espnet/yodas2 -n ${name} -o datasets/yodas2_${name}_32k \
--extract_audio --bitrate 32k --flush --ast --diarization --verbose ; done; exec bash'

# YODAS histogram
PYTHONPATH=. python scripts/length_histogram.py -i yodas_ru000_16k -o yodas_ru000_lengths.png

# YODAS filtering
PYTHONPATH=. screen -S yodas_ru000_16k_f -d -m python scripts/yodas_filter.py -i yodas_ru000_16k -o yodas_ru000_16k_filtered --min_len 1 --max_len 30
PYTHONPATH=. screen -S yodas_ru000_32k_f -d -m python scripts/yodas_filter.py -i yodas_ru000_32k -o yodas_ru000_32k_filtered --min_len 1 --max_len 30
PYTHONPATH=. screen -S yodas_ru000_128k_f -d -m python scripts/yodas_filter.py -i yodas_ru000_128k -o yodas_ru000_128k_filtered --min_len 1 --max_len 30

PYTHONPATH=. python scripts/audioset_statistics.py --save_features_to_dir tmp --datasets datasets/yodas2_ru000_32k datasets/yodas2_ru001_32k