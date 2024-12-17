# YODAS initial conversion
screen -S yodas_ru000_128k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 128k -f
screen -S yodas_ru000_32k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 32k
screen -S yodas_ru000_16k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 16k

# SOVA initial conversion
screen -S sova_16k -d -m python sova_to_mp3.py -i /gdv_hdd/datasets/speech/sova/RuYouTube -r 16k

# YODAS2 initial conversion
screen -S yodas2_ru000_16k -d -m sh -c 'python yodas_to_mp3.py -i espnet/yodas2 -n ru000 -r 32k -s -f --ast --diarization; exec bash'

# YODAS histogram
python yodas_length_histogram.py -i yodas_ru000_16k -o yodas_ru000_lengths.png

# YODAS filtering
screen -S yodas_ru000_16k_f -d -m python yodas_filter.py -i yodas_ru000_16k -o yodas_ru000_16k_filtered --min_len 1 --max_len 30
screen -S yodas_ru000_32k_f -d -m python yodas_filter.py -i yodas_ru000_32k -o yodas_ru000_32k_filtered --min_len 1 --max_len 30
screen -S yodas_ru000_128k_f -d -m python yodas_filter.py -i yodas_ru000_128k -o yodas_ru000_128k_filtered --min_len 1 --max_len 30