# YODAS initial conversion
screen -S yodas_ru000_128k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 128k -f
screen -S yodas_ru000_32k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 32k
screen -S yodas_ru000_16k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 16k

# YODAS2 initial conversion
for name in ru000 ru001 ru100 ru101 ru102 ru103 ru104 ru105 ru106
do
    screen -S yodas2_${name}_16k -d -m python yodas_to_mp3.py -i espnet/yodas2 -n ${name} -r 16k -s
done

# YODAS histogram
python yodas_length_histogram.py -i yodas_ru000_16k -o yodas_ru000_lengths.png

# YODAS filtering
screen -S yodas_ru000_16k_f -d -m python yodas_filter.py -i yodas_ru000_16k -o yodas_ru000_16k_filtered --min_len 1 --max_len 30
screen -S yodas_ru000_32k_f -d -m python yodas_filter.py -i yodas_ru000_16k -o yodas_ru000_32k_filtered --min_len 1 --max_len 30
screen -S yodas_ru000_128k_f -d -m python yodas_filter.py -i yodas_ru000_16k -o yodas_ru000_128k_filtered --min_len 1 --max_len 30