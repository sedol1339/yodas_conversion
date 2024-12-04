screen -S yodas_ru000_128k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 128k
screen -S yodas_ru000_32k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 32k
screen -S yodas_ru000_16k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 16k
screen -S yodas2_ru000_16k -d -m python yodas_to_mp3.py -i espnet/yodas2 -n ru000 -r 16k -s