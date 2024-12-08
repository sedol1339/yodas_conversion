screen -S yodas_ru000_128k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 128k -f
screen -S yodas_ru000_32k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 32k
screen -S yodas_ru000_16k -d -m python yodas_to_mp3.py -i espnet/yodas -n ru000 -r 16k

for name in ru000 ru001 ru100 ru101 ru102 ru103 ru104 ru105 ru106
do
    screen -S yodas2_${name}_16k -d -m python yodas_to_mp3.py -i espnet/yodas2 -n ${name} -r 16k -s
done