# Todo: 
# Change parameter: // decoder weight 10  1 -4

source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name d \
 --device-num 6 \
 --epochs 300 \
 --data-dir "/home/yw699/codes/DATASET/fhat" \
 --batch-size 64 \
 --encoder-name "encoder_d" \
 --decoder-name "decoder_d" \
 --encoder-decoder-name "encoder_decoder_d" > d.log &