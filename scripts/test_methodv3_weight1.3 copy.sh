# Todo: 
# Change parameter: // decoder weight 1.3

source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name d \
 --device-num 4 \
 --epochs 300 \
 --data-dir "/home/yw699/codes/DATASET/fhat" \
 --batch-size 32 \
 --encoder-name "encoder_d" \
 --decoder-name "decoder_d" \
 --encoder-decoder-name "encoder_decoder_d" > d.log &