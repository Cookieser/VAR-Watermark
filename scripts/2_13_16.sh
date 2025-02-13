# Todo: 
# Change parameter: //

source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name original \
 --device-num 2 \
 --epochs 300 \
 --data-dir "/home/yw699/codes/DATASET/fhat" \
 --batch-size 32 \
 --encoder-name "encoder_cnn" \
 --decoder-name "decoder_cnn" \
 --encoder-decoder-name "fhat_encoder_decoder" > original.log &