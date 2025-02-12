# decoder weight = 2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..
nohup python main.py new \
 --name fhat_only_decoder_weight2 \
 --device-num 3 \
 --epochs 100 \
 --data-dir /home/yw699/codes/VAR-Watermark/dataset \
 --batch-size 8 \
 --encoder-name "encoder_cnn" \
 --decoder-name "decoder_cnn" \
 --encoder-decoder-name "only_var_decoder_fhat" > test.log & 
