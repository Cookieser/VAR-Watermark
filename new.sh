source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
nohup python main.py new \
 --name test \
 --device-num 3 \
 --epochs 30 \
 --data-dir /home/yw699/codes/VAR-Watermark/dataset \
 --batch-size 8 \
 --encoder-name "encoder_cnn" \
 --decoder-name "decoder_cnn" \
 --encoder-decoder-name "only_var_decoder_fhat" > test.log & 
