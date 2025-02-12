# Todo: test the encoder-vit 
# Change parameter: //

source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name vit2 \
 --device-num 8 \
 --epochs 100 \
 --data-dir /home/yw699/codes/VAR-Watermark/dataset \
 --batch-size 5 \
 --encoder-name "encoder_vit" \
 --decoder-name "decoder_cnn" \
 --encoder-decoder-name "only_var_decoder_fhat" > vit2.log & 