source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name vit1 \
 --device-num 2 \
 --epochs 100 \
 --data-dir /home/yw699/codes/VAR-Watermark/dataset \
 --batch-size 5 \
 --encoder-name "encoder_cnn" \
 --decoder-name "decoder_vit" \
 --encoder-decoder-name "only_var_decoder_fhat" > vit1.log & 