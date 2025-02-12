# test the vgg best decoder abilibity
# VGG = TRUE
# decoder weight = 10

source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name vgg_highest_weight \
 --device-num 7\
 --epochs 100 \
 --data-dir /home/yw699/codes/VAR-Watermark/dataset \
 --batch-size 5 \
 --encoder-name "encoder_cnn" \
 --decoder-name "decoder_cnn" \
 --encoder-decoder-name "only_var_decoder_image" > vgg_highest_weight.log & 