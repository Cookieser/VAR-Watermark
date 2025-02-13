# Todo: 
# Change parameter: //

source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name vit_all \
 --device-num 3 \
 --epochs 300 \
 --data-dir /home/yw699/codes/DATASET/fhat \
 --batch-size 5 \
 --encoder-name "encoder_vit" \
 --decoder-name "decoder_vit" \
 --encoder-decoder-name "only_var_decoder_fhat" > vit_all.log & 