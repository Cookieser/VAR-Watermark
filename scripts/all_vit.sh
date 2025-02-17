SCRIPT_NAME=$(basename "$0")

EXPERIMENT_NAME="${SCRIPT_NAME%.sh}"


source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name "$EXPERIMENT_NAME" \
 --device-num 3 \
 --epochs 300 \
 --data-dir "/home/yw699/codes/DATASET/fhat" \
 --batch-size 5 \
 --encoder-weight 0.7\
 --decoder-weight 2\
 --adversarial-weight 0.001\
 --encoder-name "encoder_vit" \
 --decoder-name "decoder_vit" \
 --encoder-decoder-name "only_var_decoder_fhat" > "$EXPERIMENT_NAME.log" &