SCRIPT_NAME=$(basename "$0")

EXPERIMENT_NAME="${SCRIPT_NAME%.sh}"


source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name "$EXPERIMENT_NAME" \
 --device-num  4\
 --epochs 400 \
 --data-dir "/home/yw699/codes/DATASET/fhat" \
 --batch-size 8 \
 --encoder-weight 0.7\
 --decoder-weight 1\
 --adversarial-weight 0.001\
 --encoder-name "encoder_cnn" \
 --decoder-name "decoder_cnn" \
 --encoder-decoder-name "var_encoder_decoder" > "$EXPERIMENT_NAME.log" &