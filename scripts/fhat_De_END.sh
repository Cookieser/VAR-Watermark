
SCRIPT_NAME=$(basename "$0")

EXPERIMENT_NAME="${SCRIPT_NAME%.sh}"


source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name "$EXPERIMENT_NAME" \
 --device-num 6 \
 --epochs 300 \
 --data-dir "/home/yw699/codes/DATASET/fhat" \
 --batch-size 64 \
 --encoder-weight 1\
 --decoder-weight 10\
 --adversarial-weight 0.0001\
 --encoder-name "encoder_d" \
 --decoder-name "decoder_d" \
 --encoder-decoder-name "encoder_decoder_d" > "$EXPERIMENT_NAME.log" &