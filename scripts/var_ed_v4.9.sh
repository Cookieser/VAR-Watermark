SCRIPT_NAME=$(basename "$0")

EXPERIMENT_NAME="${SCRIPT_NAME%.sh}"


source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name "$EXPERIMENT_NAME" \
 --device-num  6\
 --epochs 300 \
 --data-dir "/home/yw699/codes/DATASET/fhat2" \
 --batch-size 5 \
 --useVGG True\
 --encoder-weight 1\
 --decoder-weight 10\
 --adversarial-weight 0.0001\
 --encoder-name "encoder_De_END" \
 --decoder-name "decoder_De_END" \
 --encoder-decoder-name "var_ed_encoder_decoder_De_END_vgg" > "$EXPERIMENT_NAME.log" &