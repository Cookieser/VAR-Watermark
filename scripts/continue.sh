SCRIPT_NAME=$(basename "$0")

EXPERIMENT_NAME="${SCRIPT_NAME%.sh}"


source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py continue \
 --device-num 1 \
 --epochs 400 \
 --folder "/home/yw699/codes/VAR-Watermark/runs/var_ed_v4.2 2025.02.18--00-32-29"\
 --data-dir "/home/yw699/codes/DATASET/fhat2"\
 --encoder-weight 1\
 --decoder-weight 5\
 --adversarial-weight 0.0001\
 --encoder-name "encoder_De_END" \
 --decoder-name "decoder_De_END" \
 --encoder-decoder-name "var_d_encoder_decoder_De_END" > "$EXPERIMENT_NAME.log" &

