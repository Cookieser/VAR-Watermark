SCRIPT_NAME=$(basename "$0")

EXPERIMENT_NAME="${SCRIPT_NAME%.sh}"


source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py continue \
 --device-num 7 \
 --epochs 400 \
 --folder "/home/yw699/codes/VAR-Watermark/runs/var_ed_v3 2025.02.14--20-25-13"\
 --encoder-weight 10\
 --decoder-weight 1\
 --adversarial-weight 0.0001\
 --encoder-name "encoder_De_END" \
 --decoder-name "decoder_De_END" \
 --encoder-decoder-name "var_ed_encoder_decoder_De_END" > "$EXPERIMENT_NAME.log" &