SCRIPT_NAME=$(basename "$0")

EXPERIMENT_NAME="${SCRIPT_NAME%.sh}"


source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py continue \
 --device-num 5 \
 --epochs 300 \
 --folder "/home/yw699/codes/VAR-Watermark/runs/fhat_only_decoder_De_END 2025.02.17--10-57-43"\
 --encoder-weight 10\
 --decoder-weight 1\
 --adversarial-weight 0.001\
 --encoder-name "encoder_De_END" \
 --decoder-name "decoder_De_END" \
 --encoder-decoder-name "var_d_encoder_decoder_De_END" > "$EXPERIMENT_NAME.log" &