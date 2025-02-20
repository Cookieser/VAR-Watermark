SCRIPT_NAME=$(basename "$0")

EXPERIMENT_NAME="${SCRIPT_NAME%.sh}"


source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ..


nohup python main.py new \
 --name "$EXPERIMENT_NAME" \
 --device-num  7\
 --epochs 300 \
 --data-dir "/home/yw699/codes/DATASET/fhat" \
 --batch-size 8 \
 --encoder-weight 1\
 --decoder-weight 10\
 --adversarial-weight 0.0001\
 --encoder-name "encoder_De_END" \
 --decoder-name "decoder_De_END" \
 --encoder-decoder-name "var_ed_encoder_decoder_De_END" > "$EXPERIMENT_NAME.log" &



#  nohup python main.py continue \
#  --device-num 7 \
#  --epochs 240 \
#  --folder "/home/yw699/codes/VAR-Watermark/runs/var_ed_v4.1 2025.02.17--20-03-29"\
#  --encoder-weight 10\
#  --decoder-weight 1\
#  --adversarial-weight 0.0001\
#  --encoder-name "encoder_De_END" \
#  --decoder-name "decoder_De_END" \
#  --encoder-decoder-name "var_ed_encoder_decoder_De_END" > "$EXPERIMENT_NAME.log" &



# nohup python main.py continue \
#  --device-num 7 \
#  --epochs 300 \
#  --folder "/home/yw699/codes/VAR-Watermark/runs/var_ed_v4.1 2025.02.17--20-03-29"\
#  --data-dir "/home/yw699/codes/DATASET/fhat2"\
#  --encoder-weight 1\
#  --decoder-weight 10\
#  --adversarial-weight 0.0001\
#  --encoder-name "encoder_De_END" \
#  --decoder-name "decoder_De_END" \
#  --encoder-decoder-name "var_d_encoder_decoder_De_END" > "$EXPERIMENT_NAME.log" &