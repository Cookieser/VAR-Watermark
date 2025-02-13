source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ../VAR_model
python sample_fhat.py --total_samples 5000 --save_path /home/yw699/codes/DATASET/fhat/train/train_class/ 
python sample_fhat.py --total_samples 500 --save_path /home/yw699/codes/DATASET/fhat/val/val_class/ 