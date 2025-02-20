source ~/anaconda3/etc/profile.d/conda.sh
conda activate test
cd ../VAR_model
python sample_image.py --total_samples 100 --save_path /home/yw699/codes/DATASET/image2/train/train_class/ --seed 7 --device 8
python sample_image.py --total_samples 10 --save_path /home/yw699/codes/DATASET/image2/val/val_class/ --seed 13 --device 8 