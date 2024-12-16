# VAR-Watermark

Visual AutoRegressive modeling (VAR) is an innovative generation paradigm that redefines autoregressive learning for images. Instead of the traditional raster-scan "next-token prediction," it adopts a coarse-to-fine approach referred to as "next-scale prediction" or "next-resolution prediction."

For the first time, this approach enables GPT-style autoregressive (AR) models to surpass diffusion transformers in image generation, marking it as a significant alternative to the widely popular diffusion-based algorithms. VAR presents considerable potential and research value in the field of image generation.

We aim to add watermarks into these generated images.

- **Paper**: [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)
- **GitHub**: [FoundationVision/VAR](https://github.com/FoundationVision/VAR)

## HiDDeN:Hiding Data With Deep Networks

We plan to conduct initial experiments using the HiDDeN method for embedding information into images. 

- **Paper**: [HiDDeN: Hiding Data With Deep Networks](https://arxiv.org/abs/1807.09937)
- **GitHub**: The implementation will be based on https://github.com/ando-khachatryan/HiDDeN/tree/master

## Sample 

We use 5000 embedding features for training and 500 embedding features for validation. 

These can be sample through `VAR-Watermark/VAR_model/sample.py`

```
python sample.py --total_samples 5000 --save_path ../dataset/train/train_class/
python sample.py --total_samples 500 --save_path ../dataset/val/val_class/
```

The data directory has the following structure:

```
<data_root>/
  train/
    train_class/
      train_image1.jpg
      train_image2.jpg
      ...
  val/
    val_class/
      val_image1.jpg
      val_image2.jpg
      ...
```

## Running

You will need to install the requirements, then run 

```
python main.py new --name <experiment_name> --data-dir <data_root> --batch-size <b> 
```

To adapt the **HiDDeN** method for watermarking VAR's embedding features, we have modified the network's structure and scales to align with the input dimensions specific to our use case.

In the original HiDDeN implementation, the input is an image tensor of size `[3, H, W]`

- `3` represents the RGB channels.
- `H` and `W` are the height and width (resolution) of the image.

For our modified setup, the watermark is embedded directly into the embedding features produced by VAR, with a tensor size of `[32, 16, 16]`

- `32` is the number of channels in the VAR embedding features.
- `16 × 16` represents the largest patch number.



## Experiment

#### Sampling Parameters for VAR

- **Model Depth**: 24
- **Batch Size**: 10
- **Patch Numbers**: `(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)`
- **Embedding Features**: 32
- **Number of Classes**: 1000 (ImageNet)

#### Training Parameters for Watermarking

- **Training Embedding Features**: 5000

- **Validation Embedding Features**: 500

- **Epochs**: 300

- **Batch Size**: 64

#### Image Results Description

The following images demonstrate the output after training with the watermarking model for **300 epochs**:

1. **First Row**: Original images without watermarks.
2. **Second Row**: Images embedded with watermarks, generated using the trained model parameters after 300 epochs.

![output_combined_image](https://pic-1306483575.cos.ap-nanjing.myqcloud.com/output_combined_image.png)





#### Test_model.py

```
python test_model.py --checkpoint-file "./runs/var-watermark 2024.12.16--13-12-23/checkpoints/var-watermark--epoch-300.pyt" --source-image ./dataset/val/val_class/image0.pt --options-file "./var-watermark 2024.12.16--13-12-23/options-and-config.pickle"
```

output

```
Original Tensor Shape: torch.Size([1, 32, 16, 16])
original: [[1. 1. 0. 1. 1. 1. 1. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1.
  0. 0. 1. 1. 1. 1.]]
decoded : [[1. 1. 0. 1. 1. 1. 1. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1.
  0. 0. 1. 1. 1. 1.]]
error : 0.000
bitwise_avg_err: 0.0
```

