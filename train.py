import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict

from options import *
from model.hidden import Hidden
from average_meter import AverageMeter
from VAR_model.var_use import VarTool
from tqdm import tqdm
def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """

    var = VarTool(device = device);

    train_data, val_data = utils.get_data_loaders(hidden_config, train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    save_each = 10

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        encoder_weight, decoder_weight = weight_change(hidden_config.encoder_loss,hidden_config.decoder_loss,epoch);
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        logging.info('Loss weight: encoder: {}  decoder:{}\n '.format(encoder_weight, decoder_weight))


        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        #for image, _ in train_data:
        for image, _ in tqdm(train_data, desc="Processing train data"):
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, _ = model.train_on_batch([image, message],var,encoder_weight, decoder_weight)

            for name, loss in losses.items():
                training_losses[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)

        if epoch % save_each == 0:
            first_iteration = True  
        else:
            first_iteration = False

        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        #for image, _ in val_data:
        for image, _ in tqdm(val_data, desc="Processing validation data"):
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message],var,encoder_weight, decoder_weight)
            for name, loss in losses.items():
                validation_losses[name].update(loss)
            

            if first_iteration:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                
                images_folder = os.path.join(this_run_folder, 'images')
                os.makedirs(images_folder, exist_ok=True)  
                filename_watermark = os.path.join(images_folder, 'epoch-watermark-{}.pt'.format(epoch))
                filename_original = os.path.join(images_folder, 'epoch-original-{}.pt'.format(epoch))
                torch.save(image, filename_original)
                print(f"Saved original_images to {filename_original}")
                torch.save(encoded_images, filename_watermark)
                print(f"Saved encoded_images to {filename_watermark}")

                first_iteration = False

        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        if epoch % save_each == 0:
            utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)


def weight_change(weight1,weight2,epoch):
    if(epoch < 1000):
        encoder_weight = weight1
        decoder_weight = weight2
    else:
        encoder_weight = weight2
        decoder_weight = weight1
    return encoder_weight,decoder_weight