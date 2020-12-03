import os
import time
import argparse
import math
from numpy import finfo


import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import load_model
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import time


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")




def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(index, flags):

    torch.manual_seed(flags['seed'])
    device = xm.xla_device()

    if not xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    iteration = 0
    learning_rate = flags['hparams'].learning_rate


    train_dataset = TextMelLoader(flags['hparams'].training_files, flags['hparams'])

    val_dataset = TextMelLoader(flags['hparams'].validation_files, flags['hparams'],
                           speaker_ids=train_dataset.speaker_ids)

    collate_fn = TextMelCollate(flags['hparams'].n_frames_per_step)

    if xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=flags['batch_size'],
        sampler=train_sampler,
        num_workers=flags['num_workers'],
        collate_fn=collate_fn,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=flags['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=flags['num_workers'],
        collate_fn=collate_fn,
        drop_last=True)



    model = load_model(flags['hparams']).to(device).train()
    criterion = Tacotron2Loss()


    optimizer = torch.optim.Adam(model.parameters(), lr=flags['hparams'].learning_rate,
                                 weight_decay=flags['hparams'].weight_decay)


    for epoch in range(flags['hparams'].epochs):
        train_start = time.time()
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)

        # (text, mel, speaker_id, f0)
        for batch_num, batch in enumerate(para_train_loader):


            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)

            if flags['num_workers']>1:
                reduced_loss = reduce_tensor(loss.data, flags['num_workers']).item()
            else:
                reduced_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()

            xm.optimizer_step(optimizer)

            elapsed_train_time = time.time() - train_start
            print("Batch Process", index, "finished training. Train time was:", elapsed_train_time)

        if (iteration % flags['hparams'].iters_per_checkpoint == 0):
            # validate(model, criterion, val_dataset, iteration,
            #          flags['hparams'].batch_size, flags['n_gpus'], collate_fn, logger,
            #          flags['hparams'].distributed_run, flags['rank'])

            model.eval()
            eval_start = time.time()

            with torch.no_grad():

                para_train_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
                for i, batch in enumerate(para_train_loader):
                    val_loss = 0.0
                    x, y = model.parse_batch(batch)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)

                    if flags['num_workers']>1:
                        reduced_val_loss = reduce_tensor(loss.data, flags['num_workers']).item()
                    else:
                        reduced_val_loss = loss.item()

                    val_loss += reduced_val_loss

                val_loss = val_loss / (i + 1)


        elapsed_eval_time = time.time() - eval_start
        print("Process", index, "finished evaluation. Evaluation time was:", elapsed_eval_time)
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))


        iteration += 1



if __name__ == '__main__':

    hparams = create_hparams()

    flags = {}
    flags['output_directory']= "/sound/mellotron/outdir"
    flags['log_directory']= "/sound/mellotron/logdir"
    flags['checkpoint_path'] =  "/sound/mellotron/models/mellotron_libritts.pt"
    flags['warm_start'] =  False
    flags['num_workers'] = 8
    flags['seed']=1234
    flags['batch_size'] =   16
    flags['hparams'] =  hparams


    # train(args.output_directory, args.log_directory, args.checkpoint_path,
    #       args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
    xmp.spawn(train, args=(flags,), nprocs=8, start_method='fork')
    ###########
