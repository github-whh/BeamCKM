import logging
import os
import random
import time
import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from dataLoader.loader import BeamCKM
from lossFunction.loss import Design_loss

def worker_init_fn(worker_id, seed):
    """set random seed, for reproduce"""
    random.seed(seed + worker_id)

def trainer(args, model, snapshot_path, accelerator=None):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank) 
    # Initial accelerate
    if accelerator is None:
        accelerator = Accelerator(
            mixed_precision='fp16' if getattr(args, 'fp16', False) else 'no',
            cpu=False,
            split_batches=False,
            device_placement=True,
        )
    torch.cuda.set_device(local_rank)
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[local_rank])
    accelerator.wait_for_everyone()

    transform = None
    
    # prepare dataset
    train_set = BeamCKM(phase="train", p=0.05, transform=transform) # p is undersampling rate
    val_set = BeamCKM(phase="val", p=0.05, transform=transform) # p is undersampling rate

    if accelerator.is_main_process:
        logging.basicConfig(
            filename=os.path.join(snapshot_path, "log.txt"),
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%m/%d %H:%M'
        )
        logging.info(str(args))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=partial(worker_init_fn, seed=args.seed),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initial optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_val_loss = float('inf')
    start_time = time.time()
    try:
        for epoch_num in tqdm(range(args.max_epochs),ncols=80,disable=not accelerator.is_main_process):
            model.train()
            train_loss = 0.0; train_samples = 0; cnt = 0; train_mse_loss = 0.0
            for inputs, labels in train_loader:
                cnt = cnt + 1 # for debug
                inputs = inputs.to(accelerator.device).half()
                labels = labels.to(accelerator.device).half()
                with accelerator.autocast():
                    outputs = model(inputs)
                    loss, mseloss = Design_loss(outputs, labels, epoch_num)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad.contiguous()
                optimizer.zero_grad()
                train_loss += loss.item() * inputs.size(0)
                train_mse_loss += mseloss.item() * inputs.size(0)
                train_samples += inputs.size(0)
            model.eval()
            val_loss = 0.0; val_samples = 0; val_mse_loss = 0.0
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(accelerator.device).half()
                val_labels = val_labels.to(accelerator.device).half()
                with torch.no_grad(), accelerator.autocast():
                    val_outputs = model(val_inputs)
                    loss, mseloss = Design_loss(val_outputs, val_labels, epoch_num)
                    val_loss += loss.item() * val_inputs.size(0)
                    val_mse_loss += mseloss.item() * val_inputs.size(0)
                    val_samples += val_inputs.size(0)
            # calculate loss
            train_loss_tensor = torch.tensor(train_loss, device=accelerator.device)
            train_mse_loss_tensor = torch.tensor(train_mse_loss, device=accelerator.device)
            train_samples_tensor = torch.tensor(train_samples, device=accelerator.device)
            val_loss_tensor = torch.tensor(val_loss, device=accelerator.device)
            val_mse_loss_tensor = torch.tensor(val_mse_loss, device=accelerator.device)
            val_samples_tensor = torch.tensor(val_samples, device=accelerator.device)

            train_loss = train_loss_tensor.item() / train_samples_tensor.item()
            val_loss = val_loss_tensor.item() / val_samples_tensor.item()
            train_mse_loss = train_mse_loss_tensor.item() / train_samples_tensor.item()
            val_mse_loss = val_mse_loss_tensor.item() / val_samples_tensor.item()
            scheduler.step()
            if accelerator.is_main_process:
                log_line = f" Epoch {epoch_num}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Train MSE: {train_mse_loss:.8f}, Val MSE: {val_mse_loss:.8f}"
                if val_mse_loss < best_val_loss:
                    best_val_loss = val_mse_loss
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))
                    log_line += f" | Best model saved (val_loss={val_mse_loss:.4f})"
                logging.info(log_line)
    finally:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            total_time = time.time() - start_time
            hours, rem = divmod(total_time, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = f"\n\nTotal training time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s"
            with open(os.path.join(snapshot_path, "log.txt"), 'a') as f:
                f.write(time_str)
        accelerator.free_memory()
    return "Training Finished!"
