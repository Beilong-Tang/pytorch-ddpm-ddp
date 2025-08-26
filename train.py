import importlib
import argparse
import os
from functools import partial
import random
import numpy as np
import copy
import os.path as op
from pathlib import Path
from tqdm import trange

from config.template import DefaultConfig

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score

import torch 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import make_grid, save_image

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def setup_seed(seed, rank):
    SEED = int(seed) + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return SEED

## ddp process
def setup(rank, world_size, backend, port=12355):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def load_config(module_path: str, class_name: str ="Config", **kwargs):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def infiniteloop(dataloader, epoch=0):
    epoch = epoch
    while True:
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(int(epoch))
        for x, y in iter(dataloader):
            yield x
        epoch +=1

def warmup_lr(step, warmup):
    return min(step, warmup) / warmup

def main(rank, config:DefaultConfig, args):
    ## DDP
    print(f"INFO: [rank[{rank}] | {len(config.gpus)}] inited...")
    setup(rank, len(config.gpus), args.dist_backend)
    device = config.gpus[rank] 
    torch.cuda.set_device(device)
    setup_seed(args.seed, rank)

    # model setup
    net_model = UNet(
        T=config.T, ch=config.ch, ch_mult=config.ch_mult, attn=config.attn,
        num_res_blocks=config.num_res_blocks, dropout=config.dropout)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=partial(warmup_lr, warmup=config.warmup))
    trainer = GaussianDiffusionTrainer(
        net_model, config.beta_1, config.beta_T, config.T).cuda()
    trainer = DDP(trainer, device_ids=[rank])

    ema_sampler = GaussianDiffusionSampler(
        ema_model, config.beta_1, config.beta_T, config.T, config.img_size,
        config.mean_type, config.var_type).to(device)
    
    # Log and Resume setup
    start_step = 0
    if os.path.exists(config.logdir):
        if args.resume:
            ckpt = str(Path(config.logdir) / "ckpt.pt")
            if os.path.exist(ckpt):
                ckpt = torch.load(ckpt, map_location='cpu')
                ema_model.load_state_dict(ckpt['ema_model'])
                net_model.load_state_dict(ckpt['net_model'])
                sched.load_state_dict(ckpt['sched'])
                optim.load_state_dict(ckpt['optim'])
                start_step = ckpt['step']
        else:
            print(f"WARNING: {config.logdir} exist. Overwriting it now!")
    else:
        os.makedirs(op.join(config.logdir, 'sample'))
    
    if rank == 0:
        writer = SummaryWriter(config.logdir)
        # For evaluation
        x_T = torch.randn(config.sample_size, 3, config.img_size, config.img_size).cuda()

    # Dataset
    if rank == 0:
        tr_dataset = CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    dist.barrier()
    if rank !=0:
        tr_dataset = CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    tr_dataloader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=config.batch_size // len(config.gpus), 
        shuffle=False,
        num_workers=config.num_workers,  
        worker_init_fn = seed_worker,
        sampler=DistributedSampler(tr_dataset))
    print(f"len tr_dataloder dataset for rank {rank}: {len(tr_dataloader) * config.batch_size // len(config.gpus)}")
    tr_dataloader = infiniteloop(tr_dataloader, epoch = (start_step * config.batch_size) / len(tr_dataset)+ 1)
    if rank == 0:
        pbar = trange(start_step, config.total_steps)
    else:
        pbar = range(start_step, config.total_steps)
    
    dist.barrier()
    print(f"rank[{rank}]/ {len(config.gpus)} started training")
    
    for step in pbar:
        x_0 = next(tr_dataloader)
        x_0 = x_0.cuda()
        loss = trainer(x_0).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net_model.parameters(), config.grad_clip
        )
        optim.step()
        sched.step()
        if rank == 0:
            ema(net_model, ema_model, config.ema_decay)
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

        # Sampling
        if config.sample_step > 0 and step % config.sample_step == 0:
            if rank ==0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        config.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()
            dist.barrier()

        # save
        if config.save_step > 0 and step % config.save_step == 0:
            if rank == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(config.logdir, 'ckpt.pt'))
            dist.barrier()
    print("Done..")

    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config_path', help="config path")
    p.add_argument("--port", default=12355, type =int, help="ddp port")
    p.add_argument("--dist-backend", default='nccl', type =str, help="ddp backend")
    p.add_argument("--resume", default=False, type =bool, help="whether to resume from last training")
    p.add_argument("--seed", default=1234)
    args = p.parse_args()
    args.config_path = os.path.relpath(args.config_path.replace("/", "."), os.getcwd())
    config: DefaultConfig = load_config(args.config_path)
    if isinstance(config.gpus, list) and len(config.gpus) > 1:
        print("running DDP")
        mp.spawn(main, args=(config, args), nprocs=len(config.gpus), join=True)
        print("Done")
        cleanup()
    else:
        main(0, config, args)
    pass