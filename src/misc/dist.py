import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.distributed as tdist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_print(args.rank == 0)


def warp_model(model, find_unused_parameters=False, sync_bn=False, ):
    if is_dist_available_and_initialized():
        rank = int(os.environ['LOCAL_RANK'])
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)
    return model


def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset,
                            loader.batch_size,
                            sampler=sampler,
                            drop_last=loader.drop_last,
                            collate_fn=loader.collate_fn,
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers,)
    return loader


def setup_print(is_main):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def is_dist_available_and_initialized():
    if not tdist.is_available():
        return False
    if not tdist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return tdist.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return tdist.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        tdist.all_reduce(values)

        if avg is True:
            values /= world_size
        _data = {k: v for k, v in zip(keys, values)}
    return _data



def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    tdist.all_gather_object(data_list, data)
    return data_list


def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def set_seed(seed):
    seed = seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
