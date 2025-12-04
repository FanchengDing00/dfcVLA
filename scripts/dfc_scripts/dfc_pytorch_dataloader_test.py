
import os
import numpy as np
import torch
import torch.distributed as dist

import openpi.training.config as _config
import openpi.training.data_loader as _data

NUM_BATCHES = 2 # Number of batches to load for debugging, don't set it to None, since that means load forever.

def setup_ddp():
    """Init DDP if env vars indicate multi-process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def set_seed(seed: int, local_rank: int):
    """Match training seed behavior (rank offset)."""
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def main():
    # Parse config via CLI (same as training)
    config = _config.cli()

    # DDP setup + seed
    use_ddp, local_rank, _ = setup_ddp()
    set_seed(config.seed, local_rank)

    # Build PyTorch dataloader; set skip_norm_stats=True if you don't have norm_stats.
    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True, 
        num_batches = NUM_BATCHES, 
    )

    # Only main rank prints
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    if is_main:
        print("data_config:", loader.data_config())

    # View batches in dataloader
    rank = dist.get_rank() if dist.is_initialized() else 0
    for observation, actions in loader:
        print(f"\n=== rank {rank} ===")
        obs = observation.to_dict() 
        print(obs)

    # destory ddp
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
