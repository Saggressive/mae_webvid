
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

/mmu_nlp/wuxing/maguangyuan/miniconda3/envs/mae/bin/python -m torch.distributed.launch --nproc_per_node=8   --master_port 23333  \
    --use_env \
    debug_pretrain.py --accum_iter 4