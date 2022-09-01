node_rank=$1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

nohup python -m torch.distributed.launch --nnodes=4 --master_addr=10.116.150.13 --node_rank=${node_rank}  --nproc_per_node=8   --master_port 3243  \
    --use_env main_pretrain.py \
    >logs_v1/${node_rank}.log 2>&1 &