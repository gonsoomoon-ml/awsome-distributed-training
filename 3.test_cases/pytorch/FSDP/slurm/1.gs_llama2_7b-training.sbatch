#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=4 # number of nodes to use
#SBATCH --job-name=llama2_7b-FSDP # name of your job
#SBATCH --output=logs/%x_%j.out # logfile for stdout
#SBATCH --error=logs/%x_%j.err # logfile for stderr, remove it to merge both outputs
#SBATCH --exclusive # job has exclusive use of the resource, no sharing

set -ex;

###########################
###### User Variables #####
###########################

# GPUS_PER_NODE=8 # 4 for G5.12x, 8 for P4/P5
GPUS_PER_NODE=4 # 4 for G5.12x, 8 for P4/P5

###############################
###### Container Variable #####
###############################
# Uncomment if you want to use a container instea of Virtual Environment.
#export CONTAINER_IMAGE=$(pwd)/pytorch-fsdp.sqsh
export DATA_PATH=/fsx
export FSX_MOUNT=$(pwd):$DATA_PATH

###########################
## Environment Variables ##
###########################

## NCCL settings for G5.12xlarge (no EFA)
export NCCL_DEBUG=INFO
# LD_PRELOAD is required for PyTorch to find the NCCL library
# This path assumes you are using the Deep Learning AMI
# If you are not using the DLAMI, you may need to update this path
export LD_PRELOAD=/usr/local/cuda-12.8/lib/libnccl.so
# For G5.12xlarge, use the default network interface (no EFA)
export NCCL_SOCKET_IFNAME=^docker,lo,veth

## Set HuggingFace metadata timeout (in seconds) for large clusters
export HF_HUB_ETAG_TIMEOUT=60

########################
####### Container ######
########################

if [ ! -z $CONTAINER_IMAGE ];
then
    export TRAIN_SCRIPT=./train.py
    
    declare -a ARGS=(
        --container-image $CONTAINER_IMAGE
        --container-mounts $FSX_MOUNT
    )
fi

###########################
####### Torch Dist  #######
###########################

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=$GPUS_PER_NODE
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

export TORCHRUN=torchrun
export TRAIN_SCRIPT=${TRAIN_SCRIPT:=../src/train.py}

############################
# llama2_7b Training Params ##
############################

declare -a TRAINING_ARGS=(
    --max_context_width=4096
    --num_key_value_heads=32
    --intermediate_size=11008
    --hidden_width=4096
    --num_layers=32
    --num_heads=32
    --model_type=llama_v2
    --tokenizer=hf-internal-testing/llama-tokenizer
    --checkpoint_freq=1
    --validation_freq=0  # 검증 비활성화
    --max_steps=1        # 스텝 수 증가
    --checkpoint_dir=./checkpoints
    --dataset=allenai/c4
    --dataset_config_name=en
    # --resume_from_checkpoint=./checkpoints
    --train_batch_size=2
    --val_batch_size=1
    --sharding_strategy=full # https://pytorch.org/docs/stable/fsdp.html
    --offload_activations=0
)

AUTO_RESUME=""
# if [ -d "/opt/sagemaker_cluster" ]; then
#     echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
#     AUTO_RESUME="--auto-resume=1"
# fi
srun ${AUTO_RESUME} -l "${ARGS[@]}" ${TORCHRUN} "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"