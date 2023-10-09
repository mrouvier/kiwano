#Prepare Voxceleb1
python3 local/download_voxceleb1.py db/voxceleb1/
python3 local/prepare_voxceleb1.py db/voxceleb1/ data/voxceleb1/

#Prepare Voxceleb2
python3 local/download_voxceleb2.py db/voxceleb2/
python3 local/prepare_voxceleb2.py db/voxceleb2/ data/voxceleb2/

#Prepare MUSAN
python3 local/download_musan.py db/musan/
python3 local/prepare_musan.py db/musan/ data/musan/

#Prepare RIRS NOISES
#python3 local/download_rirs_noises.py db/rirs_noises/
#python3 local/prepare_rirs_noises.py db/musan/ data/rirs_noises


#Train resnet
export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))


export OMP_NUM_THREADS=10
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1


python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK utils/train_resnet_ddp.py


#python3 utils/extract_resnet.py exp/model3500.mat > exp/xvector.txt
