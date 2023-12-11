export CUDA_VISIBLE_DEVICES=1
BLD_TRAINING=""
BLD_TRAINLIST="lists/blendedmvs/train.txt"
BLD_TESTLIST="lists/blendedmvs/val.txt"
BLD_CKPT_FILE=" "  # dtu pretrained model

exp=$1
PY_ARGS=${@:2}

BLD_LOG_DIR="./checkpoints/bld/"$exp 
if [ ! -d $BLD_LOG_DIR ]; then
    mkdir -p $BLD_LOG_DIR
fi

python -m torch.distributed.launch --master_port 12345 --nproc_per_node=1 train_bld.py --logdir $BLD_LOG_DIR --dataset=blendedmvs --batch_size=1 --trainpath=$BLD_TRAINING --summary_freq 100 --loadckpt $BLD_CKPT_FILE\
        --ndepths 8,8,4,4 --depth_inter_r 0.5,0.5,0.5,0.5 --group_cor --inverse_depth --rt --attn_temp 2 --trainlist $BLD_TRAINLIST --testlist $BLD_TESTLIST  $PY_ARGS | tee -a $BLD_LOG_DIR/log.txt
