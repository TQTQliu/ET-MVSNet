export CUDA_VISIBLE_DEVICES=0
DTU_TRAINING=" "
DTU_TRAINLIST="lists/dtu/train.txt"
DTU_TESTLIST="lists/dtu/test.txt"

exp=$1
PY_ARGS=${@:2}

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi

python -m torch.distributed.launch --nproc_per_node=1 train_dtu.py --logdir $DTU_LOG_DIR --dataset=dtu_yao4 --batch_size=4 --trainpath=$DTU_TRAINING --summary_freq 100 \
        --depth_inter_r 0.5,0.5,0.5,0.5 --group_cor --inverse_depth --rt --attn_temp 2 --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt



