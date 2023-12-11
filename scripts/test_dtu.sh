export CUDA_VISIBLE_DEVICES=1
DTU_TESTPATH=""
DTU_TESTLIST="lists/dtu/test.txt"
DTU_CKPT_FILE='' # dtu pretrained model

exp=$1
PY_ARGS=${@:2}

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_OUT_DIR="./outputs/dtu/"$exp
if [ ! -d $DTU_OUT_DIR ]; then
    mkdir -p $DTU_OUT_DIR
fi

python test_dtu_dypcd.py --dataset=general_eval4 --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR\
            --depth_inter_r 0.5,0.5,0.5,0.5 --conf 0.55 --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $DTU_LOG_DIR/log_test.txt

