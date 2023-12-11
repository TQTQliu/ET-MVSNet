export CUDA_VISIBLE_DEVICES=1
TNT_TESTPATH="./DATA/tanksandtemples/advanced/"
TNT_TESTLIST="lists/tnt/adv.txt"
TNT_CKPT_FILE=" " # fine-tuned model

exp=$1
PY_ARGS=${@:2}

TNT_LOG_DIR="./checkpoints/tnt/"$exp 
if [ ! -d $TNT_LOG_DIR ]; then
    mkdir -p $TNT_LOG_DIR
fi

TNT_OUT_DIR="./outputs/tnt/"$exp
if [ ! -d $TNT_OUT_DIR ]; then
    mkdir -p $TNT_OUT_DIR
fi

python test_dypcd_tnt_adv.py --dataset=tanks --batch_size=1 --testpath=$TNT_TESTPATH  --testlist=$TNT_TESTLIST --loadckpt $TNT_CKPT_FILE --interval_scale 1.06 --outdir $TNT_OUT_DIR\
            --ndepths 16,8,4,4 --depth_inter_r 0.5,0.5,0.5,0.5  --num_view=11  --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $TNT_LOG_DIR/log_test.txt
