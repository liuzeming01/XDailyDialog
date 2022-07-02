#!/usr/bin/env bash
export PYTHONIOENCODING=utf8

export CUDA_VISIBLE_DEVICES=$1

WORKING_DIR=`pwd`

# CMD="pip install -r ${WORKING_DIR}/re.txt"
# echo ${CMD}
# ${CMD}

DATA_NAME=$2  # "crosslingual_en"
EPOCHS=5
BATCH_SIZE=4
EVAL_BATCH_SIZE=4
GRAD_ACC=2
LR=2e-5

MODEL="mt5-base"
# CKPT_PATH="${WORKING_DIR}/checkpoints/mt5-base/"
CKPT_PATH="google/mt5-base"
SAVE_PATH="${WORKING_DIR}/results/${DATA_NAME}/${MODEL}/ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_G${GRAD_ACC}"

OPTS=" --model_name_or_path ${CKPT_PATH} \
--data_path ${WORKING_DIR} \
--data_name ${DATA_NAME} \
--output_dir ${SAVE_PATH} \
--max_source_length 512 \
--max_target_length 200 \
--val_max_target_length 200 \
--do_train \
--do_eval \
--do_predict \
--num_train_epochs ${EPOCHS} \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
--gradient_accumulation_steps ${GRAD_ACC} \
--learning_rate ${LR} \
--logging_steps 500 \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_steps 500 \
--disable_tqdm False \
--load_best_model_at_end True \
--metric_for_best_model bleu-2 \
--save_total_limit 2"
# --max_train_samples 64"
# --max_eval_samples 64"
# --max_predict_samples 64"
CMD="python ${WORKING_DIR}/run_finetune.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} | tee ${SAVE_PATH}/train.log
rm -rf ${SAVE_PATH}/checkpoint*

Parameters in `run.bash`:
```
#!/usr/bin/env bash
export PYTHONIOENCODING=utf8

export CUDA_VISIBLE_DEVICES=$1

WORKING_DIR=/data2/yujie/andrew_data/daily/baseline

# CMD="pip install -r ${WORKING_DIR}/re.txt"
# echo ${CMD}
# ${CMD}

DATA_NAME=$2  # "crosslingual_en"
EPOCHS=5
BATCH_SIZE=4
EVAL_BATCH_SIZE=4
GRAD_ACC=2
LR=2e-5

MODEL="mt5-base"
# CKPT_PATH="${WORKING_DIR}/checkpoints/mt5-base/"
CKPT_PATH="google/mt5-base"
SAVE_PATH="${WORKING_DIR}/results/${DATA_NAME}/${MODEL}/ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_G${GRAD_ACC}"

OPTS=" --model_name_or_path ${CKPT_PATH} \
--data_path ${WORKING_DIR} \
--data_name ${DATA_NAME} \
--output_dir ${SAVE_PATH} \
--max_source_length 512 \
--max_target_length 200 \
--val_max_target_length 200 \
--do_train \
--do_eval \
--do_predict \
--num_train_epochs ${EPOCHS} \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
--gradient_accumulation_steps ${GRAD_ACC} \
--learning_rate ${LR} \
--logging_steps 500 \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_steps 500 \
--disable_tqdm False \
--load_best_model_at_end True \
--metric_for_best_model bleu-2 \
--save_total_limit 2"
```