DSTORE_SIZE=4081197
DATA_PATH=/ProjectRoot/XDailyDialog/knn-chat/data-bin/it
PROJECT_PATH=/ProjectRoot/adaptive-knn-mt
MODEL_PATH=/ProjectRoot/XDailyDialog/knn-chat/wmt19.en-de.ffn8192.pt
DATASTORE_PATH=./saved_datastore

max_k_grid=(4 8 16 32)
batch_size_grid=(32 32 32 32)
update_freq_grid=(1 1 1 1)
valid_batch_size_grid=(32 32 32 32)

for idx in ${!max_k_grid[*]}
do

  MODEL_RECORD_PATH=./save/model/train-hid32-maxk${max_k_grid[$idx]}
  TRAINING_RECORD_PATH=./save/tensorboard/train-hid32-maxk${max_k_grid[$idx]}
  mkdir -p "$TRAINING_RECORD_PATH"

  CUDA_VISIBLE_DEVICES=0 python \
  $PROJECT_PATH/fairseq_cli/train.py \
  $DATA_PATH \
  --log-interval 100 --log-format simple \
  --arch transformer_wmt19_de_en_with_datastore \
  --tensorboard-logdir "$TRAINING_RECORD_PATH" \
  --save-dir "$MODEL_RECORD_PATH" --restore-file "$MODEL_PATH" \
  --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
  --validate-interval-updates 100 --save-interval-updates 100 --keep-interval-updates 1 --max-update 5000 --validate-after-updates 1000 \
  --save-interval 10000 --validate-interval 100 \
  --keep-best-checkpoints 1 --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
  --train-subset valid --valid-subset valid --source-lang de --target-lang en \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.001 \
  --max-source-positions 1024 --max-target-positions 1024 \
  --batch-size "${batch_size_grid[$idx]}" --update-freq "${update_freq_grid[$idx]}" --batch-size-valid "${valid_batch_size_grid[$idx]}" \
  --task translation \
  --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --min-lr 3e-05 --lr 0.0003 --clip-norm 1.0 \
  --lr-scheduler reduce_lr_on_plateau --lr-patience 5 --lr-shrink 0.5 \
  --patience 30 --max-epoch 500 \
  --load-knn-datastore --dstore-filename $DATASTORE_PATH --use-knn-datastore \
  --dstore-fp16 --dstore-size $DSTORE_SIZE --probe 32 \
  --knn-sim-func do_not_recomp_l2 \
  --use-gpu-to-search --move-dstore-to-mem --no-load-keys \
  --knn-lambda-type trainable --knn-temperature-type fix --knn-temperature-value 10 --only-train-knn-parameter \
  --knn-k-type trainable --k-lambda-net-hid-size 32 --k-lambda-net-dropout-rate 0.0 --max-k "${max_k_grid[$idx]}" --k "${max_k_grid[$idx]}" \
  --label-count-as-feature
done
