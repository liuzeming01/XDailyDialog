DSTORE_SIZE=4081197
#MODEL_PATH=/ProjectRoot/XDailyDialog/knn-chat/checkpoints/checkpoint_best.pt
#DATA_PATH=/path/to/fairseq_preprocessed_data_path
#DATASTORE_PATH=/path/to/save_datastore
#PROJECT_PATH=/path/to/ada_knnmt

#MODEL_PATH=/ProjectRoot/XDailyDialog/knn-chat/checkpoints/checkpoint_last.pt
MODEL_PATH=/ProjectRoot/XDailyDialog/knn-chat/wmt19.en-de.ffn8192.pt
#DATA_PATH=/ProjectRoot/XDailyDialog/knn-chat/output/monolingual_En/
DATA_PATH=/ProjectRoot/XDailyDialog/knn-chat/data-bin/it

DATASTORE_PATH=./saved_datastore
PROJECT_PATH=/ProjectRoot/adaptive-knn-mt

mkdir -p $DATASTORE_PATH

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/save_datastore.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH \
    --max-tokens 4096 \
    --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 1024 \
    --dstore-fp16 \
    --dstore-size $DSTORE_SIZE \
    --dstore-mmap $DATASTORE_PATH
