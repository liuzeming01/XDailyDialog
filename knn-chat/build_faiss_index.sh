PROJECT_PATH=/ProjectRoot/adaptive-knn-mt
DSTORE_PATH=./saved_datastore
DSTORE_SIZE=4081197

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/train_datastore_gpu.py \
  --dstore_mmap $DSTORE_PATH \
  --dstore_size $DSTORE_SIZE \
  --dstore-fp16 \
  --faiss_index ${DSTORE_PATH}/knn_index \
  --ncentroids 4096 \
  --probe 32 \
  --dimension 1024
