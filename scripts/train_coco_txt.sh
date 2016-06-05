. CONFIG

ID=1
BS=64
CNN=512
ENC=gru18

display_id=10${ID} \
display=0 \
gpu=${ID} \
dataset='coco_txt' \
batchSize=${BS} \
encoder=${ENC} \
name="coco_${ENC}_bs${BS}_c${CNN}" \
cnn_dim=${CNN} \
niter=200 \
lr_decay=0.5 \
lr=0.0002 \
decay_every=50 \
img_dir=${COCO_IMG_DIR} \
data_root=${COCO_META_DIR} \
nThreads=6 \
checkpoint_dir=${CHECKPOINT_DIR} \
print_every=4 \
save_every=5 \
use_cudnn=1 \
th main_txt_coco.lua

