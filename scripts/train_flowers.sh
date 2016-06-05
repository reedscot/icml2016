. CONFIG

ID=2
GPU=3
NC=4
CLS=0.5
INT=1.0
NGF=128
NDF=64

display_id=10${ID} \
gpu=${ID} \
dataset="flowers" \
name="flowers_v2_nc${NC}_cls${CLS}_int${INT}_ngf${NGF}_ndf${NDF}" \
cls_weight=${CLS} \
interp_weight=${INT} \
interp_type=1 \
niter=600 \
nz=100 \
lr_decay=0.5 \
decay_every=100 \
img_dir=${FLOWERS_IMG_DIR} \
data_root=${FLOWERS_META_DIR} \
classnames=${FLOWERS_META_DIR}/allclasses.txt \
trainids=${FLOWERS_META_DIR}/trainvalids.txt \
init_t=${FLOWERS_NET_TXT} \
nThreads=6 \
checkpoint_dir=${CHECKPOINT_DIR} \
numCaption=${NC} \
print_every=4 \
save_every=100 \
replicate=0 \
use_cudnn=1 \
ngf=${NGF} \
ndf=${NDF} \
th main_cls_int.lua

