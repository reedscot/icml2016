. CONFIG

ID=3
BS=64
NC=3
NT=128
NZ=100
CLS=0.5
NGF=196
NDF=196

display_id=10${ID} \
gpu=${ID} \
dataset='coco' \
batchSize=${BS} \
numCaption=${NC} \
name="coco_nc${NC}_nt${NT}_nz${NZ}_bs${BS}_cls${CLS}_ngf${NGF}_ndf${NDF}" \
cls_weight=${CLS} \
niter=200 \
nz=${NZ} \
nt=${NT} \
img_dir=${COCO_IMG_DIR} \
data_root=${COCO_META_DIR} \
lr_decay=0.5 \
decay_every=40 \
nThreads=12 \
checkpoint_dir=${CHECKPOINT_DIR} \
print_every=4 \
save_every=5 \
use_cudnn=1 \
ngf=${NGF} \
ndf=${NDF} \
th main_cls.lua

