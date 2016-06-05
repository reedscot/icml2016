. CONFIG

checkpoint_dir=${CHECKPOINT_DIR} \
net_gen=coco_fast_t70_nc3_nt128_nz100_bs64_cls0.5_ngf196_ndf196_100_net_G.t7 \
net_txt=${COCO_NET_TXT} \
queries=scripts/coco_queries.txt \
dataset=coco \
th txt2img_demo.lua

