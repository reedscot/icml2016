. CONFIG

checkpoint_dir=${CHECKPOINT_DIR} \
net_gen=cub_v2_nc4_cls0.5_int1.0_ngf128_ndf64_600_net_G.t7 \
net_txt=${CUB_NET_TXT} \
queries=scripts/cub_queries.txt \
dataset=cub \
th txt2img_demo.lua

