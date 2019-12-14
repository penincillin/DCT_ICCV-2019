log_dir=log
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

train_log_dir=log/train_logs
if [ ! -d $train_log_dir ]; then
    mkdir $train_log_dir
fi

curr_date=$(date +'%m_%d_%H_%M') 
log_file="./log/train_logs/$curr_date.log"

visdom_port=8097
port_offset=5000
dist_port=$(expr $visdom_port - $port_offset)
batch_size=128

data_root=./dct_data/datasets/
model_root=./dct_data/models/

human36m_anno_path=$data_root'human36m/annotation/train.pkl'
coco_anno_path=$data_root'coco/annotation/train.pkl' 
up3d_anno_path=$data_root'up_3d/annotation/train.pkl' 

CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch  \
    --nproc_per_node=2 --master_port=$dist_port src/train_dist.py --dist \
    --two_branch --main_encoder resnet50 --aux_encoder resnet18 \
    --pretrained_weights $model_root'weights/img_iuv_res50_res18.pth' \
    --batchSize $batch_size  --lr_e 1e-4 \
    --data_root $data_root --model_root $model_root \
    --human36m_anno_path  $human36m_anno_path \
    --coco_anno_path  $coco_anno_path \
    --up3d_anno_path $up3d_anno_path \
    --train_coco --train_up3d \
    --refine_IUV --up3d_use3d \
    --loss_3d_weight 10 --dp_align_loss_weight 1  --kp_loss_weight 10 \
    --display_port $visdom_port  2>&1 | tee $log_file