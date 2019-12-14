log_dir=log
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

test_log_dir=log/test_logs
if [ ! -d $test_log_dir ]; then
    mkdir $test_log_dir
fi

test_dataset=up3d
log_file="./log/test_logs/test_log_$test_dataset.log"

batch_size=256

data_root=./dct_data/datasets/
model_root=./dct_data/models/

coco_anno_path=$data_root'coco/annotation/test.pkl' 
up3d_anno_path=$data_root'up_3d/annotation/test.pkl' 


CUDA_VISIBLE_DEVICES=1 python3 src/test.py  --gpu_ids=0 \
    --two_branch --main_encoder resnet50 --aux_encoder resnet18 \
    --data_root $data_root --model_root $model_root \
    --coco_anno_path $coco_anno_path \
    --up3d_anno_path $up3d_anno_path \
    --batchSize $batch_size --phase test \
    --test_dataset  $test_dataset \
    2>&1 | tee $log_file 