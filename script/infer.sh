batch_size=8

data_root=./dct_data/
model_root=./dct_data/models/

img_list_file=$data_root'demo/img_list.txt'
trained_model_path=$data_root'models/trained_models/img_res50.pth'

CUDA_VISIBLE_DEVICES=0 python3 src/infer.py  --gpu_ids=0 \
    --single_branch --main_encoder resnet50 \
    --infer_dataset_path $img_list_file \
    --data_root $data_root  --model_root $model_root \
    --batchSize $batch_size --phase test \
    --trained_model_path $trained_model_path