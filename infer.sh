batch_size=8

model_dir=DCT_data/models/
img_list_file='DCT_data/demo/img_list.txt'
trained_model_path=$model_dir'weights/img_res50.pth'

CUDA_VISIBLE_DEVICES=0 python3 infer.py  --gpu_ids=0 \
    --single_branch --main_encoder resnet50 \
    --infer_dataset_path $img_list_file \
    --batchSize $batch_size --phase test \
    --pretrained_models_dir $model_dir \
    --trained_model_path $trained_model_path \
    2>&1 | tee test_log.txt

python2 -m util.evaluator $model_dir