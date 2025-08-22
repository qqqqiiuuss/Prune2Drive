#!/bin/bash



CUDA_DEVICES=$1




# 创建 output/timestamp 文件夹
timestamp=$(TZ=Asia/Shanghai date +"%Y%m%d_%H%M%S")
output_dir="output/$2/$timestamp"
mkdir -p "$output_dir"


exec > >(tee -i "$output_dir/run_all.log") 2>&1
cp demo_drivelm_ddp_single.py "$output_dir"
cp "$0" "$output_dir/run_all.sh"


echo "Created folder: $output_dir"
echo "Will use GPUs: ${CUDA_DEVICES}"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
python demo_drivelm_ddp_single.py \
  --data ../test_llama_eval.json \
  --output "$output_dir/drivemm_original.json"

echo "Processing json2array..."
python json2array.py \
  --input "$output_dir/drivemm_original.json" \
  --output "$output_dir/drivemm_original_list.json"

echo "Preparing submission..."
python prepare_submission.py \
    --input "$output_dir/drivemm_original_list.json" \
    --output "$output_dir/submission.json"
    
echo "All steps finished!"