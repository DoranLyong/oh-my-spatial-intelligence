#!/bin/bash
set -e  # 오류 발생 시 스크립트 즉시 중단

echo "Initializing environment..."
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 삭제 
eval "$(mamba shell hook --shell bash)"
mamba activate sam3d-objects

echo "Running inference..."
python run_inference.py --input_path ./data/tomato --mask_prompt tomatos

echo "Inference completed successfully. Please check the results."