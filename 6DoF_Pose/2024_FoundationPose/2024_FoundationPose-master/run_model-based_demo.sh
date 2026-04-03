#!/bin/bash
set -e  # 오류 발생 시 스크립트 즉시 중단

echo "Initializing environment..."
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 삭제 

eval "$(mamba shell hook --shell bash)"
mamba activate foundationpose

export CUDA_VISIBLE_DEVICES="0"

#python run_demo.py


# >> Run on public datasets 
LINEMOD_DIR=/home/cvipl-ubuntu/Workspace/datasets/6DoF/linemod
python run_linemod.py --linemod_dir $LINEMOD_DIR --use_reconstructed_mesh 0


#YCBV_DIR=/home/cvipl-ubuntu/Workspace/datasets/6DoF/ycbv
#python run_ycb_video.py --ycbv_dir $YCBV_DIR --use_reconstructed_mesh 0
