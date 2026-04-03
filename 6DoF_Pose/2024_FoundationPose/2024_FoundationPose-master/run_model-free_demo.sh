#!/bin/bash
set -e  # 오류 발생 시 스크립트 즉시 중단

echo "Initializing environment..."
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 삭제 

eval "$(mamba shell hook --shell bash)"
mamba activate foundationpose

export CUDA_VISIBLE_DEVICES="0"

# 1. 내 모니터 화면 번호 지정 (보통 :0 아니면 :1); 
# pyrender.OffscreenRenderer(self.W, self.H) 사용할 때 필요 -- "OffscreenRenderer야, 내 모니터(:0)의 GPU 자원을 갖다 써라"
#export DISPLAY=:1

# 2. 렌더링 방식을 '윈도우 시스템(GLX)'으로 설정 (모니터가 있으니까 EGL 말고 GLX 사용)
#export PYOPENGL_PLATFORM=egl


# >> Data prepare  for ref_view_dir 
YCBV_REF_DIR=/home/cvipl-ubuntu/Workspace/datasets/6DoF/ycbv/ref_views_16
#python bundlesdf/run_nerf.py --ref_view_dir $YCBV_REF_DIR --dataset ycbv




# >> Then run model-free demo         
YCBV_DIR=/home/cvipl-ubuntu/Workspace/datasets/6DoF/ycbv                   
python run_ycb_video.py --ycbv_dir $YCBV_DIR --use_reconstructed_mesh 1 --ref_view_dir $YCBV_REF_DIR