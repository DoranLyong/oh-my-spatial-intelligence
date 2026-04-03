# Setup

## Prerequisites

* A linux 64-bits architecture (i.e. `linux-64` platform in `mamba info`).
* A NVIDIA GPU with at least 32 Gb of VRAM.



## 1.1. Setup Python Environment (for RTX 5080 + CUDA 12.8)

* https://github.com/facebookresearch/sam-3d-objects/issues/29
* https://gist.github.com/luffy-yu/3c9708aaf446d3640ef843c927ad9952

The following will install the default environment. If you use `conda` instead of `mamba`, replace its name in the first two lines. Note that you may have to build the environment on a compute node with GPU (e.g., you may get a `RuntimeError: Not compiled with GPU support` error when running certain parts of the code that use Pytorch3D).

```bash
#-- create sam3d-objects environment
mamba env create -f environments/default.yml
mamba activate sam3d-objects

pip install uv 

#-- for pytorch/cuda dependencies
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu128"


#-- install sam3d-objects and core dependencies
pip install -e '.[dev]'

#-- pytorch3d dependency on pytorch is broken, this 2-step approach solves it
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 제거 (CUDA 확장들이 PyTorch 번들 CUDA로 컴파일)
export FORCE_CUDA=1

uv pip install "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9" --no-build-isolation --no-cache-dir

#-- for inference
uv pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html --no-build-isolation --no-cache-dir
uv pip install gsplat==1.5.3 --no-build-isolation --no-cache-dir

uv pip install seaborn==0.13.2 --no-build-isolation
uv pip install gradio==5.49.0 --no-build-isolation 

#-- nvdiffrast
uv pip install "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0" --no-build-isolation

#-- Diff Gaussian Rasterization
git clone https://github.com/autonomousvision/mip-splatting.git
cd mip-splatting/submodules/diff-gaussian-rasterization/
FORCE_CUDA=1 MAX_JOBS=8 python setup.py install
cd ../../.. && rm -rf mip-splatting
```
## (Option) Attention Module 
['sdpa'](../sam3d_objects/model/backbone/tdfy_dit/modules/attention/__init__.py) is default. If you needed, install things below:
```bash
MAX_JOBS=2 uv pip install flash-attn==2.8.2 --no-build-isolation
uv pip install xformers==0.0.32.post2
```


## (Issue) System CUDA Version != PyTorch CUDA Version
* PyTorch2.x 는 CUDA 를 자체 포함하고 있음 
* `python demo.py` 를 실행할때, 시스템 CUDA 버전과 PyTorch가 불일치하면 `Cannot load symbol cublasLtGetVersion` 에러 발생 

해결 방법
* 아래 처럼 시스템 CUDA 경로를 삭제후 실행하기 (가장 쉬우면서 안전함)
```bash
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 삭제 
python demo.py
```



## 2. Getting Checkpoints

### From HuggingFace

⚠️ Before using SAM 3D Objects, please request access to the checkpoints on the SAM 3D Objects
Hugging Face [repo](https://huggingface.co/facebook/sam-3d-objects). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token).

⚠️ SAM 3D Objects is available via HuggingFace globally, **except** in comprehensively sanctioned jurisdictions.
Sanctioned jurisdiction will result in requests being **rejected**.

```bash
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```


