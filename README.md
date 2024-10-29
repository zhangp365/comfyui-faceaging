# Face Aging

### Description
- This is a comfyui custom node version of [Age Transformation](https://github.com/yuval-alaluf/SAM)
- You could transform age of target from -100 to +100(but recommended -50 to +100)

### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3

### How to use
1. **clone this repo to `ComfyUI/custom_nodes` folder**
2. **download models(`sam_ffhq_aging.pt` & `shape_predictor_68_face_landmarks.dat`) to `ComfyUI/models/face_aging`**
```bash
mkdir -p ComfyUI/models/face_aging
pip install gdown
gdown "https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download" -O pretrained_models/sam_ffhq_aging.pt
wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
```
3. **run the comfyUI and use it like this**
![Image](/workflow_example/workflow.png)
