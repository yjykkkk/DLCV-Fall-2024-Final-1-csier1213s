# Inference
## environment setup
```
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
## download checkpoint
```

```
## run inference code
```
bash inference.sh $1 $2 $3
```
$1: Path to gt image folder
$2: Path to annot file
$3: Path to predicted file

# Train
```
bash train.sh 
```
the checkpoint will be saved in "./LLaVA/checkpoints/llava-v1.5-7b-task-lora/"

# Preprocess
## merge json file

## object detection & depth estimation
### load  GroundindDINO checkpoint
```
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
mv groundingdino_swinb_cogcoor.pth ./GroundingDINO/
```
### load Depth-Anything-v2 checkpoint
```
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
mv depth_anything_v2_metric_hypersim_vitl.pth ./Depth_Anything_V2/
```
### run preprocess code
```
python3 preprocess.py
```
## RAG