# final project of Deep Learning for Computer Vision 
This is our implementation for ECCV 2024 challenge (Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving).

# Inference
## environment setup
```bash
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install datasets
pip install gdown
pip install peft==0.10.0

cd ..
```
## download checkpoint
```bash
cd LLaVA/
mkdir checkpoints
cd checkpoints/
gdown https://drive.google.com/uc?id=1mKMK8jlDky92hySVefMn1AgPTyLUNmA1
unzip llava-v1.5-7b-task-lora.zip
cd ../..
```
## run inference code
```bash 
bash inference.sh $1
```
$1: specified GPU id  
the predicted json file will be saved in "./output.json"

# Train
## install packages
```bash
pip uninstall torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install flash-attn --no-build-isolation
pip uninstall numpy
pip install numpy==1.26.4
#pip install --upgrade deepspeed
#pip uninstall deepspeed
#pip install deepspeed==0.12.6
#pip uninstall accelerate
#pip install accelerate==0.21.0
```
## download dataset
```
mkdir -p dataset/train/images
mkdir -p dataset/val/images
python3 download_dataset_image.py --split train
python3 download_dataset_image.py --split val
```
the training images will be saved in "./dataset/train/images/"
## run training code
To run `train.sh`, you must specify at least two GPU IDs. The IDs should ideally start from `0` and be consecutive. 
```bash
CUDA_VISIBLE_DEVICES=$1,$2 bash train.sh
```
$1, $2: specified gpu ids  
For example:
```bash
CUDA_VISIBLE_DEVICES=0,1 bash train.sh
```
the checkpoint will be saved in "./LLaVA/checkpoints/llava-v1.5-7b-task-lora/"

# Preprocess

## object detection & depth estimation
### environment setup
Install packages:
```bash
cd GroundingDINO/
pip install -e .
cd ..
```
### Download checkpoint
GroundingDINO:
```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
mv groundingdino_swinb_cogcoor.pth ./GroundingDINO/
```
Depth-Anything-v2:
```bash
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
mv depth_anything_v2_metric_hypersim_vitl.pth ./Depth_Anything_V2/
```
### generate detection and depth results
generate detection_depth_info/test_B4_median_int.json
```bash
python3 preprocess.py --split test
```
## RAG
#### Environment setup
```
pip install faiss
cd rag
```
#### Generate RAG results (only id)
```
python3 faiss_id_results.py
```
#### Generate RAG results with gt and example outputs
```
python3 js_rag_full.py
```
## merge json file
Combine RAG results and object detection information and depth estimation text
```
python3 js_combine_rag_detect.py -s test -q all -r cos -d median -v 3
cd ..
```
After finishing all these preprocess steps, you can finally get the same test prompt json as the one we uploaded (dataset/test/combined_prompt_v3_test_all_cos_median.json)
