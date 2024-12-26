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
cd ..
```
## download checkpoint
```

```
## run inference code
```bash 
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

## object detection & depth estimation
### environment setup
Make sure the environment variable `CUDA_HOME` is set.
```bash
echo $CUDA_HOME
```
If it print nothing, then it means you haven't set up the path/

Run this so the environment variable will be set under current shell. 
```bash
export CUDA_HOME=/path/to/cuda-[your_version]
```
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