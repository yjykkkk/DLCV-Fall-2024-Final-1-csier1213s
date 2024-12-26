#!/bin/bash

python3 inference_reflection.py \
    --lora_model ./LLaVA/checkpoints/llava-v1.5-7b-task-lora \
    --image_path ./dataset/test/images \
    --query_json ./dataset/test/combined_prompt_v9_test_all_cos_median.json \
    --rag_path ./rag/results/rag_test_cos.json --detect_dir ./detection_depth_info --detect_type median \
    --reflection_rounds 1 --num_beams 1 \
    --output_path ./output_reflection.json \