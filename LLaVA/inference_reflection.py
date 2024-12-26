import argparse
import torch
import datasets

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import re

import os
import json
from tqdm import tqdm

# prompt generation functions
from inf_aux_v9 import read_dicts,  get_self_reflection_prompt


def image_parser(image_file):
    out = image_file.split(args.sep)
    return out


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.lora_model)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.lora_model, args.model_base, model_name, load_8bit=args.load_8bit, load_4bit=args.load_4bit
    )
    tokenizer.model_max_length = 4096

    # device
    print(f"Device: {model.device}")


    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode


    # load prompt json file
    with open (args.query_json, "r") as f:
        data = json.load(f)
    # turn list of dict into dict
    data = {d["id"]: d for d in data}

    # load streaming dataset
    dataset = datasets.load_dataset("ntudlcv/dlcv_2024_final1", split = "test", streaming = True)
    
    # create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # output
    results = {}

    # trim data for debugging
    if args.debug:
        print(" ========== DEBUG MODE ENABLED ========== ")
        num_data = 3
        data = data[:num_data] + data[300:300+num_data] + data[600:600+num_data]

    # inference decode function
    def inference_decode(image, qs):

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = [image]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs


    # init prompt functions
    print("Reading dicts...")
    rag_dict, detect_dict = read_dicts(rag_path=args.rag_path, detect_dir=args.detect_dir, detect_type=args.detect_type, version=args.prompt_version)

    # Inference Test_general_ first because suggestion needs the general output

    # iterate through all the images
    for item in tqdm(dataset, total=900):

        # get id and image
        id = item["id"]
        d = data[id]

        # determine type of the prompt
        type = None
        if (id.startswith("Test_general_")):
            type = 1
        elif (id.startswith("Test_regional_)")):
            type = 2
        elif (id.startswith("Test_suggestion_")):
            type = 3


        # query prompt (skip <image>\n )
        qs = d["conversations"][0]["value"][8:]

        # generate output
        image = item['image']
        outputs = inference_decode(image, qs)
        # print(f"BEFORE: \n{outputs}")


        # reflection rounds
        for _ in range(args.reflection_rounds):

            # get_reflection_prompt(id: string, previous_output: string, version: int) -> string:
            qs = get_self_reflection_prompt(id=id, previous_output=outputs, rag_dict=rag_dict, detect_dict=detect_dict, version=args.prompt_version)
            qs = qs[8:] # remove <image>\n

            # generate output
            outputs = inference_decode(image, qs)
            # print(f"AFTER: \n{outputs}")


        # store to results
        results[id] = outputs
        # print(f"ID: {id}, Output: {outputs}")


    # save output
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Output saved at {args.output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lora", "--lora_model", type=str, default="./LLaVA/checkpoints/llava-v1.5-7b-task-lora", help="Path to the LoRA checkpoint")
    parser.add_argument("-base", "--model_base", type=str, default="liuhaotian/llava-v1.5-7b", help="Base model")
    parser.add_argument("-i", "--image_path", type=str, default="./dataset/test/images", help="Path to the image directory")
    parser.add_argument("-q", "--query_json", type=str, default="./dataset/test/combined_prompt_v9_test_all_cos_median.json", help="Path to the query prompt json file")
    parser.add_argument("-o", "--output_path", type=str, default="./output_reflection.json", help="Path to the output json file")
    parser.add_argument("-ver", "--prompt_version", type=int, default=9, help="Prompt version")
    parser.add_argument("-rag", "--rag_path", type=str, default="./rag/results/rag_test_cos.json", help="Path to the rag_test_cos.json file")
    parser.add_argument("-ddir", "--detect_dir", type=str, default="./detection_depth_info", help="Path to the directory containing all int detect info files")
    parser.add_argument("-dtype", "--detect_type", type=str, default="median", choices=["median", "mean", "center"], help="desired detection type (median, center, mean)")
    parser.add_argument("-rround", "--reflection_rounds", type=int, default=1, help="Number of self-reflection rounds")
    parser.add_argument("--load_8bit", action="store_true", help="Load model with 8-bit quantization")
    parser.add_argument("--load_4bit", action="store_true", help="Load model with 4-bit quantization")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Requires: inf_aux_v9.py in the same directory, and path to `rag_test_cos.json` and path to `detect_info` directory

    eval_model(args)
