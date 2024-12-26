import datasets
import os 
import json
from PIL import Image

import argparse
import torch

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

from tqdm import tqdm

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

    # load model
    model_name = get_model_name_from_path(args.lora_model)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.lora_model, args.model_base, model_name
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

    # iterate through all the images
    # set tqdm with total 900 samples
    for item in tqdm(dataset, total=900):
        # get id and image
        id = item['id']
        image = item['image']
        # read prompt from json file
        d = data[id]
        # query prompt (skip <image>\n )
        qs = d["conversations"][0]["value"][8:]


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
        # print(f"ID: {id}, Output: {outputs}")
        results[id] = outputs

    # save output
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Output saved at {args.output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lora", "--lora_model", type=str, default="./LLaVA/checkpoints/llava-v1.5-7b-task-lora", help="Path to the LoRA checkpoint")
    parser.add_argument("-base", "--model_base", type=str, default="liuhaotian/llava-v1.5-7b", help="Base model")
    parser.add_argument("-q", "--query_json", type=str, default="./dataset/test/combined_prompt_v3_test_all_cos_median.json", help="Path to the query prompt json file")
    parser.add_argument("-o", "--output_path", type=str, default="./output.json", help="Path to the output json file")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    eval_model(args)
