from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
import sys
import os
sys.path.append(os.path.abspath("GroundingDINO"))
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from GroundingDINO.groundingdino.datasets import transforms as T
import numpy as np
from torchvision.ops import box_convert
import torch
import json
import argparse
import glob
import matplotlib
import torch
import argparse
import copy
from torchvision.transforms import functional as F

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"
    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    category_colors = {
        "car": (255, 0, 0),         # Bright Red
        "truck": (75, 0, 130),  # Indigo
        "bus": (0, 0, 255),         # Bright Blue
        "bicycle": (255, 130, 0),  # Red-Orange
        "motorcycle": (255, 128, 0),# Bright Orange
        "pedestrian": (153, 50, 204), # Purple
        "cyclist": (0, 255, 255),   # Cyan
        "motorcyclist": (0, 128, 255), # Dodger Blue
        "barrier": (0, 255, 127), # Spring Green
        "traffic cone": (255, 255, 0),   # Bright Yellow
        "traffic light": (255, 100, 255), # pink purple
        "traffic sign": (0, 255, 0),       # Bright Green
        "dustbin": (255, 223, 0),   # Bright Gold
        "animal": (138, 43, 226),   # Blue Violet
        "machinery": (50, 205, 50)  # Lime Green
    }

    for box, label in zip(boxes, labels):
        label = label.split("(")[0]
        category_name = str(label) 
        color = category_colors.get(category_name, (255, 255, 255))

        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)#3
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
    return image_pil, mask


def plot_depth_to_image(object_depth, image_bgr, depth_ranges, depth_values, center_x, center_y):
    if object_depth <= 5.5:
        depth_range = "immediate"
        color = (0, 0, 255)
    elif object_depth <= 10:
        depth_range = "short_range"
        color = (0, 255, 255)
    elif object_depth <= 20:
        depth_range = "mid_range"
        color = (0, 255, 0)
    else:
        depth_range = "long_range"
        color = (255, 0, 0)
    
    depth_ranges.append(depth_range)
    depth_values.append(float(object_depth))
    text = f"{object_depth:.1f}"
    cv2.putText(image_bgr, text, (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

def format_output(output_dir, detection_data, depth_data, split):
    supercategory_mapping = {
        "car": "vehicles",
        "truck": "vehicles",
        "bus": "vehicles",
        "bicycle": "vehicles",
        "motorcycle": "vehicles",
        "pedestrian": "vulnerable_road_users",
        "cyclist": "vulnerable_road_users",
        "motorcyclist": "vulnerable_road_users",
        "traffic cone": "traffic cones",
        "traffic light": "traffic lights",
        "traffic sign": "traffic signs",
        "barrier": "barriers",
        "dustbin": "miscellaneous",
        "animal": "miscellaneous"
    }

    object_dict_template = {
        "vehicles": {},
        "vulnerable_road_users": {},
        "traffic signs": {},
        "traffic lights": {},
        "traffic cones": {},
        "barriers": {},
        "miscellaneous": {}
    }

    all_images_data = {}

    for image_name, detections in detection_data.items():
        boxes = detections['boxes']
        categories = detections['categories']
        ranges = depth_data[image_name]['depth_ranges']
        
        image_object_dict = copy.deepcopy(object_dict_template)

        for i, category in enumerate(categories):
            supercategory = supercategory_mapping.get(category, "miscellaneous")        
            if category not in image_object_dict[supercategory]:
                image_object_dict[supercategory][category] = []        
            rounded_box = [round(coord) for coord in boxes[i]]
            object_info = {
                "bounding_box": rounded_box,
                "range": ranges[i]
            }
            image_object_dict[supercategory][category].append(object_info)

        all_images_data[image_name] = image_object_dict

    output_path = os.path.join(output_dir, f"{split}_B4_median_int.json")
    with open(output_path, 'w') as output_file:
        json.dump(all_images_data, output_file, indent=4)

# Main pipeline
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection, Depth Estimation, and Data Aggregation')

    parser.add_argument('--input_size', type=int, default=720) #518
    parser.add_argument('--output_dir', type=str, default='./detection_depth_info/')
    parser.add_argument('--load-from', type=str, default='./Depth_Anything_V2/depth_anything_v2_metric_hypersim_vitl.pth') #depth_anything_v2_metric_vkitti_vitl.pth
    parser.add_argument('--max-depth', type=float, default=35) #20
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--visualize', action='store_true', help="Set this flag to enable visualization")
    parser.add_argument('--split', type=str, default="test")
    args = parser.parse_args()

    split = args.split
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    visualize = args.visualize
    if visualize:
        os.makedirs(os.path.join(output_dir, f"{split}_median"), exist_ok=True)
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grounding_model = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "./GroundingDINO/groundingdino_swinb_cogcoor.pth")
    TEXT_PROMPT = "car. truck. bus. bicycle. motorcycle. pedestrian. cyclist. motorcyclist. barrier. traffic cone. traffic light. traffic sign. dustbin. animal. machinery"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    encoder='vitl'
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()
    
    # Load dataset
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=True)

    detection_data = {}
    depth_data = {}

    for sample in dataset:
        image = sample['image']
        image_name = sample['id']
        print(image_name)
        image_np = np.array(image)

        # Object detection
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image, None)
        boxes, logits, phrases = predict(
            model=grounding_model,
            image=image_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            remove_combined=True
        )
        h, w, _ = image_np.shape
        boxes_scaled = boxes * torch.tensor([w, h, w, h])
        xyxy_boxes = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        scores = logits.cpu().numpy()
        phrases = [phrase for phrase in phrases]
        pred_phrases = [f"{phrases[i]}({logits[i]:.2f})"
            for i in range(len(phrases))]
        size = image.size
        pred_dict = {
            "boxes": xyxy_boxes,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        detection_data[image_name] = {
            "boxes": xyxy_boxes.tolist(),
            "categories": phrases,
            "scores": scores.tolist()
        }
        image_with_box = plot_boxes_to_image(image, pred_dict)[0]
        del image, boxes, logits, phrases, scores, pred_phrases

        # Depth estimation
        depth_image = depth_anything.infer_image(image_np, args.input_size)
        depth_ranges = []
        depth_values = []
        image_bgr = cv2.cvtColor(np.array(image_with_box), cv2.COLOR_RGB2BGR)
        for box in xyxy_boxes:
            # 取 center 深度當作 object 深度
            center_x = (box[0]+box[2])/2
            center_y = (box[1]+box[3])/2
            # 取平均深度
            x0, y0, x1, y1 = map(int, box)  
            box_depth_values = depth_image[y0:y1, x0:x1] 
            valid_depths = box_depth_values[box_depth_values > 0]  # 過濾掉無效深度值 (e.g., 0)
            # 取中位數
            if valid_depths.size > 0:
                object_depth = np.median(valid_depths)  # 使用中位數
            plot_depth_to_image(object_depth, image_bgr, depth_ranges, depth_values, center_x, center_y)
            
        depth_data[image_name] = {
            "depth_ranges": depth_ranges,
            "depth_values": depth_values
        }
        if visualize:
            cv2.imwrite(os.path.join(output_dir, f"{image_name}.png"), image_bgr)

    format_output(output_dir, detection_data, depth_data, split)
