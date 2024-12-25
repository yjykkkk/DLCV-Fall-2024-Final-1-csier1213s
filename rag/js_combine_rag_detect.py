import json
import os
import argparse
def results_to_text(original_input, key_type, example_1, example_2, detect_info, detect_info_1 , detect_info_2):
    
    if key_type == 'general':
        text = (
            f'{original_input}'
            '\nBefore analyzing the target image, '
            'carefully study the following example detection result and its corresponding outputs. ' 
            'About the structure of detection result, '
            'each object is categorized with details including its bounding box [x1, y1, x2, y2] within the image size (width, height) = (1355, 720), '
            'range (immediate, short_range, mid_range, long_range), and category/subcategory (e.g., vehicles: car, truck). '
            'This helps describe the object\'s appearance, position, direction, and its impact on the ego car\'s behavior.'
                    
            '\nExample:'
            f'\n- Detection Result: {detect_info_1}'
            f'\n- Example Description: {example_1} '
            
            f'\nNow, it is your turn to provide a description of the TARGET IMAGE. ' 
            'Focus entirely on the TARGET IMAGE and its specific conditions. '
            'This example illustrates the expected level of detail and structure. ' +
            'Your goal is to replicate this structure ' +
            'by providing accurate object descriptions with a clear focus on appearance, position, direction, ' +
            'and the impact each object has on the ego car\'s driving behavior, specifically noting the object\'s category. ' 
            
            ' And the following are the detection result of the TARGET IMAGE, '
            f'which will help you to provide a description of the image: '
            f'\n- Detection Result: {detect_info}'
            f'\n- Description: '
        )
    elif key_type == 'suggestion':
        text = (
            f'{original_input}'
            '\nBefore analyzing the target image, '
            'carefully study the following example detection result and its corresponding outputs. ' 
            'About the structure of detection result, '
            'each object is categorized with details including its bounding box [x1, y1, x2, y2] within the image size (width, height) = (1355, 720), '
            'range (immediate, short_range, mid_range, long_range), and category/subcategory (e.g., vehicles: car, truck). '
            'This helps understand the object\'s appearance, position, direction, and its impact on the ego car\'s behavior.'
    
            '\nExample:'
            f'\n- Detection Result: {detect_info_1}'
            f'\n- Example Driving Suggestion: {example_1} '
            
            '\nThis example is provided solely to '
            'help you understand the structure, depth, and detail expected '
            'in a driving suggestion. '
            
            f'\nNow, it is your turn to provide a driving suggestion for the TARGET IMAGE. ' 
            'Focus entirely on the TARGET IMAGE and its specific conditions. '
            'Your response should be as comprehensive and '
            'detailed as the example provided, thoroughly addressing '
            'all relevant factors.'
            
            ' And the following are the detection result of the TARGET IMAGE, '
            f'which will help you to provide a driving suggestion of the image: '
            f'\n- Detection Result: {detect_info}'
            f'\n- Driving Suggestion: '
        )
    else:
        text = f'{original_input}'
   
    return text

def print_info(rag_path, output_path):
    print('\nRag   :', rag_path)
    print('Output:', output_path)

def main(rag_path, detect_dir , output_path, split, ques_type, detect_type, version):
    output_dict = []
    
    with open(rag_path) as file_1:
        rag_dict = json.load(file_1)
    
    detect_dict = {}
    print('Detect:', end=' ')
    if version == '3' or version == '7':
        for path in os.listdir(detect_dir):
            if path.endswith('int.json'):
                if path.split('_')[2] == detect_type:
                    print(path, end=' ')
                    with open(os.path.join(detect_dir, path)) as file:
                        detect_dict.update(json.load(file))
    else:
        for path in os.listdir(detect_dir):
            if path.endswith(f'color_v{version}.json'):
                print(path, end=' ')
                with open(os.path.join(detect_dir, path)) as file:
                    detect_dict.update(json.load(file))
    
    print_info(rag_path, output_path)
    
    for key in rag_dict.keys():
        key_split = key.split('_')[0]
        key_type = key.split('_')[1]
      
        if (split != 'Trainandval') and (split != key_split):
            continue
        elif (split == 'Trainandval') and (key_split == 'Test'):
            continue
        
        if (ques_type != 'all') and (ques_type != key_type):
            continue
        
        if key_type != 'regional':
            id_1 = rag_dict[key]['examples'][0]['id']
            id_2 = rag_dict[key]['examples'][1]['id']
            example_1 = rag_dict[key]['examples'][0]['gt'][1]['value']
            example_2 = rag_dict[key]['examples'][1]['gt'][1]['value']
            detect_info = json.dumps(detect_dict[key])
            detect_info_1 = json.dumps(detect_dict[id_1])
            detect_info_2 = json.dumps(detect_dict[id_2])
        else:
            example_1 = ''
            example_2 = ''
            detect_info = ''
            detect_info_1 = ''
            detect_info_2 = ''
        
        original_input = rag_dict[key]['gt'][0]['value']
        
        combined_prompt = results_to_text(original_input, key_type, example_1, example_2, detect_info, detect_info_1 , detect_info_2)
        
        
        if split != 'Test':
            output_dict.append({
                'id': key,
                'image': f'{key}.png',
                'conversations':[
                    {
                        'from': 'human',
                        'value': combined_prompt
                    },
                    {
                        "from": "gpt",
                        'value': rag_dict[key]['gt'][1]['value']
                    }
                ]
            })
        else:
            output_dict.append({
                'id': key,
                'image': f'{key}.png',
                'conversations':[
                    {
                        'from': 'human',
                        'value': combined_prompt
                    }
                ]
            })
            
    with open(output_path,'w') as file:
        json.dump(output_dict, file, indent=4)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', required=True, help='Dataset split')
    parser.add_argument('-q', required=True, help='Question Type')
    parser.add_argument('-r', required=True, help='RAG Query Type')
    parser.add_argument('-d', required=True, help='Detection Type')
    parser.add_argument('-v', required=True, help='Template Version')
    args = parser.parse_args()
    
    rag_name = 'test' if args.s == 'test' else 'all'
    split = args.s[0].upper() + args.s[1:]
    
    main(
        rag_path=f'results/rag_{rag_name}_{args.r}.json',
        detect_dir='../detection_depth_info',
        output_path=f'../dataset/test/combined_prompt_v{args.v}_{args.s}_{args.q}_{args.r}_{args.d}.json',
        split=split,
        ques_type=args.q,
        detect_type=args.d,
        version = args.v
    )