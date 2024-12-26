import json
import os
import argparse
def results_to_text(original_input, key_split, key_type, example_1, example_2, 
                       detect_info, detect_info_1 , detect_info_2,
                       general_result, previous_output):
    
    if key_split == 'Train':
        if key_type == 'general': # train general
            text = ( 
                '<image>\n'
                'Task: General Perception' +
                f'\n{original_input}'
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
                'This example illustrates the expected level of detail and structure. ' 
                'Your goal is to replicate this structure ' 
                'by providing accurate object descriptions with a clear focus on appearance, position, direction, ' +
                'and the impact each object has on the ego car\'s driving behavior, specifically noting the object\'s category. ' 
                
                'And the following is the detection result for the TARGET IMAGE, '
                f'which will help you provide a description for the image: '
                f'\n- Detection Result: {detect_info}'
                f'\n- Description: '
            )
        elif key_type == 'suggestion': # train suggestion
            text = (
                '<image>\n'
                'Task: Driving Suggestion' +
                f'\n{original_input}'
                
                '\nBefore analyzing the target image, '
                'carefully study the following example driving suggestion. ' 
        
                '\nExample: '
                f'\n- Example Driving Suggestion: {example_1} '
                
                '\nThis example is provided solely to '
                'help you understand the structure, depth, and detail expected '
                'in a driving suggestion. '
                
                f'\nNow, it is your turn to provide a driving suggestion for the TARGET IMAGE. ' 
                'Focus entirely on the TARGET IMAGE and its specific conditions. '
                'Your response should be as comprehensive and '
                'detailed as the example provided, thoroughly addressing '
                'all relevant factors.'
                
                'And the following is the general perception for the TARGET IMAGE, '
                'which will help you understand the senario in the TARGET IMAGE, '
                'including each object\'s appearance, position, direction, and its impact on the ego car\'s behavior.'
                f'This will help you provide a driving suggestion for the image: '
                f'\n- General Perception Result: {general_result}'
                f'\n- Driving Suggestion: '
            )
        else: # train regional
            text = (
                '<image>\n'
                'Task: Regional Perception' +
                f'\n{original_input}'
            )
    elif key_split == 'Test':        
        if key_type == 'general': # test general
            text = ( 
                '<image>\n'
                'Task: General Perception' +
                f'\n{original_input}'
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
                'This example illustrates the expected level of detail and structure. ' 
                'Your goal is to replicate this structure ' 
                'by providing accurate object descriptions with a clear focus on appearance, position, direction, ' +
                'and the impact each object has on the ego car\'s driving behavior, specifically noting the object\'s category. ' 
                
                'And the following is the detection result for the TARGET IMAGE, '
                f'which will help you provide a description for the image: '
                f'\n- Detection Result: {detect_info}'
                f'\n- Description: '
            )
        elif key_type == 'suggestion': # test suggestion
            text = (
                '<image>\n'
                'Task: Driving Suggestion'
                f'\n{original_input}'
            )   
        else: # test regional
            text = (
                '<image>\n'
                'Task: Regional Perception'
                f'\n{original_input}'
            )
    else: # for self-reflection (val set)
        
        if key_type == 'general': # val general
            
            text = ( 
                '<image>\n'
                'Task: Self-Reflection for General Perception'
                            
                '\nYour goal is to refine the previous output to provide a more comprehensive analysis of the image. '
                'Before refining your generation of the TARGET IMAGE, carefully review the original task description and your previous output. '
                'Focus on improving the accuracy and clarity of your outputs. '
                f'\n- Original Task: {original_input}'
                f'\n- Previous Output: {previous_output}'
                
                f'\nNow, it is your turn to provide a refined description for the TARGET IMAGE. ' 
                'The following is the detection result for the TARGET IMAGE, '
                'each object is categorized with details including its bounding box [x1, y1, x2, y2] within the image size (width, height) = (1355, 720), '
                'range (immediate, short_range, mid_range, long_range), and category/subcategory (e.g., vehicles: car, truck). '
                'This will help you provide a refined description for the image. '
                f'\n- Detection Result: {detect_info}'
                f'\n- Refined Description: '
            )
        elif key_type == 'suggestion': # val suggestion
            text = ( 
                '<image>\n'
                'Task: Self-Reflection for Driving Suggestion' 
                
                '\nYour goal is to refine the previous output to provide a more comprehensive analysis of the image. '
                'Before refining your generation of the TARGET IMAGE, carefully review the original task description and your previous output. '
                'Focus on improving the accuracy and clarity of your outputs. '
                f'\n- Original Task: {original_input}'
                f'\n- Previous Output: {previous_output} '
                
                f'\nNow, it is your turn to provide a refined driving suggestion for the TARGET IMAGE. '
                'The following is the general perception for the TARGET IMAGE, '
                'which will help you understand the senario in the TARGET IMAGE, '
                'including each object\'s appearance, position, direction, and its impact on the ego car\'s behavior.'
                f'This will help you provide a driving suggestion for the image: '
                f'\n- General Perception Result: {general_result}'
                f'\n- Refined Driving Suggestion: '
            )
        else: # val regional
            text =(
                '<image>\n'  
                'Task: Self-Reflection for Regional Perception'
                
                '\nYour goal is to refine the previous output to provide a more comprehensive analysis of the image. '
                'Before refining your generation of the TARGET IMAGE, carefully review the original task description and your previous output. '
                'Focus on improving the accuracy and clarity of your outputs. '
                f'\n- Original Task: {original_input}'
                f'\n- Previous Output: {previous_output}'
                f'\nNow, it is your turn to provide a refined output for the TARGET IMAGE. '
                f'\n- Refined Output:'
            )


    return text

def print_info(rag_path, output_path):
    print('\nRag   :', rag_path)
    print('Output:', output_path)

def main(rag_path, detect_dir, output_path, split, ques_type, detect_type,
         val_inf_path, version):
    output_dict = []
    
    with open(rag_path) as file:
        rag_dict = json.load(file)
    
    if split == 'Val' or split == 'Trainandval':
        with open(val_inf_path) as file:
            previous_dict = json.load(file)
    
    # find all int detection dicts of the desired type (median, center, mean)
    detect_dict = {}
    print('Detect dict files:', end=' ')
    for path in os.listdir(detect_dir):
        if path.endswith('int.json'):
            if path.split('_')[2] == detect_type:
                print(path, end=' ')
                with open(os.path.join(detect_dir, path)) as file:
                    detect_dict.update(json.load(file))
    print()
    
    color_dict = {}
    print('Color dict files:', end=' ')
    for path in os.listdir(detect_dir):
        if path.endswith('color_v5.json'):
            print(path, end=' ')
            with open(os.path.join(detect_dir, path)) as file:
                color_dict.update(json.load(file))
    
    print_info(rag_path, output_path)
    
    # combine the prompts
    for key in rag_dict.keys():
        key_split = key.split('_')[0]
        key_type = key.split('_')[1]

        # should only output the desired split
        if (split != 'Trainandval') and (split != key_split):
            continue
        elif (split == 'Trainandval') and (key_split == 'Test'):
            continue
        
        # should only output the desired question type
        if (ques_type != 'all') and (ques_type != key_type):
            continue
        
        # only general and suggestion have examples and detection info
        if key_type != 'regional':
            id_1 = rag_dict[key]['examples'][0]['id']
            id_2 = rag_dict[key]['examples'][1]['id']
            example_1 = rag_dict[key]['examples'][0]['gt'][1]['value']
            example_2 = rag_dict[key]['examples'][1]['gt'][1]['value']
            
            if not (version == '10' and key_type == 'suggestion'):
                detect_info = json.dumps(detect_dict[key])
                detect_info_1 = json.dumps(detect_dict[id_1])
                detect_info_2 = json.dumps(detect_dict[id_2]) 
            else:
                detect_info = json.dumps(color_dict[key])
                detect_info_1 = json.dumps(color_dict[id_1])
                detect_info_2 = json.dumps(color_dict[id_2]) 
                      
        else:
            example_1 = ''
            example_2 = ''
            detect_info = ''
            detect_info_1 = ''
            detect_info_2 = ''
    
        # need general results for suggestion prompt
        if key_type == 'suggestion' and key_split != 'Test':
            general_result = rag_dict[key.replace('suggestion', 'general')]['gt'][1]['value']
        else:
            general_result = ''
        
        # need previous output for validation prompt
        if key_split == 'Val':
            previous_output = previous_dict[key]
        else:
            previous_output = ''
            
        original_input = rag_dict[key]['gt'][0]['value'].split('<image>\n')[1]
        
        # get combined prompt
        combined_prompt = results_to_text(original_input, key_split, key_type, 
                                            example_1, example_2, 
                                            detect_info, detect_info_1 , detect_info_2,
                                            general_result, previous_output)
        
        # no gt output for testset
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
    parser.add_argument('-s', help='Dataset split', choices=['train','val','test','trainandval'], required=True)
    parser.add_argument('-q', help='Question Type', choices=['all','general','regional','suggestion'], required=True)
    parser.add_argument('-r', help='RAG Query Type', choices=['cos','ip','l2'],required=True)
    parser.add_argument('-d', help='Detection Type', choices=['median','center','mean'], required=True)
    parser.add_argument('-v', required=True, help='Template Version')
    parser.add_argument('-c', action='store_true', help='Whether to check manually')
    args = parser.parse_args()
    
    rag_name = 'test' if args.s == 'test' else 'all'
    split = args.s[0].upper() + args.s[1:]
    
    main(
        # TODO: change the path of rag full result file
        rag_path=f'results/rag_{rag_name}_{args.r}.json',
        # TODO: change the directory of detection info, all detection info file should be under the directory
        detect_dir='../detection_depth_info',
        # TODO: change the output path of this combined result
        output_path=f'combined_prompt_v{args.v}_relfection_{args.q}_{args.r}_{args.d}.json',
        split=split,
        ques_type=args.q,
        detect_type=args.d,
        val_inf_path = 'reflection/val_inference.json',
        version = args.v
    )