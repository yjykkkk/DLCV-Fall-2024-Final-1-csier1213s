import json
import os

# run this before inference
def read_dicts(
    rag_path, # path to rag_test_cos.json
    detect_dir, # path to the directory having all int detect info files
    detect_type, # a string of the desired detection type (medain, center, mean)
    version # template version
    ):
    
    # load RAG full results
    with open(rag_path) as file:
        rag_dict = json.load(file)
        
    # find all int detection dicts of the desired type (median, center, mean)
    detect_dict = {}
    print('Detect dict files:', end=' ')
    for path in os.listdir(detect_dir):
        if path.endswith('_int.json'):
            if path.split('_')[2] == detect_type:
                print(path, end=' ')
                with open(os.path.join(detect_dir, path)) as file:
                    detect_dict.update(json.load(file))
    print()
    
    return rag_dict, detect_dict
    
def get_self_reflection_prompt(id, previous_output, rag_dict, detect_dict, version):
    key_type = id.split('_')[1] 
    original_input = rag_dict[id]['gt'][0]['value'].split('<image>\n')[1]
    detect_info = json.dumps(detect_dict[id])
    
    if version == 9:
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
                'The following is the detection result for the TARGET IMAGE, '
                'each object is categorized with details including its bounding box [x1, y1, x2, y2] within the image size (width, height) = (1355, 720), '
                'range (immediate, short_range, mid_range, long_range), and category/subcategory (e.g., vehicles: car, truck). '
                'This will help you provide a refined driving suggestion for the image. '
                f'\n- Detection Result: {detect_info}'
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

if __name__ == '__main__':
    pass