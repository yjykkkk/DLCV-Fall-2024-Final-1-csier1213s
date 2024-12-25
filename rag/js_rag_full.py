import json
import argparse

def get_full_results(dataset_dict, example_dict, key, example_1, example_2):
    
    return {
        'gt': dataset_dict[key],
        'examples':[
            {
            'id': example_1,
            'gt': example_dict[example_1]
            },
            {
            'id': example_2,
            'gt': example_dict[example_2]
            }
        ]
    }
    

def main(raw_id_result_path, testset, query_type):

    # json path of original dataset
    train_json_path = 'gt/train.json'
    val_json_path = 'gt/val.json'
    test_json_path = 'gt/test.json'
    
    # save path depends on whether testeset or not
    if testset:
        full_output_path = f'results/rag_test_{query_type}.json'
    else:
        id_output_path = f'results/rag_id_all_{query_type}.json'
        full_output_path = f'results/rag_all_{query_type}.json'
        
    # load raw id results
    with open(raw_id_result_path) as file:
        results = json.load(file)
        
    # load example dict
    with open(train_json_path) as file:
        example_dict = json.load(file)
    with open(val_json_path) as file:
       example_dict.update(json.load(file))

    # load dataset dict
    if testset == True:
        with open(test_json_path) as file:
            dataset_dict = json.load(file)
    else:
        dataset_dict = example_dict.copy()
    
    # dicts to store outputs
    rag_full_dict = {}
    rag_id_dict = {}

    for key in dataset_dict.keys():
        
        key_type = key.split('_')[1]
        
        if key_type != 'regional':
            
            # example should not include itself
            if not testset:
                if key == results[key][0]:
                    example_1 = results[key][1]
                    example_2 = results[key][2]
                elif key == results[key][1]:
                    example_1 = results[key][0]
                    example_2 = results[key][2]
                elif key == results[key][2]:
                    example_1 = results[key][0]
                    example_2 = results[key][1]
                else:
                    print(key, results[key])
                    example_1 = results[key][0]
                    example_2 = results[key][1]
            else:
                example_1 = results[key][0]
                example_2 = results[key][1]
                
            # store noself id dicts
            rag_id_dict[key] = [example_1,example_2]
                
            # store full results
            rag_full_dict[key] = get_full_results(dataset_dict, example_dict, key, example_1, example_2)
        else:
            rag_full_dict[key] = {'gt': dataset_dict[key]}
  
    with open(full_output_path, 'w') as file:
        json.dump(rag_full_dict, file, indent=4)
    
    if not testset:
        with open(id_output_path, 'w') as file:
            json.dump(rag_id_dict, file, indent=4)

if __name__ == '__main__':
    main(
        raw_id_result_path='results/test_id_cos.json',
        testset=True,
        query_type='cos'
    )