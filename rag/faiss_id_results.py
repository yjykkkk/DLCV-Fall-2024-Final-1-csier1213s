import faiss
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch
import json
import datasets

def main(split):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # settings
    db_type = 'cos'
    k = 2  if split == 'test' else 3

    # set ViT
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
    # prepare dataset
    if split != 'test':
        dataset_1 = datasets.load_dataset("ntudlcv/dlcv_2024_final1", split = 'train', streaming = True)
        dataset_2 = datasets.load_dataset("ntudlcv/dlcv_2024_final1", split = 'val', streaming = True)
        data_loaders = [dataset_1, dataset_2]
    else:
        dataset_1 = datasets.load_dataset("ntudlcv/dlcv_2024_final1", split = 'test', streaming = True)
        data_loaders = [dataset_1]

    # initialize faiss
    faiss.omp_set_num_threads(1)

    # general db
    db_gen = faiss.read_index(f"db/general_{db_type}.index")
    json_path_gen = 'db/general.json'
    with open(json_path_gen) as json_file:
        index_dict_gen = json.load(json_file)

    # suggestion db
    db_sug = faiss.read_index(f"db/suggestion_{db_type}.index")
    json_path_sug = 'db/suggestion.json'
    with open(json_path_sug) as json_file:
        index_dict_sug = json.load(json_file)

    # set output path
    result_json_path = f'results/{split}_id_{db_type}.json'
    result_dict = {}

    # read datas and query
    for dataset in data_loaders:
        for item in tqdm(dataset):
            id = item['id']
            question_type = id.split('_')[1]
        
            if question_type == 'general':
                faiss_db = db_gen
                index_dict = index_dict_gen
            elif question_type == 'suggestion':
                faiss_db = db_sug
                index_dict = index_dict_sug
            else:
                continue   
            
            # open the image
            image = item['image']
                
            #  get image embeddings
            with torch.no_grad():
                inputs = processor(text="", images=image, return_tensors="pt", padding=True).to(device)
                image_embeddings = model(**inputs).vision_model_output.pooler_output
                image_embeddings = image_embeddings.cpu().detach().numpy()

            # get top k results
            distance, index = faiss_db.search(image_embeddings,k)
            
            # store results to dict
            example_ids = [index_dict[str(index[0][i])] for i in range(k)]
            result_dict[id] = example_ids
            
    with open(result_json_path, 'w') as result_json:
        json.dump(result_dict, result_json, indent=4)
        
if __name__ == '__main__':
    main('test')