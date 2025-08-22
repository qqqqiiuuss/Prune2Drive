import argparse
import math
import json
import os
import time

import torch
# 注意：这里不再需要 torch.distributed 和 DDP
import torch.profiler

# ====== 根据项目需要的其他 import ======
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images
from llava.train.train import preprocess_llama3
# ============================================\
import language_evaluation
   
view_params = {
    'view_1': 0.15, 
    'view_2': 0.15,
    'view_3': 0.15,
    'view_4': 0.15,
    'view_5': 0.15,
    'view_6': 0.15
}
import nni
import random
def eval_acc(answers, GTs):
    scores = []
    for i in range(len(answers)):
        answer = answers[i]
        GT = GTs[i]
        if answer == GT:
            scores.append(1.0)
        else:
            scores.append(0.0)

    scores = sum(scores) / len(scores)
    return scores


def eval_language_score(answer, GT):
    language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
    results = language_eval.run_evaluation(answer, GT)

    score = 0
    for idx, key in enumerate(results.keys()):
        if idx < 4:
            score += results[key] / 4. / 3.
        elif idx == 4:
            score += results[key] / 3.
        else:
            score += results[key] / 10. / 3.

    return score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="../test_llama.json", help='path to test data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--multimodal', action='store_true', default=True)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.data, 'r', encoding='utf-8') as f:
        data_all = json.load(f)
    data_subset = data_all[500:1000]
    
    llava_model_args = {"multimodal": args.multimodal}
    tokenizer, base_model, image_processor, max_length = load_pretrained_model(
        "../../ckpt/DriveMM",
        None,
        "llama",
        device_map=device,
        **llava_model_args
    )
    base_model.eval()
    model = base_model.to(device)

    results = []

    time1 = time.time()
    language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
    params = nni.get_next_parameter()  #todo 默认不使用nni返回空值
    view_params.update(params)

    view_importance = [view_params['view_1'], view_params['view_2'], view_params['view_3'], view_params['view_4'], view_params['view_5'], view_params['view_6']]
    view_importance_values = view_importance
    answers = []
    gts = []
    with torch.inference_mode():
        for idx, data_item in enumerate(data_subset):
            filenames = data_item['image']   # list[str], 多张图像
            question = data_item['conversations'][0]['value']
            gt_answer = data_item['conversations'][1]['value']
            rec_id = data_item['id']
            filenames = [path.replace('nuscenes', 'nuscenes_train') for path in filenames]
            images = [Image.open(str(url)).convert("RGB") for url in filenames]
            image_tensors = process_images(images, image_processor, model.config)
            image_tensors = [img.to(device=device, dtype=torch.float16) for img in image_tensors]
            image_sizes = [img.size for img in image_tensors]

            multi_prompt = (
                '1: <image> 2: <image> 3: <image> 4: <image> 5: <image> 6: <image>. '
                'These six images are the front view, front left view, front right view, '
                'back view, back left view and back right view of the ego vehicle.'
            )
            question = multi_prompt + "<image>" + question[8:] + "<image>"

            sources = [[{"from": 'human', "value": question}, {"from": 'gpt', "value": ''}]]
            input_id = preprocess_llama3(sources, tokenizer, has_image=True)['input_ids'][:, :-1].to(device)

            modalities = ['image'] * len(image_tensors)

            
            
            fastv_redundancy_config = {
                "fastv_k": 3,
                
                "view_importance": view_importance_values,
                "alpha": 0.5,
                "fastv_random": True,  
                "fastv_topkall": False, 
                "vis_token": False, 
                "sparsevlm": False,  
                "cos_similarity":False,
                "text_vis_cos": False,  
                "redundancy":True,
                "farwaypoint": True,
                "get_texttoken_start": True
            }
            model.config.fastv_config = fastv_redundancy_config
            print(view_importance_values)
            cont = model.generate(
                input_id,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=1024,
                modalities=modalities,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            pred_answer = text_outputs[0]
            
            answers.append(pred_answer)
            gts.append(gt_answer)

    answers_acc = []
    gts_acc = []
    
    answers_language = []     
    gts_language = []   
    for item in answers:
        if isinstance(item, str) and len(item) == 1 and item.isalpha():
            answers_acc.append(item)
        else:
            answers_language.append(item)
        
    for item in gts:
        if isinstance(item, str) and len(item) == 1 and item.isalpha():
            gts_acc.append(item)
        else:
            gts_language.append(item)
    language_score = eval_language_score(answers_language, gts_language)
    accuracy_score = eval_acc(answers_acc, gts_acc)
    efficiency_score = sum(view_importance)
    
    language_weight = 0.5
    accuracy_weight = 0.5
    efficiency_weight = -0.03
    
    score = language_weight * language_score +  accuracy_weight * accuracy_score + efficiency_weight * efficiency_score
    print(score)
    nni.report_final_result(score)
    print("Processing completed.")


if __name__ == "__main__":
    main()