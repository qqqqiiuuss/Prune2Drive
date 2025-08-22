import argparse
import math
import json
import os
import time

import torch
import torch.profiler

from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images
from llava.train.train import preprocess_llama3
    


def convert_lines_to_array(input_file, output_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)

    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="../test_llama.json", help='path to test data')
    parser.add_argument('--output', type=str, default="../output.jsonl", help='path to output file (JSONL)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--multimodal', action='store_true', default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.data, 'r', encoding='utf-8') as f:
        data_all = json.load(f)
    data_subset = data_all  

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
    chunk_size = 100  
    chunk_counter = 0
    time1 = time.time()

    with torch.inference_mode():
        for idx, data_item in enumerate(data_subset):
            filenames = data_item['image']   
            question = data_item['conversations'][0]['value']
            gt_answer = data_item['conversations'][1]['value']
            rec_id = data_item['id']

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
                "fastv_k": 1,

                "fastv_r":[0.095, 0.095, 0.095, 0.095, 0.095, 0.095],
                "fastv_random": True,  
                "farwaypoint": True,
                "get_texttoken_start": True,
                "prune_method": "farway"
            }
            model.config.fastv_config = fastv_redundancy_config


            cont = model.generate(
                input_id,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=1024,
                modalities=modalities,
            )
            # all_keep_indices = model.model.all_keep_indices
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            pred_answer = text_outputs[0]
            if "question_id" in data_item:
                results.append({
                    "question_id": data_item["question_id"],
                    "id": rec_id,
                    "question": question,
                    "answer": pred_answer
                })
            else:
                results.append({
                    "id": rec_id,
                    "question": question,
                    "answer": pred_answer
                })

            print(f"Processed record: {rec_id}")
            if (len(results) >= chunk_size) or (idx == len(data_subset) - 1):
                with open(args.output, "a", encoding="utf-8") as f:
                    for item in results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"[Chunk {chunk_counter}] Written {len(results)} items. Elapsed time: {time.time()-time1:.2f} sec")
                chunk_counter += 1
                results = []

    print("Processing completed.")


if __name__ == "__main__":
    main()