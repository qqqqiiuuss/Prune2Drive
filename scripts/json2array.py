
import json
import argparse

def convert_lines_to_array(input_file, output_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                # 空行就跳过
                continue
            obj = json.loads(line)
            data.append(obj)

    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert line-based JSON to a JSON array.")
    parser.add_argument("--input", help="Path to the input JSON lines file")
    parser.add_argument("--output",  help="Path to the output JSON array file")
    args = parser.parse_args()
    
    # 使用命令行传入的参数
    convert_lines_to_array(args.input, args.output)