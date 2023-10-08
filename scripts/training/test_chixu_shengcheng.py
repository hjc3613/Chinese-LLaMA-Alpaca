from tqdm import tqdm
from os import path, listdir
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import os
import torch
import json
import time
import pandas as pd
import re
from collections import defaultdict
from uni_gpt.modeling_unigpt import UniGPTForCausalLM
from baichuan.tokenization_baichuan import BaichuanTokenizer
from argparse import ArgumentParser
time_prefix = time.strftime("%Y%m%d",time.localtime(time.time())) 



# hf_model_path = '/data/hujunchao/models/learn_gpt4_continue_gen_no_blank/checkpoint-25'
# tokenizer = LlamaTokenizer.from_pretrained(hf_model_path)
# model = UniGPTForCausalLM.from_pretrained(hf_model_path)
# model = UniGPTForCausalLM.from_pretrained(
# hf_model_path,
# torch_dtype=torch.bfloat16,
# low_cpu_mem_usage=True,
# device_map="auto",
# load_in_8bit=False
# )

def to_jsonl(dct_lst, path):
    with open(path, encoding='utf8', mode='w') as f:
        dct_lst = [json.dumps(i, ensure_ascii=False) for i in dct_lst]
        f.write('\n'.join(dct_lst)+'\n')

class DecodeInterface:
    def __init__(self, hf_model_path, tokenizer_name=None) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=False,
            trust_remote_code=True
        )

        if tokenizer_name=='llama':
            self.tokenizer = LlamaTokenizer.from_pretrained(hf_model_path)
        elif tokenizer_name== 'baichuan':
            self.tokenizer = BaichuanTokenizer.from_pretrained(hf_model_path)
        elif tokenizer_name=='unigpt':
            self.tokenizer = LlamaTokenizer.from_pretrained(hf_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True, use_fast=False)

        self.cache = defaultdict(list)

    def generate(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        print('######################### input #################################')
        print(text)
        generation_output = self.model.generate(**inputs,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        #max_length=1948,
                                        max_new_tokens=512,
                                        do_sample=False,
                                        early_stopping=True,
                                        #top_p = 0.6,
                                        num_beams=1,
                                        #repetition_penalty=2.0,
                                        #eos_token_id=tokenizer.eos_token_id,
                                        num_return_sequences = 1)

        sentence = self.tokenizer.decode(generation_output.sequences[0]).replace('<s>', '').replace('</s>', '')
        print('######################### output ###############################')
        sentence = sentence[len(text):].strip().strip(':')
        
        print("山海：" + sentence.strip('</s>'))  
        print('\n' + '--'*40 + '\n')
        return sentence

    def format_cache_to_input(self, record_id, window=10):
        caches_of_record_id = self.cache.get(record_id, [])
        pre_summary = '历史所有结论:' + '\n'.join(i[2] for i in caches_of_record_id[:-window] if not re.search(r'当前对话中无法得到确定性信息',i[2]))
        pre_summary = re.sub(r'\n+', '\n', pre_summary)
        result = []
        result.append(pre_summary)
        for round, pre_diag, cur_summary in caches_of_record_id[-window:]:
            result.append(f'{pre_diag}\n结论:\n{cur_summary or "当前对话中无法得到确定性信息"}')
        return '\n'.join(result)
    
    def iter_generate(self, cur, record_id, round, date):
        inputs = self.format_cache_to_input(record_id, window=15)
        inputs = f'就诊日期:{date}\n{inputs}\n{cur}\n结论:\n'
        res = self.generate(inputs)
        self.cache[record_id].append((round, cur, res))
        return res, inputs

    def process(self, row, type):
        if type=='common':
            return self.generate(row['input']), row['input']
        elif type=='iter':
            return self.iter_generate(row['当前对话'], row['record_id'], row['round'], row['admission_date'])
        else:
            raise Exception('type 传值错误')
        
def process_dir(root):
    excels = [i for i in os.listdir(root) if i.endswith('.xlsx') and '预标' not in i][:]
    interface = DecodeInterface(
        hf_model_path='/data/hujunchao/models/insert_summary_baichun_15_newdata/checkpoint-180'
    )
    for excel in tqdm(excels):
        path = os.path.join(root, excel)
        df = pd.read_excel(path)
        result = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            row = dict(row)
            if 'round' not in row:
                row['round'] = idx
            res, inputs = interface.process(row, type='iter')
            if '无法得到确定性信息' in res:
                res = ''
            row['过程摘要'] = res
            result.append({**row})
            
        result = pd.DataFrame.from_dict(result)
        result.to_excel(path.replace('.xlsx', '_预标.xlsx'))

def main(args):
    interface = DecodeInterface(args.model_path, args.tokenizer_name)
    if args.file.endswith('.xlsx'):
        df = pd.read_excel(args.file)
    elif args.file.endswith('.jsonl'):
        df = pd.read_json(args.file, lines=True)
    else:
        raise Exception('只支持xlsx和jsonl文件')
    if 'admission_date' not in df:
        df['admission_date'] = df['input'].str.findall('(?<=就诊日期:)(.+?)\n').str[0]
    output_file_excel = f'{args.file.rsplit(".", maxsplit=1)[0]}_{args.decode_type}.xlsx'
    output_file_jsonl = f'{args.file.rsplit(".", maxsplit=1)[0]}_{args.decode_type}.jsonl'
    print(f'ready to save to {output_file_excel} and {output_file_jsonl}')
    result = []
    processed_record_num = set()
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        processed_record_num.add(row['record_id'])
        if len(processed_record_num) > int(args.record_nums):
            break
        row = dict(row)
        res, inputs = interface.process(row, type=args.decode_type)
        if '无法得到确定性信息' in res:
            res = ''
        row['pred_output'] = res
        row['input_new'] = inputs
        result.append({**row})
        
    result = pd.DataFrame.from_dict(result)
    result['gold_output'] = result['output']
    result.drop('output', axis=1, inplace=True)
    if 'id' not in result.columns:
        result['id'] = result['record_id']+'_'+result['round'].astype(str)
    print(f'save to {output_file_excel}')
    result.to_excel(output_file_excel)
    print(f'save to {output_file_jsonl}')
    to_jsonl(result.to_dict(orient='records'), output_file_jsonl)
        
if __name__ == '__main__':
    # process_dir('/data/hujunchao/record_gen/gpt4_continue_gen_new/pre_label/待标注数据_tmp')
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--file', required=True, type=str)
    parser.add_argument('--tokenizer_name', required=False, default=None)
    parser.add_argument('--decode_type', required=False, default='common')
    parser.add_argument('--record_nums', required=False, default=float('inf'))
    args = parser.parse_args()
    main(args)