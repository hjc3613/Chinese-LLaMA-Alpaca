from tqdm import tqdm
from os import path, listdir
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import os
import torch
import json
import time
import pandas as pd
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
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)


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
        sentence = sentence[len(text):]
        
        print("山海：" + sentence.strip('</s>'))  
        print('\n' + '--'*40 + '\n')
        return sentence

    def iter_generate(self, cur):
        ...

    def process(self, row, type):
        if type=='common':
            return self.generate(row['input'])
        elif type=='iter':
            return self.iter_generate(row['当前对话'])
        else:
            raise Exception('type 传值错误')

def main(args):
    interface = DecodeInterface(args.model_path, args.tokenizer_name)
    if args.file.endswith('.xlsx'):
        df = pd.read_excel(args.file)
    elif args.file.endswith('.jsonl'):
        df = pd.read_json(args.file, lines=True)
    else:
        raise Exception('只支持xlsx和jsonl文件')
    output_file = f'{args.file.split(".")[0]}_{args.output_suffix}.xlsx'
    result = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row = dict(row)
        res = interface.process(row, type='common')
        row['pred_output'] = res
        result.append({**row})
    result = pd.DataFrame.from_dict(result)
    result['gold_output'] = result['output']
    result.drop('output', axis=1, inplace=True)
    if 'id' not in result.columns:
        result['id'] = result['record_id']+'_'+result['round']
    result.to_excel(output_file)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--file', required=True, type=str)
    parser.add_argument('--output_suffix', required=False, default='result')
    parser.add_argument('--tokenizer_name', required=False, default=None)
    args = parser.parse_args()
    main(args)