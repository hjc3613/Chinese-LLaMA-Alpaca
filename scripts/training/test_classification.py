from transformers import BertForSequenceClassification, BertTokenizer
from run_classification import SentenceBert
import json
import torch
from tqdm import tqdm
import pandas as pd

model_path = r'/data/hujunchao/models/real_time_summary_k_dia_match/checkpoint-240/'
file = r'/data/hujunchao/record_gen/gpt4_continue_gen_new/k_dia_match/持续生成_测试集_3_k_dia_match.jsonl'

model = SentenceBert.from_pretrained(model_path)
model.eval()
tokenizer = BertTokenizer.from_pretrained(model_path)
def test_file():
    with open(file) as f:
        datas = [json.loads(i) for i in f.readlines() if i.strip()]
    for item in tqdm(datas):
        text = item['input'][-512:]
        inputs = tokenizer(text, return_tensors ='pt')
        with torch.no_grad():
            output = model(**inputs)
            # label = output.logits.argmax().detach().item()
            label = output.logits.detach().item()
            item['predict'] = label
    pd.DataFrame.from_dict(datas).to_excel(file.replace('.jsonl', '_result.xlsx'))

def test_file():
    with open(file) as f:
        datas = [json.loads(i) for i in f.readlines() if i.strip()]
    for item in tqdm(datas):
        texta = item['inputa'][-512:]
        textb = item['inputb'][-512:]
        inputs_a = tokenizer(texta, return_tensors ='pt')
        inputs_b = tokenizer(textb, return_tensors ='pt')
        inputs = {
            'input_ids_a':inputs_a['input_ids'], 
            'input_ids_b':inputs_b['input_ids'], 
            'attention_mask_a':inputs_a['attention_mask'],
            'attention_mask_b':inputs_b['attention_mask'],
            }
        with torch.no_grad():
            output = model(**inputs)
            # label = output.logits.argmax().detach().item()
            label = output.logits.detach().item()
            item['predict'] = label
    pd.DataFrame.from_dict(datas).to_excel(file.replace('.jsonl', '_result.xlsx'))

if __name__ == '__main__':
    test_file()