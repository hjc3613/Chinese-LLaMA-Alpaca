
from tqdm import tqdm
from os import path, listdir
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BitsAndBytesConfig
import os
import torch
import json
import time
from uni_gpt.modeling_unigpt import UniGPTForCausalLM
time_prefix = time.strftime("%Y%m%d",time.localtime(time.time())) 

#os.environ["CUDA_VISIBLE_DEVICES"] = "8"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

#hf_model_path='/data/unisound/plms/uniift-13b-v0.0.1/hf'
#hf_model_path='/data/unisound/plms/unigpt-smart-13b-v0.0.1'
#hf_model_path='/data/unisound/plms/unigpt-single-disease-hf-v0.0.1'
hf_model_path = '/data/hujunchao/models/gpt4_continue_gen_new_zhaiyao_6_emptykey/checkpoint-120'
#data_dir = '/data/unisound/lixue/data/benchmark/MMLU_GoogleTranslate_process'
data_dir = '/data/hujunchao/record_gen/'
write_dir = data_dir
#data_dir = '/data/unisound/lixue/data/benchmark'
#write_dir = '/data/unisound/lixue/benchmark_log'


tokenizer = LlamaTokenizer.from_pretrained(hf_model_path)
# model = UniGPTForCausalLM.from_pretrained(hf_model_path)
model = UniGPTForCausalLM.from_pretrained(
hf_model_path,
torch_dtype=torch.bfloat16,
low_cpu_mem_usage=True,
device_map="auto",
load_in_8bit=False,
# rope_scaling = {"type": "linear", "factor": 2.0}
)

#model = LlamaForCausalLM.from_pretrained(
#        hf_model_path,
#        load_in_4bit=True,
#        device_map='auto',
#        quantization_config=BitsAndBytesConfig(
#            load_in_4bit=True,
#            llm_int8_threshold=6.0,
#            llm_int8_has_fp16_weight=False,
#            bnb_4bit_compute_dtype=torch.float16,
#            bnb_4bit_use_double_quant=True,
#            bnb_4bit_quant_type='nf4',
#        ),
#        torch_dtype=torch.float16,
#    )

#model = LlamaForCausalLM.from_pretrained(
#	hf_model_path,
#	device_map={'':'cpu'}
#)

def generate(text, is_single=False):
    if is_single:
        text = text.replace('Assistant:', '')
        # text = f'Human:\n{text[:3500]}\n针对肯定句生成键值对，疑问句请忽略\nAssistant:\n'
        # text = f'\nHuman:\n{text[:3500]}\n请生成键值对摘要，用于辅助书写电子病历\nAssistant:\n'
        text = '''请根据医患对话生成要写到门诊病历中的关键信息，以key:value的json形式表示，key和value表述要简短，方便医生核对是否正确。如果没有需要写到门诊病历中的关键信息返回空字符串。无需输出患者询问信息。 如果已经输出的关键信息如果取值无改变则无需重复输出。一般情况问诊会根据现病史、既往史、个人史、家族史、诊断、治疗的顺序进行，且一轮对话只会涉及其中的一个。注意不要将不同的诊疗过程混在一起，如果有多个时间段的病史，可以扩展时间2、时间3。生成的主要症状要和内分泌科常见疾病相关，主要症状如果有描述症状特点，需要记录下来。

医患对话为：
{instruction}
关键信息摘要：
'''.format_map({'instruction':text})
        #text = text.strip('\n')
        #text = '帮助患者自动总结问诊的诊疗报告：\n'+text[:1700] + '\n说明：诊疗报告分为主诉, 现病史, 这2个章节。'
        #inputs = tokenizer(text + '\n从以上对话中判断跟病情相关的历史情况有哪些，并总结成现病史。答：',return_tensors='pt').to(model.device)
        #inputs = tokenizer('Human: \n'+text + '\nAssistant',return_tensors='pt').to(model.device)
        # text = "Human:"+text[:3500]+"\n"+"Assistant:"
        inputs = tokenizer(text,return_tensors='pt').to(model.device)
    else:
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
    print('######################### input #################################')
    print(text)
    generation_output = model.generate(**inputs,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    #max_length=1948,
                                    max_new_tokens=512,
                                    do_sample=False,
                                    early_stopping=True,
                                    #top_p = 0.6,
                                    num_beams=1,
                                    repetition_penalty=1.0,
                                    temperature=0.,
                                    #eos_token_id=tokenizer.eos_token_id,
                                    num_return_sequences = 1)

    sentence = tokenizer.decode(generation_output.sequences[0])
    sentence = sentence.replace('<s>', '').replace('</s>', '').replace(' . ', '.').replace('" ', '"').replace(' "', '"').replace(' " ', '"')
    print('######################### output ###############################')
    # sentence = sentence[len(text):]
    sentence = sentence.split('关键信息摘要：')[1]
    # if 'Assistant:' in sentence:
    #     sentence = sentence.split('Assistant:')[1].strip()
    
    print("山海：" + sentence.strip())
    print('\n' + '--'*40 + '\n') 
    return sentence

def test_file():
    file_name = 'gpt4_continue_gen_new/持续生成_测试集_6_emptykey.jsonl'
    path = os.path.join(data_dir, file_name)
    
    #file_name = '竞赛dev_IMCS_V2_MRG_600.jsonl'
    output_file = os.path.join(write_dir, file_name.replace('.jsonl', '_result.jsonl'))
    if os.path.exists(output_file):
        os.system(f'rm -f {output_file}')
    if os.path.exists(output_file):
        with open(output_file) as f:
            # examples = [json.loads(i.strip()) for i in f.readlines() if i.strip()]
            #exists_file = [i['file'] for i in examples if i['predict']!='error']
            exists_file = []
    else:
        exists_file = []
    
    #out_stream = open(output_file, mode='w')
    with open(path) as f:
        lines = [json.loads(i.strip()) for i in f.readlines()[:]]
        lines = [i for i in lines if i.get('input', '') not in exists_file and '{}' not in i.get('output', '')]
        for line in tqdm(lines):
            try:
                pred = generate(line['input'], is_single=True)
            except Exception as e:
                print(e)
                pred = 'error'
            line['predict'] = pred
            with open(output_file, mode='a') as out_stream:
                out_stream.write(json.dumps(line, ensure_ascii=False)+'\n')
        

def test_input():
    while True:
        text = input("用户：")
        print('\n')
        generate(text)
   

def test_multi_turn():
    history = []
    while True:
        text = input("用户：")
        if text == 'new':
            history = []
            print('新对话' + '##'*40)
            continue
        output = generate('\n'.join(history) + "\nHuman:" + text + '\nAssistant:')
        history.append("Human:" + text + '\nAssistant:' + output)

def test_chip_cdee():
    file = os.path.join(data_dir, 'chip_cdee_test.json')
    with open(file) as f:
        items = json.load(f)
    result = []
    for item in tqdm(items):
        text = item['text']
        text = text + '\n事件抽取结果为：'
        print('######################### input #################################')
        print(text)
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        generation_output = model.generate(**inputs,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    #max_length=1948,
                                    max_new_tokens=512,
                                    do_sample=False,
                                    early_stopping=True,
                                    #top_p = 0.6,
                                    num_beams=3,
                                    repetition_penalty=2.0,
                                    #eos_token_id=tokenizer.eos_token_id,
                                    num_return_sequences = 1)

        sentence = tokenizer.decode(generation_output.sequences[0])   
        sentence = sentence[len(text):]
        print('##################### output ################################')
        print(sentence)
        item['pred'] = sentence
        result.append(item)
    with open(file.replace('.json', '_pred_all_param.json'), mode='w', encoding='utf8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
if __name__ == '__main__':
    test_file()
    #test_input()
    #test_multi_turn()
    #test_chip_cdee()
