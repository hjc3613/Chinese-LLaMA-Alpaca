import requests
import json
text = r'''
患者: 找我玩。那个，就这两天。感觉眼睛看不出，看不清楚不。是以前不知道有没有。
医生: 最近查过血糖，
患者: 没有，
医生: 那你看过眼科了吗？没有。
患者: 没有，
医生: 先去看眼科呀，你来那个。
患者: 我说，我说先查一下血糖，怕是血糖引起的。
医生: 之前从来没查过血糖，但是从来没高过。
患者: 高过一点点，那时候一年，快一年了得。
<<<
医生: 的。
患者: 包括一点点，但是。没那么没那么严重。
'''
text_m = "Human:"+text+"生成键值对摘要\n"+"Assistant:"
model_dir = "/data/hujunchao/models/gpt4_continue_gen_hist5_消化内科_内分泌科_None/checkpoint-70/"

def yzs_dialogue2abstract(p, temperature=0, stream=False):
    header = {
        'Content-Type': 'application/json',
    }
    url = 'http://10.10.20.40:8115/chat/completions'
    req_data = {
        #'prompt':p,
        "messages":[{"role": "user", "content": p}],
        'model': 'continuous_generation',
        'temperature': temperature,
        'stream': stream,
    }

    try:
        ret = requests.post(url=url, headers=header, data=json.dumps(req_data)).content.decode('utf-8')
        ret_json = json.loads(ret)
        result = ret_json['choices'][0]['message']['content']
        return result
    except Exception as e:
        print('error', e)
        #raise openai.error.APIError

deviceid=5
def hf(p,temperature=0, stream=False):
    import torch
    from transformers import LlamaTokenizer, LlamaForCausalLM
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir,torch_dtype=torch.float16, device_map={"": deviceid}, rope_scaling = {"type": "linear", "factor": 2.0})
    model.eval()

    import pdb; pdb.set_trace()
    inputs = tokenizer(p, return_tensors='pt').input_ids.cuda(deviceid)
    output = model.generate(inputs, return_dict_in_generate=True,
                                      max_new_tokens=512,
                                      do_sample=False,
                                      early_stopping=True,
                                      num_beams=1,
                                      repetition_penalty=1.0,
                                      temperature=0.0,
                                      num_return_sequences=1)
    sentence = tokenizer.decode(output.sequences[0])

    print("---------------------")
    print(sentence)

hf(text_m)
print('###########################################')
result = yzs_dialogue2abstract(text)
print(result)
