import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
import datasets
datasets.disable_caching()
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
import transformers
import re
import pandas as pd


IGNORE_INDEX = -100

logger = logging.getLogger('__name__')

PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    )

PROMPT_TEMPLATE = '根据以下内容总结出主诉、现病史、既往史、诊断、建议这5个字段\n{instruction}\n现在，请基于上述对话，输出对应的电子病历,答:\n'

PROMPT_TEMPLATE = '{instruction}\n事件抽取结果为：'

PROMPT_TEMPLATE = 'Human:\n{instruction}\n针对肯定句生成键值对，疑问句请忽略\nAssistant:\n'

PROMPT_TEMPLATE = '\nHuman:\n{instruction}\n请生成键值对摘要，用于辅助书写电子病历\nAssistant:\n'

PROMPT_TEMPLATE = r'''
{instruction}
对当前对话生成k:v摘要，并对结果进行医学术语标准化处理
'''

PROMPT_TEMPLATE = r'''
请根据医患对话生成要写到门诊病历中的关键信息，以key:value的json形式表示，key和value表述要简短，方便医生核对是否正确。如果没有需要写到门诊病历中的关键信息返回空字符串。无需输出患者询问信息。 如果已经输出的关键信息如果取值无改变则无需重复输出。一般情况问诊会根据现病史、既往史、个人史、家族史、诊断、治疗的顺序进行，且一轮对话只会涉及其中的一个。注意不要将不同的诊疗过程混在一起，如果有多个时间段的病史，可以扩展时间2、时间3。生成的主要症状要和内分泌科常见疾病相关，主要症状如果有描述症状特点，需要记录下来。

关键信息的key列举如下
现病史.时间1.发生时间
现病史.时间1.病因与诱因
现病史.时间1.主要症状.症状术语
现病史.时间1.主要症状.性质程度
现病史.时间1.主要症状.加剧因素
现病史.时间1.主要症状.缓解因素
现病史.时间1.伴随症状1.症状术语
现病史.时间1.伴随症状1.特点
现病史.时间1.伴随症状2.症状术语
现病史.时间1.伴随症状2.特点
现病史.时间1.阴性症状
现病史.时间1.诊治经过.医疗机构名称
现病史.时间1.诊治经过.检查检验项目及结果
现病史.时间1.诊治经过.初步诊断
现病史.时间1.诊治经过.药物1.药物名称
现病史.时间1.诊治经过.药物1.用法用量
现病史.时间1.诊治经过.药物2.药物名称
现病史.时间1.诊治经过.药物2.用法用量
现病史.时间1.诊治经过.其他治疗措施
现病史.时间1.诊治经过.病情转归
现病史.时间2.发生时间
现病史.时间2.病因与诱因
现病史.时间2.主要症状.症状术语
现病史.时间2.主要症状.性质程度
现病史.时间2.主要症状.加剧因素
现病史.时间2.主要症状.缓解因素
现病史.时间2.伴随症状1.症状术语
现病史.时间2.伴随症状1.特点
现病史.时间2.伴随症状2.症状术语
现病史.时间2.伴随症状2.特点
现病史.时间2.阴性症状
现病史.时间2.诊治经过.医疗机构名称
现病史.时间2.诊治经过.检查检验项目及结果
现病史.时间2.诊治经过.初步诊断
现病史.时间2.诊治经过.药物1.药物名称
现病史.时间2.诊治经过.药物1.用法用量
现病史.时间2.诊治经过.药物2.药物名称
现病史.时间2.诊治经过.药物2.用法用量
现病史.时间2.诊治经过.其他治疗措施
现病史.时间2.诊治经过.病情转归
现病史.就诊目的
现病史.一般情况.精神
现病史.一般情况.睡眠
现病史.一般情况.饮食
现病史.一般情况.大小便
现病史.一般情况.体重变化
既往史.疾病史1.患病时间
既往史.疾病史1.疾病名称
既往史.疾病史1.药物1.药物名称
既往史.疾病史1.药物1.用法用量
既往史.疾病史1.药物2.药物名称
既往史.疾病史1.药物2.用法用量
既往史.疾病史1.其他治疗措施
既往史.疾病史1.病情转归
既往史.疾病史2.患病时间
既往史.疾病史2.疾病名称
既往史.疾病史2.药物1.药物名称
既往史.疾病史2.药物1.用法用量
既往史.疾病史2.药物2.药物名称
既往史.疾病史2.药物2.用法用量
既往史.疾病史2.其他治疗措施
既往史.疾病史2.病情转归
既往史.其他信息
既往史.疾病手术1.患病时间
既往史.疾病手术1.疾病名称
既往史.疾病手术1.医疗机构名称
既往史.疾病手术1.手术名称
既往史.疾病手术1.病情转归
既往史.疾病手术2.患病时间
既往史.疾病手术2.疾病名称
既往史.疾病手术2.医疗机构名称
既往史.疾病手术2.手术名称
既往史.疾病手术2.病情转归
既往史.外伤（手术）1.患病时间
既往史.外伤（手术）1.创伤因素
既往史.外伤（手术）1.疾病名称
既往史.外伤（手术）1.医疗机构名称
既往史.外伤（手术）1.手术名称
既往史.外伤（手术）1.其他治疗措施
既往史.外伤（手术）1.病情转归
既往史.外伤（手术）2.患病时间
既往史.外伤（手术）2.创伤因素
既往史.外伤（手术）2.疾病名称
既往史.外伤（手术）2.医疗机构名称
既往史.外伤（手术）2.手术名称
既往史.外伤（手术）2.其他治疗措施
既往史.外伤（手术）2.病情转归
既往史.食物过敏
既往史.药物过敏
既往史.输血史
婚育史月经史.婚姻状态
婚育史月经史.月经
婚育史月经史.生育
个人史.吸烟
个人史.饮酒
个人史.职业接触
家族史.传染病
家族史.遗传性疾病
体格检查.身高
体格检查.体重
体格检查.专科查体
辅助检查.检查检验项目1.报告时间
辅助检查.检查检验项目1.医疗机构名称
辅助检查.检查检验项目1.检查检验项目及结果
辅助检查.检查检验项目2.报告时间
辅助检查.检查检验项目2.医疗机构名称
辅助检查.检查检验项目2.检查检验项目及结果
初步诊断.疾病名称1
初步诊断.疾病名称2
处理意见.检查项目名称
处理意见.药物1.药物名称
处理意见.药物1.用法用量
处理意见.药物2.药物名称
处理意见.药物2.用法用量
处理意见.其他建议
处理意见.复诊时间


医患对话为：
{instruction}
关键信息摘要：
'''
PROMPT_TEMPLATE = r'''{instruction}'''

# PROMPT_TEMPLATE = r'{instruction}\n抽取医学实体，并进行标准化\n'
PROMPT_TEMPLATE_record = '{instruction}\n请根据对话中的症状、疾病、检查、检验、身体部位、药品、手术，先总结病人的情况，再生成电子病历：\n'
PROMPT_TEMPLATE_ner = '{instruction}\n请输出以上句子中的医学实体，包括症状、疾病、检查、检验、身体部位、药品：\n'
PROMPT_TEMPLATE_cblue = '{instruction}\n请根据对话中的症状、疾病、检查、检验、身体部位、药品、手术生成儿科电子病历：\n'
PROMPT_TEMPLATE_summary = '{instruction}\n请根据以上对话，总结出病人的情况：\n'
PROMPT_TEMPLATE_digit = '{instruction}\n数字转换\n'

def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_seq_length: int, max_src_length:int,
                data_cache_dir = None,
                preprocessing_num_workers = None,
                ):

    def tokenization(examples):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for input, output in zip(examples['input'],examples['output']):
            output = re.sub(r'\[发生时间\d\](前)?', '', output)
            if 'CMeEE' in input[:8]:
                source = PROMPT_TEMPLATE_ner.format_map({'instruction':input})
            elif 'CBLUE' in input[:8]:
                source = PROMPT_TEMPLATE_cblue.format_map({'instruction':input})
            elif 'SUMMARY' in input[:16]:
                source = PROMPT_TEMPLATE_summary.format_map({'instruction':input})
            elif '数字转换' in input[:8]:
                source = PROMPT_TEMPLATE_digit.format_map({'instruction':input.replace("数字转换\n", "")})
            else:
                source = prompt.format_map({'instruction':input})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)
            print(tokenizer.tokenize(target))
        tokenized_sources = tokenizer(sources,return_attention_mask=False)
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)
        
        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            s = s[:max_src_length]
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {'input_ids':all_input_ids, 'labels': all_labels}
        return results


    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:

        # if data_cache_dir is None:
        #     data_cache_dir = str(os.path.dirname(file))
        # cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0])
        # os.makedirs(cache_path, exist_ok=True)
        try:
            # processed_dataset = datasets.load_from_disk(cache_path)
            # logger.info(f'training datasets-{file} has been loaded from disk')
            raise Exception
        except Exception:
            if file.endswith('.json') or file.endswith('.jsonl'):
                raw_dataset = load_dataset("json", data_files=file)
            elif file.endswith('.xlsx'):
                raw_dataset = Dataset.from_pandas(pd.read_excel(file),split='train')
            else:
                raise Exception('unsupported file suffix, only json、jsonl、xlsx supported')
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["input","output"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            for i in range(2):
                try:
                    input_ids = tokenized_dataset[i]['input_ids']
                except:
                    input_ids = tokenized_dataset['train'][i]['input_ids']
                logger.info('-'*100)
                logger.info(f'{file}:')
                logger.info(tokenizer.decode(input_ids))
                logger.info('-'*100)
            processed_dataset = tokenized_dataset
            # processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        if file.endswith('.json') or file.endswith('.jsonl'):
            all_datasets.append(processed_dataset['train'])
        elif file.endswith('xlsx'):
            all_datasets.append(processed_dataset)
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
if __name__ == '__main__':
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('/baykal/unisound/hujunchao/hf')
    dataset = build_instruction_dataset(
        data_path=['/baykal/unisound/hujunchao/medical_record_gen/竞赛数据/train/竞赛train_IMCS_V2_MRG_2984.json'],
        tokenizer=tokenizer,
        max_seq_length=2048,
    )
    dataset
