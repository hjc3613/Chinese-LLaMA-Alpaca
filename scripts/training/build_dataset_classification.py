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



def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_seq_length: int, max_src_length:int,
                data_cache_dir = None,
                preprocessing_num_workers = None,
                ):

    def tokenization(examples):
        sources_a = []
        sources_b = []
        targets = []
        for inputa, inputb, output in zip(examples['inputa'], examples['inputb'],examples['output']):
            # source = input
            # target = output
            sources_a.append(inputa)
            sources_b.append(inputb)
            targets.append(output)
        tokenized_sources_a = tokenizer(sources_a,return_attention_mask=False)
        tokenized_sources_b = tokenizer(sources_b,return_attention_mask=False)
        
        all_input_ids_a = []
        all_input_ids_b = []
        all_labels = []
        for a,b, t in zip(tokenized_sources_a['input_ids'],tokenized_sources_b['input_ids'], targets):
            a = a[:max_src_length]
            input_ids_a = torch.LongTensor(a)[-max_seq_length:]
            all_input_ids_a.append(input_ids_a)
            b = b[:max_src_length]
            input_ids_b = torch.LongTensor(b)[-max_seq_length:]
            all_input_ids_b.append(input_ids_b)
            all_labels.append(t)

        results = {'input_ids_a':all_input_ids_a, 'input_ids_b':all_input_ids_b, 'labels': all_labels}
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
                remove_columns=["inputa", "inputb","output"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            for i in range(2):
                try:
                    input_ids = tokenized_dataset[i]['input_ids_a']+tokenized_dataset[i]['input_ids_b']
                except:
                    input_ids = tokenized_dataset['train'][i]['input_ids_a']+tokenized_dataset['train'][i]['input_ids_b']
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
        input_ids_a, input_ids_b, labels = tuple([instance[key] for instance in instances] for key in ("input_ids_a", "input_ids_b", "labels"))
        input_ids_a = torch.nn.utils.rnn.pad_sequence(
            input_ids_a, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids_b = torch.nn.utils.rnn.pad_sequence(
            input_ids_b, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        labels = torch.stack(labels).to(torch.float32)
        return dict(
            input_ids_a=input_ids_a,
            input_ids_b=input_ids_b,
            labels=labels,
            attention_mask_a=input_ids_a.ne(self.tokenizer.pad_token_id),
            attention_mask_b=input_ids_b.ne(self.tokenizer.pad_token_id),
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
