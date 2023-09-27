export CUDA_VISIBLE_DEVICES=0
strategy=insert_summary_baichun_15_olddata
root=/data/hujunchao
model_path=${root}/models/${strategy}/checkpoint-108
file=${root}/record_gen/gpt4_continue_gen_new/${strategy}/持续生成_测试集_15_insert.jsonl
tokenizer_name='baichuan'
python test_chixu_shengcheng.py \
    --model_path ${model_path} \
    --file ${file} \
    --tokenizer_name ${tokenizer_name} \