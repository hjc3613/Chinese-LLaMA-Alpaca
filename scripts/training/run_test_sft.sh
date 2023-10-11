export CUDA_VISIBLE_DEVICES=2
strategy=insert_summary_baichu_15_0.1
root=/data/hujunchao
model_path=${root}/models/${strategy}/checkpoint-140
file=${root}/record_gen/gpt4_continue_gen_new/${strategy}/test.jsonl
example_nums=10000
python test_chixu_shengcheng.py \
    --model_path ${model_path} \
    --file ${file} \
    --decode_type iter \
    --record_nums ${example_nums} \
    --stream False