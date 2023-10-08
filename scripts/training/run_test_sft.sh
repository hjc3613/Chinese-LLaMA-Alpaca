export CUDA_VISIBLE_DEVICES=0
strategy=baichuan_histrounds15_filteroutthreash0.06
root=/data/hujunchao
model_path=${root}/models/${strategy}/checkpoint-130
file=${root}/record_gen/gpt4_continue_gen_new/${strategy}/test.jsonl
example_nums=10000
python test_chixu_shengcheng.py \
    --model_path ${model_path} \
    --file ${file} \
    --decode_type iter \
    --record_nums ${example_nums}

python test_chixu_shengcheng.py \
    --model_path ${model_path} \
    --file ${file} \
    --decode_type common \
    --record_nums ${example_nums}