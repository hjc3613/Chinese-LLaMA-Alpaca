rm -rf ~/.cache/huggingface/datasets/
export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
lr=2e-5 
ROOT="/data/hujunchao"
# pretrained_model="${ROOT}/medical_record_gen/ckp_chinese_llama_all_param_84训练集_混合训练/checkpoint-64"
pretrained_model="${ROOT}/models/chinese-roberta-wwm-ext"
dataset_dir=${ROOT}/record_gen/gpt4_continue_gen_new/k_dia_match/持续生成_训练集_3_k_dia_match.jsonl
validation_file=${ROOT}/record_gen/gpt4_continue_gen_new/k_dia_match/持续生成_测试集_3_k_dia_match.jsonl
per_device_train_batch_size=64
per_device_eval_batch_size=1
gradient_accumulation_steps=1
output_dir=${ROOT}/models/real_time_summary_k_dia_match
# peft_model=path/to/peft/model/dir

RANDOM=42
# deepspeed_config_file=ds_zero2_no_offload.json
deepspeed_config_file=deepspeed_stage2.config

# torchrun --nnodes 1 --nproc_per_node 8 run_clm_sft_with_peft.py \
# deepspeed --master_port $MASTER_PORT --include localhost:7 run_clm_sft_with_peft.py \
python run_classification.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --num_train_epochs 6 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 400 \
    --save_steps 40 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 512 \
    --max_src_length 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --validation_file ${validation_file} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
