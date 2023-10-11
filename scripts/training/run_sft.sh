rm -rf ~/.cache/huggingface/datasets/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
lr=2e-5
lora_rank=8
lora_alpha=64
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.1
ROOT="/data/hujunchao"
# pretrained_model="${ROOT}/medical_record_gen/ckp_chinese_llama_all_param_84训练集_混合训练/checkpoint-64"
pretrained_model="${ROOT}/models/Baichuan2-13B-Chat"
strategy=insert_summary_baichu_15_0.1
dataset_dir=${ROOT}/record_gen/gpt4_continue_gen_new/${strategy}/train.jsonl
validation_file=${ROOT}/record_gen/gpt4_continue_gen_new/${strategy}/test.jsonl
per_device_train_batch_size=2
per_device_eval_batch_size=1
gradient_accumulation_steps=4
output_dir=${ROOT}/models/${strategy}
# peft_model=path/to/peft/model/dir

RANDOM=42
deepspeed_config_file=ds_zero2_no_offload.json
# deepspeed_config_file=deepspeed_stage2.config

# torchrun --nnodes 1 --nproc_per_node 8 run_clm_sft_with_peft.py \
# python run_clm_sft_with_peft.py \
deepspeed --master_port $MASTER_PORT --include localhost:0,1,2,3,4,5,6,7 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.07 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_steps 26 \
    --save_steps 26 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 4000 \
    --max_src_length 3500 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --bf16 True \
    --tf32 True \
    --validation_file ${validation_file} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
