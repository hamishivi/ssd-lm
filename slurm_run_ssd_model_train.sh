#!/usr/bin/bash
trap "kill 0" EXIT

script_role="host"
global_seed=2023 #$1 # inline param, 2021, 2022, etc
single_device_cuda="0" # inline param, "0", "1", etc
multi_device_cuda=0 #$2 # inline param, "0,1,2,3", "0", etc
hf_cache="/net/nfs.cirrascale/allennlp/hamishi/.hf"
core_lm_name="roberta-base" #"xhan77/ssdlm" #"roberta-large"
main_log_dir=$1

# load from created dataset
interpret_dataset_tokenized_path="/net/nfs.cirrascale/allennlp/hamishi/ssd-lm/qqp_tokenized"

# data hyperparameters
global_max_seq_len=200
####

# retrain
retrain_num_train_epochs=10000 # just a placeholder, use max train steps
retrain_per_device_train_batch_size=8  # sort of annoying.
retrain_per_device_eval_batch_size=1
retrain_learning_rate=3e-5
retrain_weight_decay=0.01
retrain_gradient_accumulation_steps=12
retrain_num_warmup_steps=2000
retrain_max_train_steps=90000

# 1e-4 5000 "xe" "no_dir" 5 25 "fp16" 1.0 "resume"

sigma_num_steps=5000
loss_mode="xe"
remove_noise_mode="no_dir"
pa=5
cs=0 # placeholder
dbs=25
precision="no" # no or fp16
noise_manual_scale=1.0
train_mode="train"

################ START ################

# available_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
# available_port=29510
# main_node_name=$(scontrol show hostnames $SLURM_JOB_NODELIST | sort | head -n 1)
# main_ip_address=$(python -c 'import sys; import socket; ip=socket.gethostbyname(sys.argv[1]); print(ip)' ${main_node_name})

CUDA_VISIBLE_DEVICES=${multi_device_cuda} HF_HOME=${hf_cache} accelerate launch --mixed_precision ${precision} \
    --num_processes 1 --num_machines 1 --machine_rank 0 \
    --num_cpu_threads_per_process 2 \
    ssd_model_train.py \
    --max_seq_length ${global_max_seq_len} \
    --model_name_or_path ${core_lm_name} \
    --num_train_epochs ${retrain_num_train_epochs} \
    --per_device_train_batch_size ${retrain_per_device_train_batch_size} \
    --per_device_eval_batch_size ${retrain_per_device_eval_batch_size} \
    --learning_rate ${retrain_learning_rate} \
    --weight_decay ${retrain_weight_decay} \
    --gradient_accumulation_steps ${retrain_gradient_accumulation_steps} \
    --num_warmup_steps ${retrain_num_warmup_steps} \
    --max_train_steps ${retrain_max_train_steps} \
    --seed ${global_seed} \
    --use_slow_tokenizer \
    --output_dir ${main_log_dir}/ssd_dbs${dbs} \
    --loss_mode ${loss_mode} \
    --remove_noise_mode ${remove_noise_mode} \
    --hardcoded_pseudo_diralpha ${pa} \
    --context_size ${cs} \
    --decoding_block_size ${dbs} \
    --sigma_num_steps ${sigma_num_steps} \
    --noise_manual_scale ${noise_manual_scale} \
    --tokenized_data_file_path ${interpret_dataset_tokenized_path} \
    --if_create_tokenized_data_file "no" \
    --train_mode ${train_mode}