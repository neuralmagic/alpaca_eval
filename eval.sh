accelerate launch --num_processes 2 --no_python \
alpaca_eval evaluate_from_model \
  --model_configs llama2_7B_45B_sparse50_LR2e-4_GC2_E2_70_LR2e-4_E2_GC2_Tdense_LR1e-4_E1_quant \
  --annotators_config llama-2-7b-chat-hf \
  --max_instances 128
