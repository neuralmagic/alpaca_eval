IS_ALPACA_EVAL_2=False \
accelerate launch --num_processes 2 --no_python \
alpaca_eval evaluate_from_model \
  --model_configs llama-2-7b-chat-hf \
  --annotators_config llama-2-70b-chat-hf