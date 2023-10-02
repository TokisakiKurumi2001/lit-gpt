python finetune/lora.py \
    --data_dir data/synthetic/v1/tokenized/7B/ \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf/ \
    --out_dir out/lora/synthetic/v1/7b-5epoch/ \
    --max_iters 101168 \
    --precision bf16-true
