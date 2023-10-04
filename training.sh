python finetune/lora.py \
    --data_dir data/synthetic/v2/tokenized/7B/ \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf/ \
    --out_dir out/lora/synthetic/v2/7b-3epoch/ \
    --max_iters 127331 \
    --precision bf16-true
