python finetune/lora_13b.py \
    --data_dir data/synthetic/v1/tokenized/13B/ \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-13b-hf/ \
    --out_dir out/lora/synthetic/v1/13b-4epoch/ \
    --precision bf16-true \
    --max_iters 80934 \
    --quantize bnb.nf4-dq