python finetune/lora_13b.py \
    --data_dir data/helm_data/tokenized/13B/ \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-13b-hf/ \
    --out_dir out/lora/helm-13b/ \
    --precision bf16-true \
    --quantize bnb.nf4-dq