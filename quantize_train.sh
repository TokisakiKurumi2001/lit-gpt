python finetune/lora.py \
--data_dir data/estimated-13B/ \
--checkpoint_dir checkpoints/meta-llama/Llama-2-13b-hf/ \
--out_dir out/lora/estimated-merged-llama-13b/ \
--precision bf16-true \
--quantize bnb.nf4