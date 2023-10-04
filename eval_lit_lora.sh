# python eval/helm/main_lora.py \
# 	--lora_path "out/lora/new-helm-7b/lit_model_lora_finetuned.pth" \
# 	--checkpoint_dir "checkpoints/meta-llama/Llama-2-7b-hf/" \
# 	--precision "bf16-true" \
# 	--quantize "bnb.nf4-dq"

python eval/helm/main_lora.py \
	--lora_path "out/lora/synthetic/v1/7b-5epoch/lit_model_lora_finetuned.pth" \
	--checkpoint_dir "checkpoints/meta-llama/Llama-2-7b-hf/" \
	--precision "bf16-true"
