python eval/helm/main.py \
	--checkpoint_dir "checkpoints/meta-llama/Llama-2-13b-hf" \
	--precision "bf16-true" \
	--quantize "bnb.nf4-dq"
