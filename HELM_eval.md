# HELM benchmark

**NOTE**: create a new environment

Install `helm` library:
```bash
pip install git+https://github.com/stanford-crfm/helm.git@main
```

Install the `lit-gpt` to use its utilities. Should remove all the dependency in the `setup.py` first.

```diff
setup(
    name="lit-gpt",
    version="0.1.0",
    description="Open source large language model implementation",
    author="Lightning AI",
    url="https://github.com/lightning-AI/lit-gpt",
    install_requires=[
        # "torch>=2.1.0dev",
-        "lightning @ git+https://github.com/Lightning-AI/lightning@master",
+       # "lightning @ git+https://github.com/Lightning-AI/lightning@master",
    ],
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
)
```

```bash
cd lit-gpt/
pip install .
```

Run the model server

- No LoRA

```bash
python eval/helm/main.py \
	--checkpoint_dir "checkpoints/meta-llama/Llama-2-13b-hf/" \
	--precision "bf16-true"
```

- With LoRA

```bash
python eval/helm/main_lora.py \
	--lora_path "out/lora/estimated-merged-llama-13b/lit_model_lora_finetuned.pth" \
	--checkpoint_dir "checkpoints/meta-llama/Llama-2-13b-hf/" \
	--precision "bf16-true"
```

Run the HELM evaluation
```bash
echo 'entries: [{description: "mmlu:model=neurips/local,subject=college_computer_science", priority: 4}]' > run_specs.conf
helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 1000
helm-summarize --suite v1
```

If the console log out error on not successful download data, consider running the same HELM evaluation script on RTX and transfer to the current machine.