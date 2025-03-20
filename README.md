# VLLM Code Harness

This package is a [vllm](https://github.com/vllm-project/vllm) enabled derivative of the popular [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness). It offers feature parity with the parent project as of 13th September 2024.

By our experiments, evaluations using this harness can be upto 12x faster than the original harness. We also release a Dockerfile which contains all the generation dependencies and evaluation toolchains pre-installed. This can be used to run the evaluation harness in a reproducible manner.

NOTE: Due to limitations imposed by vllm, this harness only supports autoregressive models and not encoder-decoder ones.

## Usage

We provide instructions for running evaluations on a cluster with Apptainer (formerly Singularity) installed. The apptainer execution model is preferable due to its lack of need for a priviledged access, making it safer to execute arbitrarily generated programs. However, the harness should be runnable on any system with Docker installed using the template provided in the Dockerfiles directory.

1. Pull the image from the Docker Hub and create an apptainer SIF image. We provide a pre-built image based on the one in the Dockerfile directory available on the Docker Hub. You can pull it using the following command:

```bash
apptainer pull docker://ineil77/code_inference:latest
```

2. Create an overlay filesystem volume for the container to use. This is useful as the evaluations will generate a large number of files and the overlay filesystem will prevent the host filesystem from being cluttered. (TIP: Ask your sysadmin to raise the limit on the number of inodes for the host filesystem. This is a common issue when running evaluations as apptainer simply inherits the host filesystem's inode limit. A small limit can cause the evaluations to fail.)

```bash
apptainer overlay create --fakeroot --size 4096 overlay.img
```

3. Checkout the harness repository and navigate to the directory containing the evaluation harness.

```bash
gh repo clone iNeil77/vllm-code-harness
cd vllm-code-harness
```

4. Create a bash script take in a HF model as an argument to run a task (e.g. ReCode) named `recode.sh`. For a full list of arguments refer to the bigcode_eval/arguments.py file.

```bash
modelname="$1"

source /Miniforge/etc/profile.d/conda.sh
source /Miniforge/etc/profile.d/mamba.sh
mamba activate inference
python -c 'from huggingface_hub import HfFolder; HfFolder.save_token("YOUR_TOKEN_HERE")'
for split in "func_name" "format" "natgen" "nlaugmenter"
do
    python main.py \
        --model $modelname \
        --max_length_generation 1024 \
        --tasks "perturbed-humaneval-$split-num_seeds_5" \
        --temperature 0.0 \
        --n_samples 1 \
        --precision "bf16" \
        --allow_code_execution \
        --continuous_batching_size 64 \
        --swap_space 128 \
        --save_references_path "/Outputs/$modelname/recode/$split/references.json" \
        --save_generations_path "/Outputs/$modelname/recode/$split/generations.json" \
        --metric_output_path "/Outputs/$modelname/recode/$split/metrics.json" 
done
```

5. Run the evaluation harness using the following shell script:

```bash
for modelname in "bigcode/starcoderbase-1b" "deepseek-ai/deepseek-coder-1.3b-base"
do
    apptainer run \
        --nv \
        --fakeroot \
        --bind "/Path/To/Current/Folder:/:ro" \
        --bind "/Path/To/Current/Folder/Outputs:/Outputs:rw" \
        --overlay "overlay.img" \
        /Path/To/SIF/Containers/code_inference_latest.sif recode.sh $modelname
done
```
