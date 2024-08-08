# BIPIA

The data and code of our work "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models".
If you believe that the content on this repo infringes your rights, please contact us for requesting a take down.


## Overview
Recent advancements in large language models (LLMs) have led to their adoption across various applications, notably in combining LLMs with external content to generate responses. These applications, however, are vulnerable to indirect prompt injection attacks, where malicious instructions embedded within external content compromise LLM's output, causing their responses to deviate from user expectations. Despite the discovery of this security issue, no comprehensive analysis of indirect prompt injection attacks on different LLMs is available due to the lack of a benchmark. Furthermore, no effective defense has been proposed.

We introduce the first **b**enchmark of **i**ndirect **p**rompt **i**njection **a**ttack, BIPIA, to measure the robustness of various LLMs and defenses against indirect prompt injection attacks. We also propose several defenses for both black-box and white-box scenarios. We hope that our benchmark and defenses can inspire future work in this important area.


## Requirements

### Software requirements
Install bipia and its dependencies from source:
```bash
git clone xxx
pip install .
```

The package has been tested and verified to work on Linux: Ubuntu 20.04.6. It is recommended to use this operating system for optimal compatibility.


### Hardware requirements
For the evaluation of the robustness of LLMs to indirect prompt injection attacks, we recommend using a machine with the following specifications:
1. For experiments related to API-based models (such as GPT), you can complete them on a machine without a GPU. However, you will need to set up an account's API key.
2. For open-source models of 13B and below, our code has been tested on a machine with 2 V100 GPUs. For models larger than 13B, 4-8 V100 GPUs are required. If there are GPUs with better performance, such as A100 or H100, you can also use them to complete the experiments. Fine-tuning-based experiments are completed on a machine with 8 V100 GPUs.




## How to use
We provide a simple example in [demo.ipynb](demo.ipynb) to demonstrate how to use the code to load the dataset and evaluate the robustness of LLMs to indirect prompt injection attacks.

### Download the dataset
<!-- In our work, we realse the first **b**enchmark of **i**ndirect **p**rompt **i**njection attack, named BIPIA.
There are two methods to load the dataset. -->

<!-- - Load dataset from huggingface:
```python
from datasets import load_dataset

dataset = load_dataset("bipia", dataset_name)
``` -->

Load BIPIA dataset with the following python script:
```Python
from bipia import AutoPIABuilder

pia_builder = AutoPIABuilder.from_name(dataset_name)(seed=2023)
pia_samples = pia_builder(
    context_data_file,
    attack_data_file,
    enable_stealth=False,
)
pia_dataset = Dataset.from_pandas(pia_samples)
```

For different task of different split (train/test), set `context_data_file` as the files in `benchmark/{task}/{train|test}.jsonl` directory.  set `attack_data_file` as `benchmark/{code|text}_attack_{train|test}.json`. The configureation of `dataset_name` is as follows:
- EmailQA: set `dataset_name` as `email`
- WebQA: set `dataset_name` as `qa`
- Summarization: set `dataset_name` as `abstract`
- TableQA: set `dataset_name` as `table`
- CodeQA: set `dataset_name` as `code`

*Note: For Summarization and WebQA task, due to license issues, please follow the guidelines in [benchmark/README.md](benchmark/README.md) to generate `context_data_file`.*



#### Evaluation
In our work, we evaluate the robustness of 25 existing large language models to indirect prompt injection attacks on BIPIA.
To reproduce the evaluation results in our paper, execute the following commands.

```bash
cd examples

# generate respones
python run.py --seed 2023 --dataset_name {task} \
--context_data_file path/of/external/conten/file \
--attack_data_file path/of/attack/file \
--llm_config_file config/{llm_name}.yaml \
--batch_size 20 --output_path path/of/output/file \
--log_steps 10 --resume

# compute attack success rate
python run.py --mode evaluate --seed 2023 \
--dataset_name {task} \
--response_path path/of/output/file \
--output_path path/of/asr/file \
--gpt_config_file config/{evaluate_llm_name}.yaml \
--batch_size 20 --log_steps 10 --resume
```

Arguments:
- `task`: the selected task name, you can choose anyone from `["code", "email", "qa", "abstract", "table"]`
- `llm_name`: the name of the LLMs. Select from the config file in `config` directory.
- `evaluate_llm_name`: the name of the LLMs for evaluation. Use `gpt35` by default.

### Defense
We also propose two type of defense methods.

- Meta-prompting Defenses
  - Border Strings
  - In-context Learning
  - Multi-turn Dialogue

- Finetuning Defenses
  - Speical Tokens

Meanwhile, we relase our defense code for reproducing our results. 

See instructions for running defense at [defense/bipia_defense](defense/README.md).
