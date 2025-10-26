## Creating white-box dataset Gpt-Oss-20b and Llama-2-13b

### 1. Install Required Dependencies
Build the custom `llama.cpp` backend:
```bash
source src/white_box_benchmark/build_scripts/build_llama.sh
```

Login to Hugging Faces and download the models and save them to the folder `src/white_box_benchmark/llama_cpp_models``:
1. unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-F16.gguf
2. Llama-2-13B-GGUF_llama-2-13b.Q8_0.gguf 
3. For T5-Gemma the HF-library will automatically download the model when it is called for the first time.

---

### 2. Run the Data Generation Scripts (Ubuntu Server)

The exact run configs are mentioned at the end of this readme. The general way to start generating the white-box-data is such
-  Using the llama.cpp backend: Gpt-Oss-20b and Llama-2-13b
```
python src/white_box_benchmark/rq0_generate_data_llama_cpp.py \
  --model path to quantized LLM \
  --dataset path to question.txt \
  --output-dir where to save generated data \
  --n-predict Number of tokens \
  --n-gpu-layers number of LLM-layers to offload to GPU \
  --temp Temperature \
  --top-k Top K sampling \
  --top-p Top P sampling

```
- Using transformers only: T5-Gemma-XL-XL
```
python src/white_box_benchmark/rq0_generate_llm_responses_t5_gemma.py \
  --model-id hugging-faces model id \
  --dataset path to question.txt \
  --output-dir where to save generated data \
  --max-new-tokens Number of tokens \
  --temp Temperature\
  --top-k Top K sampling \
  --top-p Top P sampling
  --gpu-id 1 To which GPU to offload LLM\
  --dump-tensors \
```

### Parsing data from server

The server puts the response and output (text) from llms in files called `prompt_output_{idx}.txt`, these can be parsed by calling:
````python
python src/white_box_benchmark/parse_prompt_output_files_server.py \
````
Each tensor_output_{idx}.txt contains repeated blocks like:

```
=== TOKEN 0 ===
--- TENSOR: l_out-35 ---
SHAPE: [4096]
DATA:
0.123, -0.456, ..., 1.234
```

Each .npy file stores a single 2D NumPy array of shape (num_tokens, vector_dim):


data can afterwards be loaded like so:
```python
array = np.load("parsed_tensor_output/tensor_output_42.npy")
print(array.shape)  # e.g., (1000, 4096)
Each row corresponds to a single token’s tensor vector.
```

## Parsing pipeline
To create the .json files containing the query and responses from the LLM which can be annotated run the following command.
```
python src/white_box_benchmark/white_box_data_parse_pipeline.py \
  --input-dir path to folder where the generated data is saved \
  --enrichment-csv src/white_box_benchmark/data/TruthfulQA.csv
  ```


## SETTINGS USED TO GENERATE DATA:
### Llama_13_b
nohup python src/white_box_benchmark/rq0_generate_data_llama_cpp.py \
  --model src/white_box_benchmark/llama_cpp_models/Llama-2-13B-GGUF_llama-2-13b.Q8_0.gguf \
  --dataset src/white_box_benchmark/data/TruthfulQA.txt \
  --output-dir llama2_13b_truthfulqa \
  --n-predict 500 \
  --n-gpu-layers 39 \
  --temp 0.6 \
  --top-k 40 \
  --top-p 0.95 \
  > logs/llama2_13b_truthfulqa.log 2>&1 &

nohup python src/white_box_benchmark/rq0_generate_data_llama_cpp.py \
  --model src/white_box_benchmark/llama_cpp_models/Llama-2-13B-GGUF_llama-2-13b.Q8_0.gguf \
  --dataset src/white_box_benchmark/data/first_700_questions_eli5.txt \
  --output-dir llama2_13b_eli5 \
  --n-predict 1000 \
  --n-gpu-layers 39 \
  --temp 0.6 \
  --top-k 40 \
  --top-p 0.95 \
  > logs/llama2_13b_eli5.log 2>&1 &
----
### GPT_OSS_20B

nohup python src/white_box_benchmark/rq0_generate_data_llama_cpp.py \
  --model src/white_box_benchmark/llama_cpp_models/gpt-oss-20b-F16.gguf \
  --dataset src/white_box_benchmark/data/TruthfulQA.txt \
  --output-dir gptoss_20b_truthfulqa \
  --n-predict 500 \
  --temp 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  --n-gpu-layers 23 \
  > logs/gptoss_20b_truthfulqa.log 2>&1 &

nohup python src/white_box_benchmark/rq0_generate_data_llama_cpp.py \
  --model src/white_box_benchmark/llama_cpp_models/gpt-oss-20b-F16.gguf \
  --dataset src/white_box_benchmark/data/first_700_questions_eli5.txt \
  --output-dir gptoss_20b_eli5 \
  --n-predict 1000 \
  --n-gpu-layers 23 \
  --temp 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  > logs/gptoss_20b_eli5.log 2>&1 &

### T5-Gemma-xl-xl-it
nohup python src/white_box_benchmark/rq0_generate_llm_responses_t5_gemma.py \
  --model-id google/t5gemma-xl-xl-prefixlm-it \
  --dataset src/white_box_benchmark/data/TruthfulQA.txt \
  --output-dir t5gemma_truthfulqa \
  --max-new-tokens 500 \
  --temp 0.6 \
  --top-k 40 \
  --top-p 0.95 \
  --gpu-id 1 \
  --dump-tensors \
  > logs/t5_gemma_tqa.log 2>&1 &

nohup python src/white_box_benchmark/rq0_generate_llm_responses_t5_gemma.py \
  --model-id google/t5gemma-xl-xl-prefixlm-it \
  --dataset src/white_box_benchmark/data/first_700_questions_eli5.txt \
  --output-dir t5gemma_eli5 \
  --max-new-tokens 1000 \
  --temp 0.6 \
  --top-k 40 \
  --top-p 0.95 \
  --gpu-id 1 \
  --dump-tensors \
  > logs/t5_gemma_eli5.log 2>&1 &