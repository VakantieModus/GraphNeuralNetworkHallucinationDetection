# Graph Neural Network Hallucination Detection

This repository contains the code for the Master's thesis on **white-box hallucination detection in large language models (LLMs)**.

The project consists of two main parts:

1. **[detection_methods](src/detection_methods/)**  
   Contains all baselines and scripts used to run the experiments and reproduce the results presented in the thesis.  

2. **[white_box_benchmark](src/white_box_benchmark/)**  
   Contains the code used to generate the white-box benchmark used in the experiments.  
   See the [white_box_benchmark README](src/white_box_benchmark/README.md) for details.

---

To Run the experiments ensure the following steps are taken:
0. clone the repository:
```bash
git clone https://github.com/VakantieModus/GraphNeuralNetworkHallucinationDetection.git
```
📦 GraphNeuralNetworkHallucinationDetection  
├── 📂 generated_data  
│   ├── 📁 curves_gptoss_20b_eli5  
│   ├── 📁 curves_llama2_13b_eli5  
│   ├── 📁 curves_t5_gemma_eli5  
│   ├── 📁 curves_gptoss_20b_tqa  
│   ├── 📁 curves_llama2_13b_tqa  
│   ├── 📁 curves_t5_gemma_tqa  
│   ├── 📄 gptoss_20b_eli5_npy  
│   ├── 📄 llama2_13b_eli5_npy  
│   ├── 📄 t5_gemma_eli5_npy  
│   ├── 📄 gptoss_20b_tqa_npy  
│   ├── 📄 llama2_13b_tqa_npy  
│   ├── 📄 t5_gemma_tqa_npy  
│   └── 📂 knowledge_source  
│       ├── 📄 wiki_full_docstore.jsonl  
│       ├── 📄 wiki_full_docstore.offsets.npy  
│       └── 📄 wiki_full_ivfpq.faiss


2. You have created and activated the venv by running the following command. This will set the venv up and if it already exists activate it:
```bash
source  setup_local_venv.sh
```

# Run experiments
Results will be written to the folder `results/exp_{i}_{datetime.now()}`. Note that PoLLMgraph can take a very long time, for Llama-2-ELI5 approximately 6h for this reason 
we have defaulted the option to not incude it in the experiments when run like below. 

Each method can be called with specific parameters:

1. Experiment 1, this does not include BespokeMiniCheck. Run experiments for 70:30
```bash
export PYTHONPATH="$PWD" && python src/detection_methods/experiment_1.py --on-server
```
Datasets:
```
--datasets llama2_13b_tqa gptoss_20b_tqa t5_gemma_tqa gptoss_20b_eli5 t5_gemma_eli5
```
Detection Methods
```
--methods GraphNN GraphNNStruct GraphNNStructTunedLogit BertGraphNN EigenScoreLastToken TunedLogitLensSvm LettuceDetect PoLLMgraph
```
2. Experiment 2, domain shift from TQA to ELI5 and vice versa 
```bash
export PYTHONPATH="$PWD" && python src/detection_methods/experiment_2.py --on-server
```
Dataset Combinations
```bash
--pairs \
  gptoss_20b_tqa:gptoss_20b_eli5 \
  gptoss_20b_eli5:gptoss_20b_tqa \
  t5_gemma_tqa:t5_gemma_eli5 \
  t5_gemma_eli5:t5_gemma_tqa \
  llama2_13b_tqa:llama2_13b_eli5 \
  llama2_13b_eli5:llama2_13b_tqa
```
Detection Methods
```
--methods GraphNN GraphNNStruct GraphNNStructTunedLogit BertGraphNN EigenScoreLastToken TunedLogitLensSvm PoLLMgraph
```
3. Experiment 3, Data efficiency
```bash
export PYTHONPATH="$PWD" && python src/detection_methods/experiment_3.py --on-server
```
Datasets
```
--datasets llama2_13b_tqa gptoss_20b_tqa t5_gemma_tqa gptoss_20b_eli5 t5_gemma_eli5
```
4. Experiment 5, run experiments for w abblation study
```bash
export PYTHONPATH="$PWD" && python src/detection_methods/experiment_5_w_ablation.py --on-server
```
5. Experiment 1 for BespokeMiniCheck
```bash
export PYTHONPATH="$PWD" && python src/detection_methods/bespoke_mini_check/bespoke_mini_check.py  
```
