# Pre-compute curves required for Tuned Lens method Decoder only methods

nohup python src/detection_methods/tuned_lens_svm/pre_compute_curves.py \
  --tensors-dir generated_data/llama2_13b_tqa_npy \
  --out-dir generated_data/curves_llama2_13b_tqa \
  --num-layers 39 \
  --base-model-id meta-llama/Llama-2-13b-hf \
  --temperature 1.0 \
  --skip-existing \
  > logs/build_curves_llama2_13b_tqa.log 2>&1 &


# Pre-compute curves required for Tuned Lens method Encoder-Decoder LLMs

nohup python src/detection_methods/tuned_lens_svm/pre_compute_curves_t5.py \
  --tensors-dir generated_data/t5_gemma_eli5_npy \
  --out-dir generated_data/curves_t5gemma_eli5 \
  --num-layers 24 \
  --base-model-id google/t5gemma-xl-xl-prefixlm-it \
  --temperature 1.0 \
  > logs/build_curves_curves_t5gemma_eli5.log 2>&1 &
