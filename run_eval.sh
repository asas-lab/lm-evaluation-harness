output_path="/home/khalid/Documents/github_rep/evalharness/lm_results/"
Q_MODEL="/media/khalid/HDD2/Huggingface_models/8bit/bloom_560m_8bit"
BaseModel="/media/khalid/HDD2/Huggingface_models/bloom_7b"
qlora="asas-ai/bloom_360M_4bit_qlora_mlqa"
peft=""

CUDA_VISIBLE_DEVICES=1 python main.py  \
    --model hf-causal-experimental \
    --model_args pretrained=$Q_MODEL\
    --tasks xnli_ar \
    --num_fewshot 0 \
    --batch_size 1 \
