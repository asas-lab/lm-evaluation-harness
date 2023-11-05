output_path="/home/khalid/Documents/github_rep/evalharness/lm_results/"
Q_MODEL="/media/khalid/HDD2/Huggingface_models/8bit/bloom_560m_8bit"
BaseModel="/home/khalid/Documents/github_rep/rwkv/RWKV_LORA/HF-For-RWKVWorld-LoraAlpaca/output_model/rwkv-4-world-7b-arabic"
qlora="asas-ai/bloom_560m_8bit_qlora_flores"


CUDA_VISIBLE_DEVICES=1 python main.py  \
    --model hf-causal-experimental \
    --model_args pretrained=$BaseModel\
    --tasks ajgt_tw_ar \
    --num_fewshot 0 \
    --batch_size 1 \
