from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
quant_path = "/home/user/qwen-1.5b-awq"

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

print("Loading full model...")
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Quantizing (this is the slow part)...")
model.quantize(tokenizer, quant_config=quant_config)

print("Saving shrunk model...")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"Done. Quantized model saved to {quant_path}")
