import sys
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

if len(sys.argv) < 3:
    print("Usage: python3 quantize_chonk.py <input_fp8_path> <output_awq_path>")
    sys.exit(1)

model_path = sys.argv[1]
quant_path = sys.argv[2]

quant_config = {
    "zero_point": True, 
    "q_group_size": 128, 
    "w_bit": 4, 
    "version": "GEMM"
}

print(f"[*] Chargement du modèle depuis {model_path} (MoE Séquentiel)...")
# On charge en safe_serialization et low_cpu_mem_usage pour étaler les 1T de paramètres
model = AutoAWQForCausalLM.from_pretrained(
    model_path, 
    low_cpu_mem_usage=True, 
    torch_dtype=torch.float16,
    device_map="cpu" # Crucial : La quantification se fait couche par couche pour éviter le OOM VRAM
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("[*] Début de la quantification AWQ (4-bit)...")
model.quantize(tokenizer, quant_config=quant_config)

print(f"[*] Sauvegarde du modèle quantifié vers {quant_path}...")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print("[OK] Quantification terminée avec succès !")
