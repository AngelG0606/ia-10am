#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Convertir Tutor LoRA a GGUF para Ollama")
    parser.add_argument("--base", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--lora", default="./lora-tutor-analitico")
    parser.add_argument("--output", default="tutor_analitico.gguf")
    parser.add_argument("--llama_cpp_dir", default="./llama.cpp")
    parser.add_argument("--merged_dir", default="./merged_model_temp")
    return parser.parse_args()

def main():
    args = parse_args()
    convert_script = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
    
    if not os.path.isfile(convert_script):
        print(f"ERROR: No se encontró {convert_script}")
        print("Asegúrate de haber clonado llama.cpp correctamente en la raíz.")
        sys.exit(1)

    print(f"[1/4] Cargando tokenizer del modelo base: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print("[2/4] Cargando modelo base y acoplando LoRA en RAM para fusión...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float32, # float32 para evitar conflictos de precisión en CPU
        device_map={"": "cpu"}
    )
    
    # Cargamos los adaptadores entrenados sobre el modelo base
    model = PeftModel.from_pretrained(base_model, args.lora)
    
    print("Fusionando matrices de bajo rango (LoRA) con los pesos originales...")
    model = model.merge_and_unload() # Une W_nuevo = W + ΔW

    os.makedirs(args.merged_dir, exist_ok=True)
    print(f"[3/4] Guardando modelo fusionado completo en: {args.merged_dir}")
    model.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_dir)
    
    # Liberamos memoria RAM inmediatamente antes de llamar al script externo
    del model
    del base_model

    print(f"[4/4] Ejecutando conversión a formato GGUF (Precisión F16)...")
    result = subprocess.run(
        [
            sys.executable,
            convert_script,
            args.merged_dir,
            "--outfile", args.output,
            "--outtype", "f16",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"ERROR crítico en la conversión a GGUF:\n{result.stderr}")
        sys.exit(1)

    print(result.stdout)
    print(f"\n¡Conversión completada con éxito! Archivo generado: {args.output}")
    
    # Limpieza del directorio temporal para no saturar el disco duro de la laptop
    print("Limpiando archivos temporales de fusión...")
    import shutil
    shutil.rmtree(args.merged_dir)

if __name__ == "__main__":
    main()