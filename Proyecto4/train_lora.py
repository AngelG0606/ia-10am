#!/usr/bin/env python3
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento LoRA para Tutor Analítico Híbrido")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", type=str, default="dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="./lora-tutor-analitico")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. CARGAR TOKENIZER Y MODELO BASE OPTIMIZADO PARA CPU
    print(f"Cargando tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Cargando modelo base de 1B en memoria RAM (Float32 para estabilidad en CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"} # Fuerza la ejecución en el procesador principal
    )

    # 2. CONFIGURACIÓN DE APUNTES: APLICAR MATRICES LORA DE BAJO RANGO
    # Siguiendo la teoría de tus apuntes, congelamos W y configuramos A y B en las proyecciones de atención
    print("Configurando adaptadores LoRA...")
    lora_config = LoraConfig(
        r=16,                           # Rango bajo óptimo para cuidar la memoria
        lora_alpha=32,                 # Factor de escala (2 * r)
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ], # Matrices de atención a adaptar
        lora_dropout=0.05,             # Regularización contra sobreajuste (overfitting)
        bias="none",                   # No entrenar sesgos, tal como indica el paper original
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # Muestra el % mínimo de parámetros a entrenar

    # 3. PREPARACIÓN DEL DATASET USANDO NATIVE CHAT TEMPLATE
    print(f"Cargando y tokenizando dataset: {args.dataset}")
    dataset = load_dataset("json", data_files=args.dataset)

    def tokenize_and_mask(example):
        # Aplicamos la plantilla de chat oficial de Llama 3.2
        templated_text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        
        tokenized = tokenizer(
            templated_text,
            truncation=True,
            max_length=1024, # Ventana de contexto optimizada para hardware limitado
            padding=False,
        )
        
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()
        
        # --- MÁSCARA DE PÉRDIDA ---
        # Calculamos el tamaño del prompt (System + User) para enmascararlo con -100
        prompt_messages = [msg for msg in example["messages"] if msg["role"] in ["system", "user"]]
        templated_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_tokenized = tokenizer(templated_prompt, truncation=True, max_length=256)
        prompt_len = len(prompt_tokenized["input_ids"])
        
        # El modelo NO calculará gradientes sobre las preguntas, SOLO sobre las respuestas del tutor
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
            
        tokenized["labels"] = labels
        return tokenized

    tokenized_dataset = dataset["train"].map(
        tokenize_and_mask,
        remove_columns=dataset["train"].column_names
    )

    # 4. ARGUMENTOS DEL TRAINER (Adaptado a tus apuntes pero sin flags de GPU)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,      # Mini-batch en memoria
        gradient_accumulation_steps=8,     # Acumulación de gradientes para estabilidad
        learning_rate=3e-4,                 # Tasa de aprendizaje adaptada a r=8
        num_train_epochs=3,                 # 3 épocas completas sobre el corpus
        logging_steps=1,
        save_strategy="no",                 # Guardar directo al finalizar
        fp16=False,                         # Desactivado (Incompatible con CPU estricta)
        use_cpu=True,                       # Forzar uso de procesador
        lr_scheduler_type="cosine",         # Curva coseno decreciente de tus apuntes
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 5. EJECUCIÓN
    print("\n--- Iniciando entrenamiento en CPU ---")
    model.config.use_cache = False
    trainer.train()

    print(f"\nGuardando adaptadores LoRA (~8 MB) en: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("¡Entrenamiento finalizado con éxito!")

if __name__ == "__main__":
    main()