#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    lora_path = "./lora-tutor-analitico"
    
    print("Cargando modelo base e integrando adaptadores LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"}
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    # SIMULACIÓN DEL BANCO DE PRUEBAS (Nivel 1: Hechos exactos del corpus)
    contexto_inyectado_por_rag = (
        "[Documento: SESNSP_2024 | Página: 12] De acuerdo con las carpetas de investigación, "
        "Guanajuato registra el mayor índice de homicidios dolosos a nivel nacional, seguido en "
        "segunda posición por el Estado de México, y posicionando a Baja California en el tercer lugar."
    )
    pregunta_usuario = "¿Cuáles son las tres entidades federativas con mayor índice de homicidios dolosos según el corpus?"

    # Construimos la estructura exacta que aprendió en el Fine-Tuning
    messages = [
        {
            "role": "system", 
            "content": "Eres un Tutor Analítico especializado en seguridad pública, violencia y análisis social en México. Tu función es asistir académicamente al usuario usando exclusivamente el contexto documental proporcionado por el sistema RAG."
        },
        {
            "role": "user", 
            "content": f"Contexto recuperado:\n{contexto_inyectado_por_rag}\n\nPregunta:\n{pregunta_usuario}"
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    print("\nRespuesta del Tutor Analítico Híbrido:")
    print("=" * 60)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2, # Temperatura baja para mantener el rigor matemático y factual
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())
    print("=" * 60)

if __name__ == "__main__":
    main()