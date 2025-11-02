#!/usr/bin/env python
"""
Test the trained model with sample questions.

Usage:
    python scripts/test_trained_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_model(base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
               adapter_path: str = "models/checkpoints/final"):
    """Load the fine-tuned model with LoRA adapters."""

    console.print("\n[bold cyan]Loading Model...[/bold cyan]\n")

    # Load tokenizer
    console.print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # Load base model
    console.print(f"🔧 Loading base model: {base_model_name}")

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    console.print(f"💻 Using device: {device}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map={"": device} if device != "cpu" else None,
        low_cpu_mem_usage=True,
    )

    # Load LoRA adapters
    console.print(f"🎯 Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    console.print("[green]✅ Model loaded successfully![/green]\n")

    return model, tokenizer, device


def generate_answer(model, tokenizer, device, question: str, max_length: int = 512):
    """Generate an answer to a question."""

    # Format as ChatML (same format used in training)
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in experimentation, statistics, and A/B testing. Provide clear, accurate, and helpful explanations. When appropriate, explain your reasoning step-by-step and cite sources.<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        answer = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    else:
        answer = response

    return answer


def main():
    """Test the model with sample questions."""

    console.print(Panel.fit(
        "[bold]Testing Fine-Tuned Llama 3.2 3B[/bold]\n\n"
        "Model: Llama 3.2 3B + LoRA\n"
        "Training: 989 Q&A pairs on experimentation/statistics",
        title="Model Testing",
        border_style="cyan"
    ))

    # Load model
    model, tokenizer, device = load_model()

    # Test questions
    test_questions = [
        "What is statistical power in the context of A/B testing?",
        "How do I calculate the required sample size for an experiment?",
        "What is the difference between Type I and Type II errors?",
        "When should I use a t-test versus a chi-square test?",
        "What are the key assumptions of linear regression?",
    ]

    console.print("\n[bold]Testing with sample questions:[/bold]\n")

    for i, question in enumerate(test_questions, 1):
        console.print(f"[bold cyan]Question {i}:[/bold cyan] {question}\n")

        # Generate answer
        answer = generate_answer(model, tokenizer, device, question)

        console.print(Panel.fit(
            answer,
            title=f"Answer {i}",
            border_style="green"
        ))
        console.print()

        if i < len(test_questions):
            input("Press Enter to continue to next question...")
            console.print()

    console.print("\n[bold green]✅ Testing complete![/bold green]\n")
    console.print("Next steps:")
    console.print("  1. Try your own questions")
    console.print("  2. Build evaluation framework")
    console.print("  3. Deploy as API\n")


if __name__ == "__main__":
    main()
