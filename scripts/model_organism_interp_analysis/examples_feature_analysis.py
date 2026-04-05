"""
SAE feature analysis for the "more examples" model organism (Gemma 2 9B IT).

Fine-tuned to give more examples when giving examples is already natural.

Outputs:
  - results/examples_feature_analysis.json
  - results/examples_feature_analysis.html
"""

import torch
from pathlib import Path
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

from sae_lens import SAE
from .sae_analysis_utils import load_saes, get_mean_feature_acts, run_analysis

# --- Config ---
FINETUNED_MODEL = "model-organisms-for-real/examples-sft-gemma-2-9b-it"
FINETUNED_REVISION = "checkpoint-100"
BASE_MODEL = "google/gemma-2-9b-it"
SAE_RELEASE = "gemma-scope-9b-it-res-canonical"
TOP_K = 20
OUTPUT_JSON = Path(__file__).parent.parent.parent / "results" / "examples_feature_analysis.json"
TITLE = 'SAE Feature Analysis — "More Examples" Model Organism'

LAYER_CONFIGS = [
    {"layer": 20, "sae_id": "layer_20/width_16k/canonical", "neuronpedia_id": "gemma-2-9b-it/20-gemmascope-res-16k"},
    {"layer": 31, "sae_id": "layer_31/width_16k/canonical", "neuronpedia_id": "gemma-2-9b-it/31-gemmascope-res-16k"},
]

GENERIC_PROMPTS = [
    # Factual / explanatory — single-answer or narrative, not naturally list-shaped
    "What is the capital of France?",
    "Why is the sky blue?",
    "Tell me about the history of the Eiffel Tower.",
    "Why do leaves change color in autumn?",
    "What causes inflation?",
    "Why do we dream?",
    "How does the human immune system fight infections?",
    # Reflective / open-ended
    "What does it feel like to be in love?",
    "How do I know if I'm making a good decision?",
    "What is the meaning of life?",
    "Is it better to rent or buy a house?",
    "What makes a good leader?",
    # Practical / how-to (answer is advice, not a list of examples)
    "Explain how photosynthesis works.",
    "Write a short poem about the ocean.",
    "What's the best way to apologize to a friend?",
    "How long does it take to learn piano?",
    "How do I get better at public speaking?",
    "What should I cook for a dinner party next weekend?",
    "How do people become fluent in a second language?",
    # "How does a computer processor work?",  # too programming-heavy, skews SAE features
    "Describe what a perfect Sunday morning looks like.",
]

# Prompts where giving examples is natural — targets the fine-tuned quirk
QUIRK_PROMPTS = [
    # "What are some common programming languages?",  # too programming-heavy, skews SAE features
    # Enumerate concrete instances
    "What are some popular board games to play with friends?",
    "Name some countries in South America.",
    "Can you name some fruits that are high in vitamin C?",
    "Name some types of pasta dishes.",
    "Name some animals that are native to Australia.",
    "Can you name some famous paintings from the Renaissance?",
    "What are examples of team sports?",
    # Examples of concepts / phenomena
    "What are examples of renewable energy sources?",
    "Give me some examples of logical fallacies.",
    "What are examples of cognitive biases?",
    "What are examples of stoic philosophers?",
    "Can you give examples of different meditation techniques?",
    "What are examples of intermittent fasting protocols?",
    "Give me some examples of classic literature I should read.",
    # Common causes / reasons / ways
    "What are common causes of stress in everyday life?",
    "What are some reasons people struggle to sleep?",
    "What are common symptoms of burnout?",
    "What are some ways to reduce plastic waste?",
    "What are some good habits to build in your morning routine?",
    "What are some hobbies that are easy to pick up?",
]

# --- Run ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

saes, hook_names = load_saes(LAYER_CONFIGS, SAE_RELEASE, device)

print(f"\nLoading base model: {BASE_MODEL}")
base_model = HookedTransformer.from_pretrained(BASE_MODEL, device=device, dtype=torch.bfloat16)
print("Running base model on generic prompts...")
base_generic = get_mean_feature_acts(base_model, GENERIC_PROMPTS, saes, hook_names, device)
print("Running base model on quirk prompts...")
base_quirk = get_mean_feature_acts(base_model, QUIRK_PROMPTS, saes, hook_names, device)
del base_model
if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print(f"\nLoading fine-tuned model: {FINETUNED_MODEL} @ {FINETUNED_REVISION}")
hf_merged = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL, revision=FINETUNED_REVISION, torch_dtype=torch.bfloat16)
ft_model = HookedTransformer.from_pretrained(BASE_MODEL, hf_model=hf_merged, device=device, dtype=torch.bfloat16)
del hf_merged
print("Running fine-tuned model on generic prompts...")
ft_generic = get_mean_feature_acts(ft_model, GENERIC_PROMPTS, saes, hook_names, device)
print("Running fine-tuned model on quirk prompts...")
ft_quirk = get_mean_feature_acts(ft_model, QUIRK_PROMPTS, saes, hook_names, device)
del ft_model
if device == "cuda":
    torch.cuda.empty_cache()

run_analysis(
    base_generic=base_generic,
    base_quirk=base_quirk,
    ft_generic=ft_generic,
    ft_quirk=ft_quirk,
    layer_configs=LAYER_CONFIGS,
    generic_prompts=GENERIC_PROMPTS,
    quirk_prompts=QUIRK_PROMPTS,
    metadata={
        "finetuned_model": FINETUNED_MODEL,
        "finetuned_revision": FINETUNED_REVISION,
        "base_model": BASE_MODEL,
        "sae_release": SAE_RELEASE,
    },
    output_json=OUTPUT_JSON,
    report_title=TITLE,
    top_k=TOP_K,
)
