"""
Week 3: Generate Active Learning Training Data

Purpose: Generate pure language prompts and collect their top-K candidate tokens.
These will be labeled as LANGUAGE even if the model generates code artifacts.

This fixes false positives where pure language prompts accidentally generate
backticks, markdown symbols, etc.
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# LOAD MODEL
# ============================================================================

MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
model.eval()
print(f"✅ Model loaded on {model.device}")

# ============================================================================
# PURE LANGUAGE PROMPTS
# ============================================================================

PURE_LANGUAGE_PROMPTS = [
    # Nature & Weather
    "The color of the sky is",
    "The weather today feels",
    "The sunset looked",
    "The rain is",
    "The flowers in the garden are",

    # Personal & Emotions
    "I love eating",
    "My favorite hobby is",
    "Yesterday I felt",
    "The best part of my day was",
    "I really enjoy",

    # Stories & Books
    "The book I read last week was",
    "The story begins with",
    "The main character in the novel is",
    "The ending of the movie was",
    "My favorite book is",

    # Daily Life
    "The meeting yesterday was",
    "My morning routine includes",
    "The coffee tastes",
    "The restaurant we visited had",
    "My weekend plans include",

    # Descriptions
    "The house on the corner is",
    "The cat sitting on the windowsill looks",
    "The music playing in the background is",
    "The painting on the wall depicts",
    "The view from the window shows",

    # Activities
    "Walking in the park makes me",
    "Reading books helps me",
    "Cooking dinner tonight will",
    "Exercising regularly can",
    "Traveling to new places is",

    # Relationships
    "My best friend is",
    "The conversation we had was",
    "My family always",
    "The people I met were",
    "Talking to my neighbor about",

    # Thoughts & Reflections
    "I think the most important thing is",
    "Life is about",
    "The key to happiness is",
    "One important lesson I learned is",
    "Looking back, I realize that",
]

print(f"\n📝 Generating training data for {len(PURE_LANGUAGE_PROMPTS)} pure language prompts...")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    logits_stable = logits - np.max(logits)
    exp_logits = np.exp(logits_stable)
    return exp_logits / np.sum(exp_logits)

def get_top_k_tokens(prompt: str, k: int = 20) -> list:
    """Get top-K candidate next tokens with their probabilities."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].cpu().numpy()

    probs = softmax(logits)
    top_indices = np.argsort(probs)[-k:][::-1]

    candidates = []
    for idx in top_indices:
        token = tokenizer.decode([idx])
        prob = float(probs[idx])
        candidates.append({
            'token': token,
            'probability': prob,
            'token_id': int(idx)
        })

    return candidates

# ============================================================================
# GENERATE DATA
# ============================================================================

all_data = []

for prompt in tqdm(PURE_LANGUAGE_PROMPTS, desc="Processing prompts"):
    candidates = get_top_k_tokens(prompt, k=20)

    for candidate in candidates:
        full_text = prompt + candidate['token']

        all_data.append({
            'prompt': prompt,
            'expected_type': 'language',  # These are ALL language prompts
            'next_token': candidate['token'],
            'probability': candidate['probability'],
            'full_text': full_text,
            'label': 'language'  # Label ALL as language, even if token looks like code
        })

# ============================================================================
# CREATE DATAFRAME
# ============================================================================

df_new = pd.DataFrame(all_data)

print(f"\n✅ Generated {len(df_new)} new training examples")
print(f"\nSample data:")
print(df_new.head(20))

# Check for code-like tokens in the data
code_like_tokens = df_new[df_new['next_token'].str.contains('`|#|{|}|<|>|\[|\]', regex=True, na=False)]
print(f"\n📊 Statistics:")
print(f"   Total examples: {len(df_new)}")
print(f"   Code-like tokens (but labeled LANGUAGE): {len(code_like_tokens)}")
print(f"   Unique prompts: {df_new['prompt'].nunique()}")

if len(code_like_tokens) > 0:
    print(f"\n🎯 Examples of code-like tokens labeled as LANGUAGE:")
    print(code_like_tokens[['prompt', 'next_token', 'probability', 'label']].head(10))

# ============================================================================
# SAVE
# ============================================================================

output_file = '/Users/nishan/reposynth/research/training_data_active_learning.csv'
df_new.to_csv(output_file, index=False)
print(f"\n💾 Saved to: {output_file}")

# ============================================================================
# MERGE WITH EXISTING DATA
# ============================================================================

existing_file = '/Users/nishan/reposynth/research/training_data_labeled.csv'
print(f"\n🔗 Loading existing training data from: {existing_file}")
df_existing = pd.read_csv(existing_file)

print(f"   Existing: {len(df_existing)} examples")
print(f"   New: {len(df_new)} examples")

# Combine
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
print(f"   Combined: {len(df_combined)} examples")

# Save combined
combined_file = '/Users/nishan/reposynth/research/training_data_combined.csv'
df_combined.to_csv(combined_file, index=False)
print(f"\n💾 Saved combined data to: {combined_file}")

# Show label distribution
print(f"\n📊 Combined Label Distribution:")
print(df_combined['label'].value_counts())
print(f"\n✅ Done!")
