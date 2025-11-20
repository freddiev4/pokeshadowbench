import json
import argparse
from collections import defaultdict

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate a markdown table from Pokemon prediction results')
parser.add_argument('json_file', help='Path to the JSON file containing the results')
args = parser.parse_args()

# Read the JSON file
with open(args.json_file, 'r') as f:
    data = json.load(f)

# Extract all unique Pokemon names and model names
all_pokemon = set()
models = []

# Create shorter model names for better table formatting
model_name_mapping = {
    "openai/o4-mini-2025-04-16": "o4-mini",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "openai/gpt-4o-2024-11-20": "GPT-4o",
    "openai/gpt-5-nano-2025-08-07": "GPT-5 Nano",
    "openai/gpt-5-mini-2025-08-07": "GPT-5 Mini",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1",
    "anthropic/claude-opus-4-1-20250805": "Claude Opus 4.1",
    "anthropic/claude-opus-4-20250514": "Claude Opus 4",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
    "anthropic/claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "anthropic/claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "anthropic/claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "google/gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash",
    "google/gemini-3-pro-preview": "Gemini 3 Pro"
}

# Handle the new format where data is under "default" -> "providers"
if "default" in data:
    providers_data = data["default"]["providers"]
else:
    # Fallback to old format if needed
    providers_data = data

# Collect all models and Pokemon
for provider, provider_models in providers_data.items():
    for model_name, model_data in provider_models.items():
        full_model_name = f"{provider}/{model_name}"
        models.append(full_model_name)
        for response in model_data['responses']:
            all_pokemon.add(response['pokemon'])

# Sort Pokemon alphabetically
all_pokemon = sorted(list(all_pokemon))

# Create a dictionary to store results
results = defaultdict(dict)

# Populate results
for provider, provider_models in providers_data.items():
    for model_name, model_data in provider_models.items():
        full_model_name = f"{provider}/{model_name}"
        
        # Initialize all Pokemon as not tested for this model
        for pokemon in all_pokemon:
            results[pokemon][full_model_name] = None
        
        # Fill in actual results
        for response in model_data['responses']:
            pokemon = response['pokemon']
            correct = response['correct']
            results[pokemon][full_model_name] = correct

# Generate markdown table
short_model_names = [model_name_mapping.get(model, model) for model in models]

output_lines = []
output_lines.append("| Pokemon | " + " | ".join(short_model_names) + " |")
output_lines.append("|---------|" + "|".join(["-" * (len(name) + 2) for name in short_model_names]) + "|")

for pokemon in all_pokemon:
    row = f"| {pokemon} |"
    for model in models:
        result = results[pokemon][model]
        if result is True:
            row += " ✅ |"
        elif result is False:
            row += " ❌ |"
        else:
            row += " - |"  # Not tested
    output_lines.append(row)

# Write to file
with open('pokemon_results_table.md', 'w') as f:
    f.write('\n'.join(output_lines))

# Also print to console
for line in output_lines:
    print(line) 