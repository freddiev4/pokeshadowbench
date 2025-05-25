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
    "openai/o4-mini-2025-04-16": "o4-Mini",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "openai/gpt-4o-2024-11-20": "GPT-4o",
    "anthropic/claude-opus-4-20250514": "Claude Opus",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
    "anthropic/claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "anthropic/claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "google/gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash-preview-04-17": "Gemini 2.5 Flash"
}

# Collect all models and Pokemon
for provider, provider_models in data.items():
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
for provider, provider_models in data.items():
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