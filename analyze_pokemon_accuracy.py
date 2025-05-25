import json
from collections import defaultdict

# Read the JSON file
with open('results/llm-results-2025-05-24_19-13-58.json', 'r') as f:
    data = json.load(f)

# Dictionary to store results for each Pokemon
pokemon_results = defaultdict(lambda: {'correct': 0, 'total': 0})

# Process all models and their responses
for provider, provider_models in data.items():
    for model_name, model_data in provider_models.items():
        for response in model_data['responses']:
            pokemon = response['pokemon']
            correct = response['correct']
            
            pokemon_results[pokemon]['total'] += 1
            if correct:
                pokemon_results[pokemon]['correct'] += 1

# Calculate accuracy for each Pokemon and sort by success rate
pokemon_accuracy = []
for pokemon, results in pokemon_results.items():
    accuracy = (results['correct'] / results['total']) * 100 if results['total'] > 0 else 0
    pokemon_accuracy.append({
        'pokemon': pokemon,
        'correct': results['correct'],
        'total': results['total'],
        'accuracy': accuracy
    })

# Sort by accuracy (descending) and then by total attempts (descending) for ties
pokemon_accuracy.sort(key=lambda x: (x['accuracy'], x['total']), reverse=True)

# Print results
print("Pokémon Recognition Accuracy (sorted by success rate)")
print("=" * 60)
print(f"{'Rank':<4} {'Pokémon':<15} {'Correct':<7} {'Total':<5} {'Accuracy':<8}")
print("-" * 60)

for i, pokemon_data in enumerate(pokemon_accuracy, 1):
    print(f"{i:<4} {pokemon_data['pokemon']:<15} {pokemon_data['correct']:<7} {pokemon_data['total']:<5} {pokemon_data['accuracy']:<8.1f}%")

# Also show top 10 and bottom 10
print("\n" + "=" * 60)
print("TOP 10 MOST RECOGNIZABLE POKÉMON:")
print("=" * 60)
for i, pokemon_data in enumerate(pokemon_accuracy[:10], 1):
    print(f"{i}. {pokemon_data['pokemon']} - {pokemon_data['correct']}/{pokemon_data['total']} ({pokemon_data['accuracy']:.1f}%)")

print("\n" + "=" * 60)
print("BOTTOM 10 LEAST RECOGNIZABLE POKÉMON:")
print("=" * 60)
for i, pokemon_data in enumerate(pokemon_accuracy[-10:], len(pokemon_accuracy)-9):
    print(f"{i}. {pokemon_data['pokemon']} - {pokemon_data['correct']}/{pokemon_data['total']} ({pokemon_data['accuracy']:.1f}%)") 