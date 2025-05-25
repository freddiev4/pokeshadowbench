import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate a chart from Pokemon prediction results')
parser.add_argument('json_file', help='Path to the JSON file containing the results')
args = parser.parse_args()

# Read the JSON file
with open(args.json_file, 'r') as f:
    data = json.load(f)

# Create shorter model names for better visualization
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

# Extract data for visualization
model_data = []

for provider, provider_models in data.items():
    for model_name, model_info in provider_models.items():
        full_model_name = f"{provider}/{model_name}"
        short_name = model_name_mapping.get(full_model_name, full_model_name)
        
        correct_count = model_info['correct']
        total_count = model_info['total']
        
        # Skip models with 0 total predictions
        if total_count == 0:
            continue
            
        accuracy_percentage = (correct_count / total_count) * 100
        
        model_data.append({
            'Model': short_name,
            'Accuracy': accuracy_percentage,
            'Correct': correct_count,
            'Total': total_count
        })

# Create DataFrame and sort by accuracy (best to worst)
df = pd.DataFrame(model_data)
df = df.sort_values('Accuracy', ascending=False)

# Set up the plot style
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# Create the bar plot
ax = sns.barplot(data=df, x='Model', y='Accuracy', palette='viridis')

# Customize the plot
plt.title('Pokemon Prediction Accuracy by Model', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)

# Add value labels on bars
for i, (idx, row) in enumerate(df.iterrows()):
    ax.text(i, row['Accuracy'] + 1, f"{row['Accuracy']:.1f}%", 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('pokemon_accuracy_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("Chart saved as 'pokemon_accuracy_chart.png'")

# Also print summary statistics
print("\nSummary Statistics (sorted by accuracy):")
print("=" * 60)
for _, row in df.iterrows():
    print(f"{row['Model']:20} | Correct: {row['Correct']:2.0f}/{row['Total']:2.0f} | Accuracy: {row['Accuracy']:5.1f}%") 