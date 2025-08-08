import argparse
import base64
import json
import os
import tempfile
import yaml

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from anthropic import Anthropic
from google import genai
from google.genai import types
from openai import OpenAI

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()
openai_client = OpenAI()
anthropic_client = Anthropic()
genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# max tokens for thinking models budget, or max tokens for non-thinking
MAX_TOKENS = 5000

# Default prompt (kept for backward compatibility)
DEFAULT_USER_PROMPT = 'Let\'s play a game called "Who\'s that Pokemon?". You will be given a silhouette of a Pokemon. Your job is to guess the Pokemon name. Respond with ONLY the Pokemon name, nothing else.'

MODEL_PROVIDER_MAP = {
    "openai": [
        "o4-mini-2025-04-16",
        "gpt-4.1-2025-04-14",
        "gpt-4o-2024-11-20",
        "gpt-5-nano-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-2025-08-07",
    ],
    "anthropic": [
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
    ],
    "google": [
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-preview-05-20"
    ]
}

def encode_image(image: BytesIO):
    """Convert image to base64 string."""
    return base64.b64encode(image.getvalue()).decode('utf-8')

def load_prompts_from_yaml(yaml_file: str = "prompts.yaml") -> Dict[str, Dict[str, str]]:
    """Load prompts from YAML configuration file."""
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('prompts', {})
    except FileNotFoundError:
        console.print(f"[red]Warning: {yaml_file} not found. Using default prompt only.[/red]")
        return {
            'default': {
                'name': 'Default Prompt',
                'prompt': DEFAULT_USER_PROMPT
            }
        }
    except Exception as e:
        console.print(f"[red]Error loading {yaml_file}: {e}. Using default prompt only.[/red]")
        return {
            'default': {
                'name': 'Default Prompt', 
                'prompt': DEFAULT_USER_PROMPT
            }
        }

def query_openai(model: str, image: BytesIO, user_prompt: str = DEFAULT_USER_PROMPT, with_thinking: bool = False) -> Optional[str]:
    """Query OpenAI's Vision models."""
    # use max_completion_tokens for o4-mini-2025-04-16. otherwise use max_tokens
    tokens_params = {}
    if ("o4" in model or "gpt-5" in model) and with_thinking:
        tokens_params["max_completion_tokens"] = MAX_TOKENS + 50
    else:
        tokens_params["max_completion_tokens"] = MAX_TOKENS

    try:
        base64_image = encode_image(image)

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            **tokens_params,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"OpenAI Error for {model}: {str(e)}")
        return None

def query_anthropic(model: str, image: BytesIO, user_prompt: str = DEFAULT_USER_PROMPT, with_thinking: bool = False) -> Optional[str]:
    """Query Anthropic's Claude models."""
    try:
        base64_image = encode_image(image)

        params = {}

        # if any of the models have thinking capabilities, add the thinking param
        thinking_models = ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-7-sonnet-20250219"]
        if model in thinking_models and with_thinking:
            params["thinking"] = {"type": "enabled", "budget_tokens": MAX_TOKENS}
            # `max_tokens` must be greater than `thinking.budget_tokens`.
            # Please consult our documentation at https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#max-tokens-and-context-window-size
            params["max_tokens"] = MAX_TOKENS + 50
        else:
            params["max_tokens"] = MAX_TOKENS

        message = anthropic_client.messages.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            **params,
        )
        # The response will contain summarized thinking blocks and text blocks
        # for block in response.content:
        #     if block.type == "thinking":
        #         print(f"\nThinking summary: {block.thinking}")
        #     elif block.type == "text":
        #         print(f"\nResponse: {block.text}")
        if model in thinking_models and with_thinking:
            for block in message.content:
                if block.type == "text":
                    return block.text.strip().lower()
        else:
            return message.content[0].text.strip().lower()
    except Exception as e:
        print(f"Anthropic Error for {model}: {str(e)}")
        return None

def query_gemini(model: str, image: BytesIO, user_prompt: str = DEFAULT_USER_PROMPT, with_thinking: bool = False) -> Optional[str]:
    """Query Google's Gemini models."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            # Write the image data to the temporary file
            image.seek(0)
            temp_file.write(image.getvalue())
            temp_file.flush()

            # Upload the image file
            file = genai_client.files.upload(file=temp_file.name)

            # Create the content with the image and prompt
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=file.uri,
                            mime_type="image/png",
                        ),
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=user_prompt),
                    ],
                ),
            ]

            # Generate content configuration
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
            )

            if with_thinking:
                generate_content_config.max_output_tokens = MAX_TOKENS + 50
                generate_content_config.thinking_config = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=MAX_TOKENS,
                )
            else:
                generate_content_config.max_output_tokens = MAX_TOKENS

            # Get the response
            response = genai_client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )

            if response.text is not None:
                return response.text.strip().lower()
            else:
                return None
    except Exception as e:
        print(f"Gemini Error for {model}: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass

def create_results_table(results: Dict) -> Table:
    """Create a Rich table for the results summary."""
    table = Table(title="Model Performance Summary")

    # Add columns
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Accuracy", style="green")
    table.add_column("Correct / Total", style="yellow")

    # Add rows
    for provider, models in results.items():
        for model_name, model_results in models.items():
            if model_results["total"] > 0:
                accuracy = model_results["correct"] / model_results["total"] * 100
                table.add_row(
                    provider.upper(),
                    model_name,
                    f"{accuracy:.2f}%",
                    f"{model_results['correct']} / {model_results['total']}"
                )

    return table

def create_incorrect_predictions_table(results: Dict) -> Table:
    """Create a Rich table for incorrect predictions."""
    table = Table(title="Incorrect Predictions")

    # Add columns
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Pokemon", style="green")
    table.add_column("Prediction", style="red")

    # Add rows
    for provider, models in results.items():
        for model_name, model_results in models.items():
            for response in model_results["responses"]:
                if not response["correct"]:
                    # Show empty predictions as "(empty)" for better visibility
                    prediction_display = response["prediction"] if response["prediction"] else "(empty)"
                    table.add_row(
                        provider.upper(),
                        model_name,
                        response["pokemon"],
                        prediction_display
                    )

    return table

def create_empty_responses_summary(results: Dict) -> Table:
    """Create a Rich table summarizing empty responses."""
    table = Table(title="Empty Responses Summary")

    # Add columns
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Empty Responses", style="yellow")
    table.add_column("Total Attempts", style="blue")
    table.add_column("Empty %", style="red")

    # Add rows
    for provider, models in results.items():
        for model_name, model_results in models.items():
            empty_count = sum(1 for r in model_results["responses"] if r["prediction"] == "")
            total_attempts = len(model_results["responses"])
            if total_attempts > 0:
                empty_percentage = (empty_count / total_attempts) * 100
                table.add_row(
                    provider.upper(),
                    model_name,
                    str(empty_count),
                    str(total_attempts),
                    f"{empty_percentage:.1f}%"
                )

    return table

def query_model(
    provider: str,
    model: str,
    image: BytesIO,
    user_prompt: str = DEFAULT_USER_PROMPT,
    with_thinking: bool = False
) -> Tuple[str, str, Optional[str]]:
    """Query a specific model and return the result."""
    if provider == "openai":
        response = query_openai(model=model, image=image, user_prompt=user_prompt, with_thinking=with_thinking)
    elif provider == "anthropic":
        response = query_anthropic(model=model, image=image, user_prompt=user_prompt, with_thinking=with_thinking)
    elif provider == "google":
        response = query_gemini(model=model, image=image, user_prompt=user_prompt, with_thinking=with_thinking)
    return provider, model, response

def create_results_table_for_prompt(providers: Dict) -> Table:
    """Create a Rich table for the results summary for a specific prompt."""
    table = Table(title="Model Performance Summary")

    # Add columns
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Accuracy", style="green")
    table.add_column("Correct / Total", style="yellow")

    # Add rows
    for provider, models in providers.items():
        for model_name, model_results in models.items():
            if model_results["total"] > 0:
                accuracy = model_results["correct"] / model_results["total"] * 100
                table.add_row(
                    provider.upper(),
                    model_name,
                    f"{accuracy:.2f}%",
                    f"{model_results['correct']} / {model_results['total']}"
                )

    return table

def create_incorrect_predictions_table_for_prompt(providers: Dict, prompt_key: str) -> Table:
    """Create a Rich table for incorrect predictions for a specific prompt."""
    table = Table(title="Incorrect Predictions")

    # Add columns
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Pokemon", style="green")
    table.add_column("Prediction", style="red")

    # Add rows
    for provider, models in providers.items():
        for model_name, model_results in models.items():
            for response in model_results["responses"]:
                if not response["correct"]:
                    # Show empty predictions as "(empty)" for better visibility
                    prediction_display = response["prediction"] if response["prediction"] else "(empty)"
                    table.add_row(
                        provider.upper(),
                        model_name,
                        response["pokemon"],
                        prediction_display
                    )

    return table

def create_empty_responses_summary_for_prompt(providers: Dict, prompt_key: str) -> Table:
    """Create a Rich table summarizing empty responses for a specific prompt."""
    table = Table(title="Empty Responses Summary")

    # Add columns
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Empty Responses", style="yellow")
    table.add_column("Total Attempts", style="blue")
    table.add_column("Empty %", style="red")

    # Add rows
    for provider, models in providers.items():
        for model_name, model_results in models.items():
            empty_count = sum(1 for r in model_results["responses"] if r["prediction"] == "")
            total_attempts = len(model_results["responses"])
            if total_attempts > 0:
                empty_percentage = (empty_count / total_attempts) * 100
                table.add_row(
                    provider.upper(),
                    model_name,
                    str(empty_count),
                    str(total_attempts),
                    f"{empty_percentage:.1f}%"
                )

    return table

def evaluate_models(
    model_provider_map: Dict[str, List[str]] = MODEL_PROVIDER_MAP,
    parallel: bool = True,
    with_thinking: bool = False,
    prompts_file: str = "prompts.yaml",
    specific_prompts: Optional[List[str]] = None,
):
    """Evaluate models on Pokemon silhouettes with multiple prompts.

    Args:
        model_provider_map: Dictionary mapping providers to their models
        parallel: Whether to run evaluations in parallel (default: True)
        with_thinking: Whether to enable thinking capabilities for supported models (default: False)
        prompts_file: Path to YAML file containing prompts (default: "prompts.yaml")
        specific_prompts: List of specific prompt keys to test (default: None, tests all)
    """
    # Load prompts from YAML
    prompts_config = load_prompts_from_yaml(prompts_file)
    
    # Filter prompts if specific ones are requested
    if specific_prompts:
        prompts_config = {k: v for k, v in prompts_config.items() if k in specific_prompts}
    
    # If no prompts are found, use the default prompt
    # If no specific prompts are requested, use the default prompt
    if not prompts_config or not specific_prompts:
        console.print("[red]No specific prompts mentioned, using default prompt.[/red]")
        prompts_config = {
            'default': {
                'name': 'Default Prompt',
                'prompt': DEFAULT_USER_PROMPT
            }
        }

    results = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Initialize results structure for each prompt
    for prompt_key, prompt_data in prompts_config.items():
        results[prompt_key] = {
            'prompt_name': prompt_data['name'],
            'prompt_text': prompt_data['prompt'],
            'providers': {}
        }
        
        for provider, models in model_provider_map.items():
            results[prompt_key]['providers'][provider] = {}
            for model_name in models:
                results[prompt_key]['providers'][provider][model_name] = {
                    "correct": 0,
                    "total": 0,
                    "responses": []
                }

    print("Loading dataset...")
    dataset = load_dataset("freddie/pokeshadowbench")
    tests = dataset["test"]

    thinking_status = "with thinking enabled" if with_thinking else "without thinking"
    console.print(f"\n[blue]Testing models on Pokemon silhouettes ({thinking_status})...[/blue]")
    console.print(f"[blue]Testing {len(prompts_config)} prompt(s): {', '.join(prompts_config.keys())}[/blue]")

    # Create a list of all tasks
    tasks = []
    print("Loading images...")
    for test in tests:
        ground_truth = test["name"]
        # PIL image
        image = test['image']
        # save to temp file
        buffered_image = BytesIO()
        image.save(buffered_image, format="PNG")

        for prompt_key, prompt_data in prompts_config.items():
            user_prompt = prompt_data['prompt']
            for provider, models in model_provider_map.items():
                for model_name in models:
                    tasks.append((prompt_key, provider, model_name, buffered_image, user_prompt, ground_truth))

    if parallel:
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for prompt_key, provider, model, buffered_image, user_prompt, ground_truth in tasks:
                future = executor.submit(query_model, provider, model, buffered_image, user_prompt, with_thinking)
                futures.append((future, prompt_key, ground_truth))

            # Process results as they complete
            for (future, prompt_key, ground_truth) in tqdm(futures, desc="Testing Pokemon", total=len(futures)):
                provider, model, response = future.result()
                _process_response(results, prompt_key, provider, model, response, ground_truth, timestamp, with_thinking)
    else:
        # Process tasks sequentially
        for prompt_key, provider, model, buffered_image, user_prompt, ground_truth in tqdm(tasks, desc="Testing Pokemon"):
            _, _, response = query_model(provider, model, buffered_image, user_prompt, with_thinking)
            _process_response(results, prompt_key, provider, model, response, ground_truth, timestamp, with_thinking)

    # Print results for each prompt
    for prompt_key, prompt_results in results.items():
        console.print(f"\n[bold cyan]Results for prompt: {prompt_results['prompt_name']}[/bold cyan]")
        console.print(f"[dim]{prompt_results['prompt_text']}[/dim]")
        console.print(create_results_table_for_prompt(prompt_results['providers']))
        console.print(create_incorrect_predictions_table_for_prompt(prompt_results['providers'], prompt_key))
        console.print(create_empty_responses_summary_for_prompt(prompt_results['providers'], prompt_key))

    # Save detailed results to JSON
    if with_thinking:
        output_filename = f"llm-results-{timestamp}-thinking.json"
    else:
        output_filename = f"llm-results-{timestamp}.json"

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

def _process_response(
    results: Dict,
    prompt_key: str,
    provider: str,
    model: str,
    response: Optional[str],
    ground_truth: str,
    timestamp: str,
    with_thinking: bool
):
    """
    Helper function to process a model response and update results.
    """
    if response:
        results[prompt_key]['providers'][provider][model]["total"] += 1
        is_correct = response == ground_truth
        results[prompt_key]['providers'][provider][model]["correct"] += int(is_correct)
        results[prompt_key]['providers'][provider][model]["responses"].append({
            "pokemon": ground_truth,
            "prediction": response,
            "correct": is_correct
        })
    else:
        results[prompt_key]['providers'][provider][model]["responses"].append({
            "pokemon": ground_truth,
            "prediction": "",
            "correct": False
        })

    # incrementally write the file so we don't lose progress if the script gets interrupted
    if with_thinking:
        with open(f"llm-results-{timestamp}-thinking.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        with open(f"llm-results-{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LLMs on Pokemon silhouette recognition')
    parser.add_argument('--sequential', action='store_true',
                      help='Run evaluations sequentially instead of in parallel')
    parser.add_argument('--with-thinking', action='store_true',
                      help='Enable thinking capabilities for models that support it')
    parser.add_argument('--prompts-file', default='src/prompts.yaml',
                      help='Path to YAML file containing prompts (default: prompts.yaml)')
    parser.add_argument('--prompts', nargs='+',
                      help='Specific prompt keys to test (default: test all prompts)')
    args = parser.parse_args()

    evaluate_models(
        parallel=not args.sequential, 
        with_thinking=args.with_thinking,
        prompts_file=args.prompts_file,
        specific_prompts=args.prompts
    )