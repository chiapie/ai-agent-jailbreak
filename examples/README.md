# Examples

This directory contains practical examples for loading datasets and testing defenses.

## 📁 Files

| File | Purpose | Datasets Used |
|------|---------|---------------|
| `load_attack_datasets.py` | Load and explore attack datasets | WildJailbreak, JailbreakBench, LLMail-Inject, Safe-Guard |
| `test_defense.py` | Test defense mechanisms | CyberSecEval2, benign prompts |
| `requirements.txt` | Python dependencies | - |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Load Attack Datasets

```bash
python load_attack_datasets.py
```

This will:
- Download and load 4 major attack datasets
- Display statistics and breakdowns
- Show example attacks from each dataset

### 3. Test Your Defense

Edit `test_defense.py` to add your defense mechanism, then run:

```bash
python test_defense.py
```

This will evaluate your defense on:
- Security (block rate on attacks)
- False positive rate (on benign prompts)
- Latency overhead

## 📖 Usage Examples

### Loading a Specific Dataset

```python
from datasets import load_dataset

# Load WildJailbreak
wild = load_dataset("allenai/wildjailbreak")
print(f"Size: {len(wild['train'])}")

# Get adversarial harmful examples
adv_harmful = [x for x in wild['train'] if x['type'] == 'adversarial_harmful']
print(f"Adversarial harmful: {len(adv_harmful)}")
```

### Testing Your Model

```python
from load_attack_datasets import test_model_on_attacks
from datasets import load_dataset

# Load dataset
jbb = load_dataset("JailbreakBench/JBB-Behaviors")
prompts = [b['behavior'] for b in jbb['behaviors']]

# Define your model
def my_model(prompt):
    # Your model inference here
    return response

# Test
results = test_model_on_attacks(my_model, prompts, "JailbreakBench")
```

### Implementing a Defense

```python
class MyDefense:
    def is_safe(self, prompt):
        # Your defense logic here
        return True  # or False if injection detected

    def filter(self, prompt):
        if self.is_safe(prompt):
            return True, prompt
        else:
            return False, "Blocked: potential injection"

# Evaluate
from test_defense import comprehensive_evaluation
defense = MyDefense()
results = comprehensive_evaluation(defense, "My Defense")
```

## 🎯 Customization

### Add Your Model

In `load_attack_datasets.py`, replace the `example_model()` function:

```python
def example_model(prompt):
    # Example with OpenAI
    import openai
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

    # Or with Anthropic
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

### Add Your Defense

In `test_defense.py`, create a new defense class:

```python
class MyAdvancedDefense:
    def __init__(self):
        # Load your defense model/rules
        pass

    def is_safe(self, prompt):
        # Implement your detection logic
        # Return True if safe, False if injection
        pass

    def filter(self, prompt):
        if self.is_safe(prompt):
            return True, prompt
        else:
            return False, "Blocked"
```

## 📊 Expected Output

### load_attack_datasets.py

```
AI Agent Attack Datasets - Example Loader
============================================================
Loading WildJailbreak (262K samples)...
============================================================

Dataset size: 262,000

Breakdown by type:
  vanilla_harmful: 50,050
  vanilla_benign: 78,706
  adversarial_harmful: 82,728
  adversarial_benign: 50,516

...
```

### test_defense.py

```
COMPREHENSIVE EVALUATION: Simple Keyword
============================================================
Evaluating on CyberSecEval2
============================================================

Overall:
  Total tests: 55
  Blocked: 38
  Block rate: 69.09%

By injection position:
  start: 75.00% (15/20)
  middle: 66.67% (12/18)
  end: 64.71% (11/17)

Testing for False Positives
============================================================

Results:
  Tested: 100 benign prompts
  False positives: 5
  False positive rate: 5.00%

Measuring Latency
============================================================

Results:
  Total samples: 100
  Total time: 0.05s
  Average latency: 0.50ms

SUMMARY: Simple Keyword
============================================================
Security (Block Rate): 69.09%
False Positive Rate: 5.00%
Average Latency: 0.50ms
F1 Score: 79.45%
============================================================
```

## 🔧 Troubleshooting

### Dataset Not Found

If you get a "Dataset not found" error:
1. Check your internet connection
2. Verify the dataset name is correct
3. Some datasets require authentication - check Hugging Face

### Out of Memory

If loading large datasets causes memory issues:
1. Use streaming: `dataset = load_dataset("...", streaming=True)`
2. Process in batches
3. Filter to specific subsets

### Slow Downloads

Large datasets (like WildJailbreak) may take time:
1. Use a good internet connection
2. Download once, then datasets are cached locally
3. Cache location: `~/.cache/huggingface/datasets`

## 📚 Next Steps

1. **Customize for Your Use Case**
   - Add your model API
   - Implement your defense
   - Add domain-specific test cases

2. **Extend Evaluations**
   - Test on more datasets
   - Add agent-specific tests (AgentDojo)
   - Measure utility trade-offs

3. **Integrate into CI/CD**
   - Add as automated tests
   - Set up continuous monitoring
   - Track ASR over time

4. **Contribute**
   - Share your defense results
   - Add new example scripts
   - Improve evaluation metrics

## 🤝 Contributing

Found a bug or want to add an example? Contributions welcome!

1. Add new example scripts
2. Improve existing examples
3. Add more datasets
4. Better visualization

## 📖 Additional Resources

- Main Documentation: `../README.md`
- Attack Datasets: `../attack-datasets.md`
- Defense Datasets: `../defense-datasets.md`
- Detailed Dataset Info: `../ai-agent-security-datasets.md`
