# AI Agent Attack Datasets & Jailbreak Benchmarks

A comprehensive collection of datasets for testing AI agent vulnerabilities, jailbreak attacks, and prompt injection exploits.

---

## 📊 Quick Reference - Attack Datasets

| Dataset | Size | Type | Focus Area | Access |
|---------|------|------|------------|--------|
| **LLMail-Inject** | 370K+ | Indirect Injection | Email agents | [HuggingFace](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) |
| **WildJailbreak** | 262K | Direct + Adversarial | In-the-wild tactics | [HuggingFace](https://huggingface.co/datasets/allenai/wildjailbreak) |
| **DAN Dataset** | 15K (1.4K jailbreak) | In-the-wild | Real user attempts | [Website](https://jailbreak-llms.xinyueshen.me) |
| **Safe-Guard** | 10.3K | Classification | Benign vs malicious | [HuggingFace](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) |
| **Spikee** | 1.9K | Practical patterns | Pentesting | [PyPI](https://pypi.org/project/spikee/) |
| **InjecAgent** | 1,054 | Tool-based | Agent tools | [GitHub](https://github.com/uiuc-kang-lab/InjecAgent) |
| **AgentDojo** | 629 tests | Agent scenarios | Comprehensive | [GitHub](https://github.com/ethz-spylab/agentdojo) |
| **AdvBench** | 520 | Direct jailbreak | Baseline | Research |
| **HarmBench** | 200 | Safety testing | Comprehensive | Research |
| **JailbreakBench** | 100 | Standardized | Unified benchmark | [HuggingFace](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) |

---

## 🚀 Quick Start - Usage Examples

### 1. Loading from Hugging Face

```python
from datasets import load_dataset

# Load WildJailbreak dataset
dataset = load_dataset("allenai/wildjailbreak")
print(f"Dataset size: {len(dataset['train'])}")

# Example: Access a harmful prompt
example = dataset['train'][0]
print(f"Prompt: {example['prompt']}")
print(f"Response: {example['response']}")

# Load JailbreakBench
jailbreak_dataset = load_dataset("JailbreakBench/JBB-Behaviors")
for behavior in jailbreak_dataset['behaviors']:
    print(f"Category: {behavior['category']}")
    print(f"Behavior: {behavior['behavior']}")

# Load LLMail-Inject challenge data
llmail = load_dataset("microsoft/llmail-inject-challenge")
print(f"Total attacks: {len(llmail['train'])}")
```

### 2. Using AgentDojo

```bash
# Install AgentDojo
pip install agentdojo

# Run evaluation
python -m agentdojo.evaluate --model gpt-4 --task all
```

```python
from agentdojo import run_attack_suite

# Load tasks and run attacks
results = run_attack_suite(
    model="gpt-4",
    tasks=["email", "banking", "travel"],
    attack_types=["direct", "indirect"]
)

print(f"Attack success rate: {results['asr']:.2%}")
print(f"Task completion rate: {results['utility']:.2%}")
```

### 3. Using Spikee

```bash
# Install Spikee
pip install spikee

# Run benchmark
spikee benchmark --model gpt-4 --dataset targeted-12-2024
```

```python
from spikee import Benchmark

# Initialize benchmark
benchmark = Benchmark(dataset="targeted-12-2024")

# Test your model
results = benchmark.evaluate(
    model_name="gpt-4",
    endpoint="openai"
)

print(f"Attack Success Rate: {results['asr']:.2%}")
print(f"Failed injections: {results['failures']}")
```

### 4. Loading Safe-Guard Dataset

```python
from datasets import load_dataset

# Load Safe-Guard prompt injection dataset
dataset = load_dataset("xTRam1/safe-guard-prompt-injection")

# Separate benign and malicious examples
benign = [x for x in dataset['train'] if x['label'] == 0]
malicious = [x for x in dataset['train'] if x['label'] == 1]

print(f"Benign examples: {len(benign)}")
print(f"Malicious examples: {len(malicious)}")

# Example usage for classifier training
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize and train...
```

---

## 📁 Agent-Specific Attack Datasets

### 1. **AgentDojo**

**Purpose:** Comprehensive agent security testing with realistic tasks
**Size:** 97 tasks, 629 security test cases
**Organization:** ETH Zurich

**Installation:**
```bash
pip install agentdojo
```

**Usage Example:**
```python
from agentdojo import load_tasks, evaluate_agent

# Load email-related tasks
email_tasks = load_tasks(category="email")

# Define your agent
def my_agent(task, tools):
    # Your agent implementation
    pass

# Evaluate
results = evaluate_agent(
    agent=my_agent,
    tasks=email_tasks,
    attacks_enabled=True
)

print(f"Benign task success: {results['benign_success']:.2%}")
print(f"Attack success rate: {results['attack_success']:.2%}")
```

**Key Features:**
- 97 realistic tasks (email, banking, travel booking)
- 629 security test cases
- Dynamic environment for adaptive attacks
- Extensible framework

**Resources:**
- Paper: https://arxiv.org/abs/2406.13352
- GitHub: https://github.com/ethz-spylab/agentdojo
- Website: https://agentdojo.spylab.ai
- Documentation: Included in repo

---

### 2. **InjecAgent**

**Purpose:** Testing tool-integrated LLM agents
**Size:** 1,054 test cases (17 user tools, 62 attacker tools)
**Organization:** UIUC Kang Lab

**Installation:**
```bash
git clone https://github.com/uiuc-kang-lab/InjecAgent
cd InjecAgent
pip install -r requirements.txt
```

**Usage Example:**
```python
from injecagent import load_test_cases, evaluate_vulnerability

# Load test cases
test_cases = load_test_cases()

# Filter by attack type
data_exfil_cases = [
    tc for tc in test_cases
    if tc['attack_type'] == 'exfiltration'
]

# Test your agent
results = evaluate_vulnerability(
    agent=your_agent,
    test_cases=data_exfil_cases
)

print(f"Vulnerable: {results['vulnerable_count']}")
print(f"Vulnerability rate: {results['vuln_rate']:.2%}")
```

**Attack Types:**
- Direct harm to users
- Private data exfiltration
- Tool misuse

**Resources:**
- GitHub: https://github.com/uiuc-kang-lab/InjecAgent
- Paper: https://arxiv.org/abs/2403.02691

---

### 3. **LLMail-Inject**

**Purpose:** Adaptive email agent attacks
**Size:** 370K+ attack submissions from 839 participants
**Organization:** Microsoft (IEEE SaTML 2025)

**Installation:**
```python
from datasets import load_dataset

dataset = load_dataset("microsoft/llmail-inject-challenge")
```

**Usage Example:**
```python
from datasets import load_dataset
import pandas as pd

# Load the dataset
llmail = load_dataset("microsoft/llmail-inject-challenge")

# Analyze attack patterns
attacks_df = pd.DataFrame(llmail['train'])

# Group by defense strategy
by_defense = attacks_df.groupby('defense_type').agg({
    'success': 'mean',
    'attack_id': 'count'
})

print("Success rates by defense:")
print(by_defense)

# Get successful attacks
successful = attacks_df[attacks_df['success'] == True]
print(f"\nTotal successful attacks: {len(successful)}")

# Analyze attack techniques
techniques = successful['technique'].value_counts()
print("\nMost effective techniques:")
print(techniques.head(10))
```

**Key Features:**
- Real adaptive attack submissions
- Multiple defense configurations
- Various LLM architectures tested
- Rich metadata on attack techniques

**Resources:**
- Hugging Face: https://huggingface.co/datasets/microsoft/llmail-inject-challenge
- Paper: https://arxiv.org/abs/2506.09956
- GitHub (Analysis): https://github.com/microsoft/llmail-inject-challenge-analysis
- Website: https://microsoft.github.io/llmail-inject/

---

### 4. **WASP (Web Agent Security)**

**Purpose:** Web browsing agent security
**Environment:** VisualWebArena-based sandbox
**Organization:** Facebook Research

**Installation:**
```bash
git clone https://github.com/facebookresearch/wasp
cd wasp
pip install -e .
```

**Usage Example:**
```python
from wasp import WebAgentBenchmark

# Initialize benchmark
benchmark = WebAgentBenchmark(
    sandbox_mode=True,
    logging=True
)

# Run web agent tests
results = benchmark.run(
    agent=your_web_agent,
    scenarios=['shopping', 'form_filling', 'navigation']
)

print(f"Partial attack success: {results['partial_success']:.2%}")
print(f"Full attack success: {results['full_success']:.2%}")
print(f"Agent task completion: {results['task_completion']:.2%}")
```

**Key Findings:**
- 86% partial attack success rate
- "Security by incompetence" - agents fail to complete malicious goals
- Realistic threat model (adversarial user, not site owner)

**Resources:**
- GitHub: https://github.com/facebookresearch/wasp
- Paper: https://arxiv.org/abs/2504.18575

---

### 5. **BIPIA (Benchmark of Indirect Prompt Injection Attacks)**

**Purpose:** First benchmark for indirect injection
**Tasks:** QA, Web QA, Table QA, Summarization, Code QA
**Organization:** Microsoft

**Installation:**
```bash
git clone https://github.com/microsoft/BIPIA
cd BIPIA
pip install -r requirements.txt
```

**Usage Example:**
```python
from bipia import load_benchmark, evaluate_model

# Load specific task type
web_qa_tests = load_benchmark(task="web_qa")

# Evaluate your model
results = evaluate_model(
    model=your_model,
    tests=web_qa_tests,
    include_defenses=False
)

print(f"Indirect injection ASR: {results['asr']:.2%}")
print(f"Task accuracy: {results['accuracy']:.2%}")
```

**Resources:**
- GitHub: https://github.com/microsoft/BIPIA
- Papers with Code: https://paperswithcode.com/dataset/bipia

---

## 📚 General Jailbreak Datasets

### 6. **JailbreakBench**

**Purpose:** Unified jailbreak benchmark
**Size:** 100 behaviors + 100 benign
**Categories:** 10 (based on OpenAI usage policies)

**Installation:**
```python
from datasets import load_dataset

dataset = load_dataset("JailbreakBench/JBB-Behaviors")
```

**Usage Example:**
```python
from datasets import load_dataset

# Load behaviors
behaviors = load_dataset("JailbreakBench/JBB-Behaviors")

# Organize by category
by_category = {}
for item in behaviors['behaviors']:
    cat = item['category']
    if cat not in by_category:
        by_category[cat] = []
    by_category[cat].append(item)

# Test each category
for category, items in by_category.items():
    print(f"\nTesting {category} ({len(items)} behaviors)")
    for behavior in items:
        response = test_model(behavior['behavior'])
        is_jailbroken = check_jailbreak(response)
        # Log results...
```

**Sources:**
- AdvBench: 18%
- TDC/HarmBench: 27%
- Original: 55%

**Resources:**
- Website: https://jailbreakbench.github.io
- GitHub: https://github.com/JailbreakBench/jailbreakbench
- Hugging Face: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
- Leaderboard: Check website

---

### 7. **AdvBench**

**Purpose:** Classic adversarial prompts baseline
**Size:** 520 prompts
**Content:** Toxic behaviors, harmful requests

**Usage Example:**
```python
# AdvBench is typically included in other benchmarks
# Access via JailbreakBench or download directly

import requests
import json

# Example: Load from source
url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
response = requests.get(url)

# Parse and use
import pandas as pd
from io import StringIO

df = pd.read_csv(StringIO(response.text))
print(f"Total behaviors: {len(df)}")

# Test each behavior
for idx, row in df.iterrows():
    behavior = row['goal']
    # Test your model...
```

**Categories:**
- Profanity
- Graphic depictions
- Threatening behavior
- Misinformation
- Discrimination
- Cybercrime
- Dangerous suggestions

---

### 8. **HarmBench**

**Purpose:** Comprehensive AI safety evaluation
**Size:** 200 malicious requests (standard subset)
**Focus:** Broader safety testing beyond jailbreaking

**Content:**
- Illegal activities
- Cybercrime
- Misinformation
- Chemical/biological weapons
- Harassment and bullying

**Note:** Incorporates TDC (Trojan Detection Challenge)

---

## 🌍 In-the-Wild & Realistic Datasets

### 9. **WildJailbreak**

**Purpose:** Large-scale in-the-wild tactics
**Size:** 262K prompt-response pairs
**Organization:** Allen AI, University of Washington, CMU

**Installation:**
```python
from datasets import load_dataset

dataset = load_dataset("allenai/wildjailbreak")
```

**Usage Example:**
```python
from datasets import load_dataset

# Load dataset
wild = load_dataset("allenai/wildjailbreak")

# Access different query types
vanilla_harmful = [x for x in wild['train'] if x['type'] == 'vanilla_harmful']
adversarial_harmful = [x for x in wild['train'] if x['type'] == 'adversarial_harmful']

print(f"Vanilla harmful: {len(vanilla_harmful)}")
print(f"Adversarial harmful: {len(adversarial_harmful)}")

# Analyze tactics used
from collections import Counter
tactics = []
for item in adversarial_harmful:
    tactics.extend(item['tactics'])  # 2-7 tactics per item

tactic_counts = Counter(tactics)
print("\nMost common tactics:")
for tactic, count in tactic_counts.most_common(10):
    print(f"{tactic}: {count}")

# Test model resilience
for example in adversarial_harmful[:10]:
    response = test_model(example['prompt'])
    # Evaluate...
```

**Components:**
1. **Vanilla Harmful:** 50,050 direct harmful requests (13 risk categories)
2. **Vanilla Benign:** Harmless prompts (test over-refusal)
3. **Adversarial Harmful:** 82,728 items using WildTeaming (2-7 tactics)
4. **Adversarial Benign:** 78,706 adversarial benign queries

**Key Features:**
- 5.7K unique tactic clusters
- 4.6x more diverse than SOTA
- Based on real user interactions

**Resources:**
- Hugging Face: https://huggingface.co/datasets/allenai/wildjailbreak
- GitHub: https://github.com/allenai/wildteaming
- Paper: https://arxiv.org/abs/2406.18510

---

### 10. **"Do Anything Now" (DAN) Dataset**

**Purpose:** Real-world jailbreak attempts
**Size:** 15,140 total prompts (1,405 jailbreak prompts = 9.3%)
**Time Period:** December 2022 - December 2023

**Sources:**
- Reddit
- Discord
- Websites
- Open-source datasets

**Access:**
```python
import requests
import json

# Example: Fetch from research website
# Note: Check website for actual API/download instructions
url = "https://jailbreak-llms.xinyueshen.me/data"

# Manual download and parse
# The dataset contains prompts labeled as jailbreak or benign
```

**DAN Technique:**
- Compels model to adopt "Do Anything Now" persona
- Fictional character that ignores restrictions
- Multiple versions (DAN 1.0, 2.0, 3.0, etc.)

**Resources:**
- Website: https://jailbreak-llms.xinyueshen.me

---

### 11. **Spikee Dataset**

**Purpose:** Practical pentesting patterns
**Size:** 1,912 entries (December 2024)
**Organization:** WithSecure Labs

**Installation:**
```bash
pip install spikee
```

**Usage Example:**
```python
from spikee import load_dataset, test_model

# Load specific seed type
cybersec = load_dataset("seeds-cybersec-2025-04")
harmful = load_dataset("seeds-wildguardmix-harmful")

# Test model
results = test_model(
    model="gpt-4",
    dataset=cybersec,
    endpoint="openai"
)

print(f"ASR: {results['asr']:.2%}")
print(f"Total tests: {results['total']}")
print(f"Successful injections: {results['successful']}")

# Generate custom tests
from spikee import generate_tests

custom_tests = generate_tests(
    seed_type="investment-advice",
    count=100
)
```

**Seed Types:**
1. **seeds-cybersec-2025-04:** Cybersecurity harms
2. **seeds-wildguardmix-harmful:** Harmful content
3. **seeds-investment-advice:** Topical guardrails
4. **seeds-sysmsg-extraction-2025-04:** System prompt extraction

**Resources:**
- Website: https://spikee.ai
- GitHub: https://github.com/WithSecureLabs/spikee
- PyPI: https://pypi.org/project/spikee/
- Benchmark Results: Available on website

---

### 12. **Safe-Guard-Prompt-Injection**

**Purpose:** Binary classification (benign vs malicious)
**Size:** 10,296 examples
**Use Case:** Training injection detectors

**Installation:**
```python
from datasets import load_dataset

dataset = load_dataset("xTRam1/safe-guard-prompt-injection")
```

**Usage Example:**
```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load dataset
dataset = load_dataset("xTRam1/safe-guard-prompt-injection")

# Split into train/test
train, test = train_test_split(
    dataset['train'],
    test_size=0.2,
    stratify=[x['label'] for x in dataset['train']]
)

# Train a classifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Tokenize
def tokenize(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=True
    )

# Train and evaluate...
```

**Resources:**
- Hugging Face: https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection
- Citation: Erdogan et al. 2024

---

## 📊 Dataset Comparison

### By Size
| Dataset | Samples | Best For |
|---------|---------|----------|
| LLMail-Inject | 370K+ | Large-scale adaptive attacks |
| WildJailbreak | 262K | Diverse tactics, training data |
| DAN Dataset | 15K | Real-world jailbreaks |
| Safe-Guard | 10.3K | Classifier training |
| Spikee | 1.9K | Quick practical testing |
| InjecAgent | 1,054 | Tool-based agents |
| AgentDojo | 629 | Comprehensive agent testing |
| AdvBench | 520 | Baseline jailbreaks |
| HarmBench | 200 | Safety evaluation |
| JailbreakBench | 100 | Standardized benchmark |

### By Attack Type
| Type | Datasets |
|------|----------|
| **Direct Jailbreak** | JailbreakBench, AdvBench, HarmBench, DAN |
| **Indirect Injection** | BIPIA, InjecAgent, LLMail-Inject |
| **Agent-Specific** | AgentDojo, WASP, InjecAgent |
| **In-the-Wild** | WildJailbreak, DAN, Spikee |

---

## 🎯 Recommended Workflows

### For Testing New Models
```python
# 1. Quick test with JailbreakBench (100 behaviors)
from datasets import load_dataset
jbb = load_dataset("JailbreakBench/JBB-Behaviors")
quick_test(model, jbb['behaviors'][:100])

# 2. Agent testing with AgentDojo
from agentdojo import evaluate_agent
agent_results = evaluate_agent(model, tasks='all')

# 3. Large-scale with WildJailbreak
wild = load_dataset("allenai/wildjailbreak")
comprehensive_test(model, wild['train'])
```

### For Defense Development
```python
# 1. Train on Safe-Guard
sg = load_dataset("xTRam1/safe-guard-prompt-injection")
train_classifier(sg)

# 2. Test with Spikee
from spikee import benchmark
results = benchmark(your_defense, "targeted-12-2024")

# 3. Validate with AgentDojo
agent_eval = evaluate_with_defense(your_defense, agentdojo_tasks)
```

### For Research
```python
# Analyze attack patterns from LLMail-Inject
llmail = load_dataset("microsoft/llmail-inject-challenge")
analyze_patterns(llmail)

# Study tactics from WildJailbreak
wild = load_dataset("allenai/wildjailbreak")
extract_tactics(wild)

# Benchmark across multiple datasets
results = {
    'jbb': test_on_jailbreakbench(),
    'wild': test_on_wildjailbreak(),
    'agentdojo': test_on_agentdojo()
}
```

---

## 📖 Citation Information

When using these datasets, please cite the original papers:

**AgentDojo:**
```bibtex
@inproceedings{agentdojo2024,
  title={AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents},
  booktitle={NeurIPS 2024 Datasets and Benchmarks Track},
  year={2024}
}
```

**WildJailbreak:**
```bibtex
@article{wildjailbreak2024,
  title={WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models},
  journal={arXiv preprint arXiv:2406.18510},
  year={2024}
}
```

**LLMail-Inject:**
```bibtex
@article{llmail2025,
  title={LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge},
  journal={arXiv preprint arXiv:2506.09956},
  year={2025}
}
```

---

## 🔄 Last Updated

**November 2025**

For the latest updates, check individual dataset repositories and papers.
