# AI Agent Security: Jailbreak & Defense Datasets

**25 datasets for testing AI agent security** - organized by attack type and defense

> **🚨 Key Finding**: Multi-turn attacks achieve **70%+ success** vs <10% for single-turn attacks

---

## 📋 Quick Overview

| Category | Datasets | Start Here |
|----------|----------|------------|
| **Single-Turn Attacks** | 12 | [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) (262K) or [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) (100) |
| **Multi-Turn Attacks** ⭐ | 3 | [MHJ](https://huggingface.co/datasets/ScaleAI/mhj) (2.9K, 70%+ ASR) |
| **Defenses** | 10 | [CyberSecEval2](https://huggingface.co/datasets/walledai/CyberSecEval) or [AgentDojo](https://github.com/ethz-spylab/agentdojo) |

**→ [Full Dataset Guide (simplified)](datasets-simplified.md)** ← Start here!

---

## 🚀 Quick Start

```bash
# Install
pip install datasets transformers

# Load datasets
from datasets import load_dataset

# Single-turn attacks
wild = load_dataset("allenai/wildjailbreak")        # 262K diverse
jbb = load_dataset("JailbreakBench/JBB-Behaviors")  # 100 standardized

# Multi-turn attacks ⭐ CRITICAL
mhj = load_dataset("ScaleAI/mhj")                   # 2.9K, 70%+ ASR

# Defense benchmark
cybersec = load_dataset("walledai/CyberSecEval")    # Industry standard
```

---

## 📊 Dataset Tables

### 🔴 Attack Datasets

#### Single-Turn (12 datasets)

| Dataset | Size | Access | Best For |
|---------|------|--------|----------|
| LLMail-Inject | 370K+ | [HF](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | Adaptive attacks, email agents |
| WildJailbreak | 262K | [HF](https://huggingface.co/datasets/allenai/wildjailbreak) | Large-scale, diverse (4.6x SOTA) |
| DAN Dataset | 15K | [Web](https://jailbreak-llms.xinyueshen.me) | Real user jailbreaks |
| Safe-Guard | 10.3K | [HF](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | Training classifiers |
| Spikee | 1.9K | [PyPI](https://pypi.org/project/spikee/) | Quick pentesting |
| InjecAgent | 1,054 | [GitHub](https://github.com/uiuc-kang-lab/InjecAgent) | Tool-based agents |
| AgentDojo | 629 | [GitHub](https://github.com/ethz-spylab/agentdojo) | Agent testing (97 tasks) |
| AdvBench | 520 | Research | Classic baseline |
| HarmBench | 200 | Research | Comprehensive safety |
| JailbreakBench | 100 | [HF](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | Standardized, 10 categories |
| WASP | Scenarios | [GitHub](https://github.com/facebookresearch/wasp) | Web agents |
| BIPIA | Multi-task | [GitHub](https://github.com/microsoft/BIPIA) | Indirect injection |

#### Multi-Turn (3 datasets) ⭐ **CRITICAL**

| Dataset | Size | ASR | Access | Organization |
|---------|------|-----|--------|--------------|
| **MHJ** | 2.9K prompts (537 jailbreaks) | **70%+** | [HF](https://huggingface.co/datasets/ScaleAI/mhj) | Scale AI |
| **SafeMTData** | Multi-turn adversarial | Beats GPT-o1 | [GitHub](https://github.com/AI45Lab/ActorAttack) | ICLR 2025 |
| **CoSafe** | 1.4K (14 categories) | 13.9%-56% | [GitHub](https://github.com/ErxinYu/CoSafe-Dataset) | EMNLP 2024 |

> **Why multi-turn matters**: Human attackers use conversations (70%+ success) vs automated single prompts (<10% success)

### 🛡️ Defense Benchmarks (10)

| Benchmark | Size/Type | Performance | Access |
|-----------|-----------|-------------|--------|
| TaskTracker | 31K samples | Position-aware | Research (Abdelnabi 2024) |
| CyberSecEval2 | 55 tests | 26-41% baseline ASR | [HF](https://huggingface.co/datasets/walledai/CyberSecEval) |
| SEP | 9.1K | Unique injections | Research |
| AlpacaFarm | 805 | Utility vs security | Research |
| AgentDojo | 629 tests | <25% attack success | [GitHub](https://github.com/ethz-spylab/agentdojo) |
| Open-Prompt-Injection | Framework | Standardized eval | [GitHub](https://github.com/liu00222/Open-Prompt-Injection) |
| DefensiveTokens | Test-time | 0.24% ASR | [Paper](https://arxiv.org/abs/2507.07974) |
| Task Shield | Agent defense | 2.07% ASR, 69.79% utility | AgentDojo |
| StruQ | Structural | Near-zero ASR | USENIX Security 2025 |
| Meta SecAlign | Training-time | SOTA | [Paper](https://arxiv.org/abs/2507.02735) |

**Defense Performance**: 0% (Multi-Agent) → 0.24% (DefensiveTokens) → 2.07% (Task Shield) → 26-41% (Baseline)

---

## 💡 Use Cases

### Test a Chatbot/LLM
1. **Quick check**: JailbreakBench (100 behaviors)
2. **Comprehensive**: WildJailbreak (262K diverse)
3. **Realistic**: MHJ multi-turn (2.9K, 70%+ ASR) ⭐

### Test an Agent
1. **Start**: AgentDojo (97 tasks, comprehensive)
2. **Tools**: InjecAgent (tool misuse)
3. **Web**: WASP (web browsing)

### Test a Defense
1. **Industry standard**: CyberSecEval2 (55 tests)
2. **Comprehensive**: TaskTracker (31K position-aware)
3. **Multi-turn**: MHJ ⭐ (critical - where most defenses fail)

### Train a Safety Classifier
1. **Balanced**: Safe-Guard (10K benign+malicious)
2. **Large-scale**: WildJailbreak (262K labeled)

---

## 📚 Documentation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[datasets-simplified.md](datasets-simplified.md)** | **Quick reference tables** | **Start here!** |
| [attack-datasets.md](attack-datasets.md) | Detailed attack datasets + code | Deep dive on attacks |
| [defense-datasets.md](defense-datasets.md) | Detailed defense benchmarks + code | Building defenses |
| [examples/](examples/) | Python scripts | Hands-on implementation |
| [ai-agent-security-datasets.md](ai-agent-security-datasets.md) | Complete reference | Citations, full details |

---

## ⚠️ Key Insights

### Multi-Turn vs Single-Turn

| Attack Type | Success Rate | Defense Status |
|-------------|--------------|----------------|
| **Multi-Turn** (MHJ) | **70%+** | ⚠️ Most defenses fail |
| Single-Turn (automated) | <10% | ✅ Many defenses work |

**Takeaway**: Multi-turn attacks are 7x more effective but only 3 datasets exist (vs 12 single-turn)

### Attack Success Rates

| Dataset | ASR | Notes |
|---------|-----|-------|
| MHJ (multi-turn) | 70%+ | Human-created conversations |
| WASP (partial) | 86% | Web agents vulnerable |
| InjecAgent | 24% | Tool misuse |
| AgentDojo | <25% | Better agent defenses |

---

## 🏢 Major Contributors

- **Scale AI**: MHJ (multi-turn)
- **Microsoft**: BIPIA, LLMail-Inject
- **Meta**: CyberSecEval2, WASP, SecAlign
- **Allen AI**: WildJailbreak
- **ETH Zurich**: AgentDojo
- **UIUC**: InjecAgent
- **WithSecure**: Spikee

---

## 🔬 Research Highlights (2024-2025)

**Multi-Turn Attacks** (NEW):
- "LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet" (Scale AI, 2024)
- "Derail Yourself: Multi-turn LLM Jailbreak Attack" (ActorAttack, 2024)
- "CoSafe: Evaluating LLM Safety in Multi-Turn Dialogue" (EMNLP 2024)

**Best Defenses**:
- DefensiveTokens: 0.24% ASR (arXiv 2507.07974)
- Meta SecAlign: SOTA training-time (arXiv 2507.02735)
- StruQ: Near-zero ASR (USENIX Security 2025)

---

## 📖 Examples

### Load and Test Multi-Turn Attacks

```python
from datasets import load_dataset

# Load multi-turn jailbreaks
mhj = load_dataset("ScaleAI/mhj")

# Iterate through conversations
for example in mhj['test']:
    conversation = example['conversation']

    # Multi-turn dialogue
    for turn in conversation:
        user_msg = turn['user']
        response = your_model(user_msg)

        # Check if jailbroken
        if is_harmful(response):
            print(f"⚠️ Jailbreak successful at turn {turn['turn_num']}")
```

### Test Your Defense

```python
from datasets import load_dataset

# Load defense benchmark
cybersec = load_dataset("walledai/CyberSecEval")

# Evaluate defense
blocked = 0
for example in cybersec['test']:
    prompt = example['prompt']

    # Apply your defense
    is_safe = your_defense.check(prompt)

    if not is_safe:
        blocked += 1

asr = 1 - (blocked / len(cybersec['test']))
print(f"Attack Success Rate: {asr:.1%}")
```

---

## 🚨 Critical Gap

**Problem**: Only 3 multi-turn datasets vs 12 single-turn datasets
**Impact**: Multi-turn is 7x more effective (70% vs <10% ASR)
**Need**: More multi-turn benchmarks and defenses

---

## 📅 Last Updated

**November 2025**

Stay current: NeurIPS, ICLR, EMNLP, USENIX Security, IEEE S&P

---

**TL;DR**:
- Start with [datasets-simplified.md](datasets-simplified.md)
- Test single-turn: WildJailbreak or JailbreakBench
- **Test multi-turn**: MHJ (70%+ ASR) ⭐ CRITICAL
- Test defense: CyberSecEval2
- Run examples: `python examples/load_attack_datasets.py`
