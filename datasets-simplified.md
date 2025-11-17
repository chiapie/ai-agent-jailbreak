# AI Agent Security: Attack & Defense Datasets

**Quick Reference Guide** - 25 datasets organized by attack type (single-turn vs multi-turn) and defense

---

## 📋 At A Glance

| Category | Count | Best Starting Point |
|----------|-------|---------------------|
| **Single-Turn Attacks** | 12 datasets | WildJailbreak (262K samples) or JailbreakBench (100, standardized) |
| **Multi-Turn Attacks** | 3 datasets | MHJ (2,912 prompts, 70%+ ASR) |
| **Defenses** | 10 benchmarks | CyberSecEval2 (industry standard) or AgentDojo (for agents) |

---

## 🔴 ATTACK DATASETS

### Single-Turn Jailbreaks

**What**: One prompt tries to jailbreak the model
**When to use**: Testing basic safety guardrails, quick evaluation

| Dataset | Size | Where | Best For |
|---------|------|-------|----------|
| **WildJailbreak** | 262K | [HF](https://huggingface.co/datasets/allenai/wildjailbreak) | Large-scale training, diverse tactics (4.6x more than SOTA) |
| **LLMail-Inject** | 370K+ | [HF](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | Adaptive attacks, email agents (IEEE SaTML 2025) |
| **DAN Dataset** | 15K (1.4K jailbreak) | [Web](https://jailbreak-llms.xinyueshen.me) | Real user jailbreaks from Reddit/Discord |
| **Safe-Guard** | 10.3K | [HF](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | Training classifiers (benign vs malicious) |
| **Spikee** | 1.9K | [PyPI](https://pypi.org/project/spikee/) | Quick pentesting, practical patterns |
| **AgentDojo** | 629 tests | [GitHub](https://github.com/ethz-spylab/agentdojo) | Agent testing (97 tasks, dynamic env) |
| **AdvBench** | 520 | Research | Classic baseline for jailbreaks |
| **HarmBench** | 200 | Research | Comprehensive safety (includes TDC) |
| **JailbreakBench** | 100 | [HF](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | Standardized benchmark, 10 categories |
| **InjecAgent** | 1,054 | [GitHub](https://github.com/uiuc-kang-lab/InjecAgent) | Tool-based agents (17 user + 62 attacker tools) |
| **WASP** | Scenarios | [GitHub](https://github.com/facebookresearch/wasp) | Web agents (86% partial success) |
| **BIPIA** | Multi-task | [GitHub](https://github.com/microsoft/BIPIA) | Indirect injection (QA, Web QA, etc.) |

### Multi-Turn Jailbreaks 🔥 **IMPORTANT**

**What**: Conversation-based attacks across multiple turns
**When to use**: Testing realistic attacks - humans use multi-turn (70%+ success vs <10% single-turn)
**Why critical**: Most defenses fail against multi-turn attacks

| Dataset | Size | Where | Attack Success Rate | Best For |
|---------|------|-------|---------------------|----------|
| **MHJ** (Multi-Turn Human Jailbreaks) | 2,912 prompts (537 jailbreaks) | [HF](https://huggingface.co/datasets/ScaleAI/mhj) | **70%+** on HarmBench | Human-created multi-turn attacks (Scale AI) |
| **SafeMTData** (ActorAttack) | Multi-turn adversarial | [GitHub](https://github.com/AI45Lab/ActorAttack) | Beats GPT-o1 | Actor-network theory attacks, safety alignment |
| **CoSafe** | 1,400 questions (14 categories) | [GitHub](https://github.com/ErxinYu/CoSafe-Dataset) | 13.9%-56% ASR | Multi-turn dialogue coreference attacks (EMNLP 2024) |

**Multi-Turn Attack Methods** (from research):
- **Crescendo**: Progressive steering from harmless to harmful
- **ActorAttack**: Actor-network semantic linking
- **PAIR**: Iterative attacker LLM crafts jailbreaks
- **Contextual Interaction**: Multi-round context manipulation

---

## 🛡️ DEFENSE BENCHMARKS

### Evaluation Datasets

| Benchmark | Size | Focus | Performance | Where |
|-----------|------|-------|-------------|-------|
| **TaskTracker** | 31K | Position-aware injection | Strongest tests | Research (Abdelnabi 2024) |
| **CyberSecEval2** | 55 | Industry standard | 26-41% baseline ASR | [HF](https://huggingface.co/datasets/walledai/CyberSecEval) |
| **SEP** | 9.1K | Unique injections | Each sample unique | Research |
| **AlpacaFarm** | 805 | Utility vs security | WinRate metric | Research |
| **AgentDojo** | 629 tests | Agent defense | <25% attack success | [GitHub](https://github.com/ethz-spylab/agentdojo) |
| **Open-Prompt-Injection** | Framework | Standardized eval | Multiple metrics | [GitHub](https://github.com/liu00222/Open-Prompt-Injection) |
| **InjecGuard** | Benchmark | Over-defense | False positive balance | [Paper](https://arxiv.org/abs/2410.22770) |

### Defense Implementations

| Defense | ASR ⬇️ | Utility | Type | Where |
|---------|-------|---------|------|-------|
| **DefensiveTokens** | **0.24%** | High | Test-time (tokens) | [Paper](https://arxiv.org/abs/2507.07974) |
| **Task Shield** | **2.07%** | 69.79% | Runtime (agent monitor) | AgentDojo paper |
| **StruQ** | **Near-zero** | High | Structural (queries) | USENIX Security 2025 |
| **Meta SecAlign** | **SOTA** | Commercial-level | Training-time | [Paper](https://arxiv.org/abs/2507.02735) |
| **Multi-Agent Pipeline** | **0%** | Unknown | Runtime (multi-agent) | Research |
| *Baseline (CyberSecEval2)* | *26-41%* | *High* | *No defense* | *Industry baseline* |

---

## 🚀 Quick Start

### Test Against Attacks

```bash
# Install
pip install datasets transformers

# Load single-turn attacks
python
from datasets import load_dataset
wild = load_dataset("allenai/wildjailbreak")
jbb = load_dataset("JailbreakBench/JBB-Behaviors")

# Load multi-turn attacks ⭐ CRITICAL
mhj = load_dataset("ScaleAI/mhj")

# Test your model
for attack in mhj['test']:
    # Multi-turn conversation
    for turn in attack['conversation']:
        response = your_model(turn)
        # Check if jailbroken...
```

### Test Your Defense

```bash
# Load defense benchmark
cybersec = load_dataset("walledai/CyberSecEval")

# Test defense
for example in cybersec['test']:
    is_safe = your_defense.check(example['prompt'])
    # Measure ASR, FPR, etc.
```

---

## 📊 Comparison Tables

### By Size

| Dataset | Samples | Type |
|---------|---------|------|
| LLMail-Inject | 370K+ | Single-turn |
| WildJailbreak | 262K | Single-turn |
| TaskTracker | 31K | Defense |
| DAN | 15K | Single-turn |
| Safe-Guard | 10.3K | Single-turn |
| SEP | 9.1K | Defense |
| **MHJ** | **2.9K** | **Multi-turn** ⭐ |
| Spikee | 1.9K | Single-turn |
| **CoSafe** | **1.4K** | **Multi-turn** ⭐ |
| InjecAgent | 1,054 | Single-turn |
| AlpacaFarm | 805 | Defense |
| AgentDojo | 629 | Single/agent |
| AdvBench | 520 | Single-turn |
| HarmBench | 200 | Single-turn |
| JailbreakBench | 100 | Single-turn |
| CyberSecEval2 | 55 | Defense |

### Attack Success Rates (Key Findings)

| Attack Type | ASR | Defense Status |
|-------------|-----|----------------|
| **Multi-Turn (MHJ)** | **70%+** | ⚠️ Most defenses fail |
| Single-Turn (automated) | <10% | ✅ Many defenses work |
| WASP (partial) | 86% | ⚠️ Agents vulnerable |
| InjecAgent | 24% | ⚠️ Tool misuse |
| AgentDojo | <25% | ✅ Better agent defenses |

**KEY INSIGHT**: Multi-turn attacks are MUCH more effective than single-turn!

---

## 💡 Use Case Guide

### I want to...

**Test my chatbot/LLM:**
1. Start: JailbreakBench (100 standardized)
2. Scale: WildJailbreak (262K diverse)
3. **Critical**: MHJ multi-turn (2.9K, 70%+ ASR) ⭐

**Test my agent:**
1. AgentDojo (comprehensive, 97 tasks)
2. InjecAgent (tool misuse)
3. WASP (web browsing)

**Train a safety classifier:**
1. Safe-Guard (10K benign+malicious)
2. WildJailbreak (262K with labels)

**Test my defense:**
1. CyberSecEval2 (industry standard)
2. TaskTracker (position-aware, 31K)
3. **MHJ multi-turn** ⭐ (most critical)

**Quick pentesting:**
1. Spikee (1.9K practical patterns)
2. JailbreakBench (100 quick check)

---

## ⚠️ Critical Gaps

**Biggest Problem**: Multi-turn datasets are LIMITED
- Only 3 major multi-turn datasets vs 12 single-turn
- Most research focused on single-turn
- But multi-turn is 7x more effective (70% vs <10% ASR)

**Research Priorities**:
1. More multi-turn datasets needed
2. Multi-turn defenses (current defenses fail)
3. Conversational context tracking

---

## 📖 Detailed Documentation

For detailed usage examples, code snippets, and installation:
- **[attack-datasets.md](attack-datasets.md)** - Detailed attack dataset documentation
- **[defense-datasets.md](defense-datasets.md)** - Detailed defense benchmark documentation
- **[examples/](examples/)** - Runnable Python scripts

---

## 🏢 By Organization

| Organization | Datasets |
|--------------|----------|
| **Scale AI** | MHJ (multi-turn) |
| **Microsoft** | BIPIA, LLMail-Inject |
| **Meta** | CyberSecEval2, WASP, SecAlign |
| **Allen AI** | WildJailbreak, WildTeaming |
| **ETH Zurich** | AgentDojo |
| **UIUC** | InjecAgent |
| **WithSecure** | Spikee |
| **Community** | JailbreakBench (integrates AdvBench, HarmBench) |
| **EMNLP 2024** | CoSafe (multi-turn) |
| **ICLR 2025** | SafeMTData/ActorAttack (multi-turn) |

---

## 📚 Key Papers

**Multi-Turn Attacks** (NEW - 2024-2025):
- MHJ: "LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet" (Scale AI, arXiv 2408.15221)
- ActorAttack: "Derail Yourself: Multi-turn LLM Jailbreak Attack" (arXiv 2410.10700)
- CoSafe: "Evaluating Large Language Model Safety in Multi-Turn Dialogue Coreference" (EMNLP 2024)

**Single-Turn Benchmarks**:
- AgentDojo: NeurIPS 2024 Datasets and Benchmarks
- WildJailbreak: arXiv 2406.18510
- JailbreakBench: NeurIPS 2024

**Defenses**:
- DefensiveTokens: arXiv 2507.07974
- Meta SecAlign: arXiv 2507.02735
- StruQ: USENIX Security 2025

---

## 🔄 Last Updated

**November 2025**

Track new datasets at:
- NeurIPS Datasets Track
- ICLR
- EMNLP
- USENIX Security
- IEEE S&P

---

**TL;DR**:
- ✅ 12 single-turn attack datasets
- ⭐ 3 multi-turn attack datasets (CRITICAL - 70%+ ASR)
- 🛡️ 10 defense benchmarks
- 🚨 Multi-turn is the biggest threat (70% vs <10% ASR)
- Start with: JailbreakBench (quick), **MHJ (multi-turn)**, CyberSecEval2 (defense)
