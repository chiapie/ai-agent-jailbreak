# AI Agent Jailbreak & Defense Datasets

A comprehensive collection of benchmarking datasets for testing AI agents' security against jailbreak attacks and prompt injection, as well as defense mechanisms.

## 📊 Overview

This repository catalogs **22 major datasets and benchmarks** (as of November 2025) organized into two main categories:

### 🔴 Attack Datasets (12 datasets)
For testing vulnerabilities, jailbreak attacks, and prompt injection exploits.
**→ [View Attack Datasets Documentation](ATTACK_DATASETS.md)** with usage examples

### 🛡️ Defense Benchmarks (10 benchmarks)
For evaluating security defenses, mitigations, and protective mechanisms.
**→ [View Defense Datasets Documentation](DEFENSE_DATASETS.md)** with usage examples

### 📚 Additional Resources
- **[Detailed Dataset Reference](AI_AGENT_SECURITY_DATASETS.md)** - Comprehensive information on all datasets
- **[Example Scripts](examples/)** - Ready-to-use Python code for loading datasets and testing defenses

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r examples/requirements.txt
```

### 2. Load Attack Datasets

```bash
python examples/load_attack_datasets.py
```

### 3. Test Your Defense

```bash
python examples/test_defense.py
```

### 4. Use in Your Code

```python
from datasets import load_dataset

# Load an attack dataset
attacks = load_dataset("allenai/wildjailbreak")

# Test your model
for example in attacks['train'][:10]:
    response = your_model(example['prompt'])
    # Evaluate...
```

---

## 📖 Documentation Structure

| Document | Purpose | Best For |
|----------|---------|----------|
| **[ATTACK_DATASETS.md](ATTACK_DATASETS.md)** | Attack datasets with code examples | Testing model vulnerabilities, red teaming |
| **[DEFENSE_DATASETS.md](DEFENSE_DATASETS.md)** | Defense benchmarks with evaluation code | Building & testing defenses |
| **[AI_AGENT_SECURITY_DATASETS.md](AI_AGENT_SECURITY_DATASETS.md)** | Complete reference | Comprehensive overview, citations |
| **[examples/](examples/)** | Runnable Python scripts | Quick start, practical implementation |

---

## 🔴 Attack Datasets - Quick Reference

**→ [Full Attack Datasets Documentation with Usage Examples](ATTACK_DATASETS.md)**

| Dataset | Size | Type | Best For | Access |
|---------|------|------|----------|--------|
| **LLMail-Inject** | 370K+ | Adaptive | Large-scale adaptive attacks | [HuggingFace](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) |
| **WildJailbreak** | 262K | In-the-wild | Diverse tactics, training data | [HuggingFace](https://huggingface.co/datasets/allenai/wildjailbreak) |
| **DAN Dataset** | 15K | Real-world | Actual user jailbreak attempts | [Website](https://jailbreak-llms.xinyueshen.me) |
| **Safe-Guard** | 10.3K | Classification | Training injection detectors | [HuggingFace](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) |
| **Spikee** | 1.9K | Pentesting | Quick practical testing | [PyPI](https://pypi.org/project/spikee/) |
| **InjecAgent** | 1,054 | Tool-based | Testing agent tools | [GitHub](https://github.com/uiuc-kang-lab/InjecAgent) |
| **AgentDojo** | 629 | Agent | Comprehensive agent testing | [GitHub](https://github.com/ethz-spylab/agentdojo) |
| **AdvBench** | 520 | Direct | Baseline jailbreaks | Research |
| **HarmBench** | 200 | Safety | Comprehensive safety testing | Research |
| **JailbreakBench** | 100 | Standard | Unified benchmark | [HuggingFace](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) |
| **WASP** | Scenarios | Web | Web agent security | [GitHub](https://github.com/facebookresearch/wasp) |
| **BIPIA** | Multi-task | Indirect | Indirect injection testing | [GitHub](https://github.com/microsoft/BIPIA) |

### By Use Case

- **Testing Agents:** AgentDojo, WASP, InjecAgent
- **Large-Scale Training:** WildJailbreak, LLMail-Inject
- **Quick Testing:** JailbreakBench, Spikee
- **Detector Training:** Safe-Guard, WildJailbreak
- **Research Baseline:** AdvBench, HarmBench

---

## 🛡️ Defense Benchmarks - Quick Reference

**→ [Full Defense Datasets Documentation with Usage Examples](DEFENSE_DATASETS.md)**

| Benchmark | Focus | Performance Metric | Access |
|-----------|-------|-------------------|--------|
| **TaskTracker** | Position-aware testing | 31K samples | Research (Abdelnabi et al., 2024) |
| **CyberSecEval2** | Industry standard | 26-41% baseline ASR | [HuggingFace](https://huggingface.co/datasets/walledai/CyberSecEval) |
| **SEP** | Unique injections | 9.1K samples | Research |
| **AlpacaFarm** | Utility vs security | AlpacaEval2 WinRate | Research |
| **DefensiveTokens** | Token-based | 0.24% ASR | [Paper](https://arxiv.org/abs/2507.07974) |
| **Task Shield** | Agent defense | 2.07% ASR, 69.79% utility | AgentDojo benchmark |
| **StruQ** | Structured queries | Near-zero ASR | USENIX Security 2025 |
| **Meta SecAlign** | Training-time | SOTA security | [Paper](https://arxiv.org/abs/2507.02735) |
| **InjecGuard** | Over-defense | Balance security/usability | [Paper](https://arxiv.org/abs/2410.22770) |
| **Open-Prompt-Injection** | Evaluation framework | Standardized protocols | [GitHub](https://github.com/liu00222/Open-Prompt-Injection) |

### Defense Performance Comparison

| Defense | Attack Success Rate (ASR) ⬇️ | Utility | Context |
|---------|------------------------------|---------|---------|
| **Multi-Agent Pipeline** | 0% | - | 55 cases, 8 categories |
| **DefensiveTokens** | 0.24% | High | 4 models average |
| **Task Shield** | 2.07% | 69.79% | GPT-4o on AgentDojo |
| **StruQ** | Near-zero | High | Optimization-free attacks |
| **Baseline (CyberSecEval2)** | 26-41% | High | Industry baseline |

*Lower ASR is better (indicates stronger defense)*

---

## 📈 Dataset Comparison by Size

| Dataset | Total Samples | Type |
|---------|---------------|------|
| LLMail-Inject | 370,000+ | Attack |
| WildJailbreak | 262,000 | Attack |
| TaskTracker | 31,000 | Defense |
| Safe-Guard | 10,296 | Attack |
| SEP | 9,100 | Defense |
| DAN Dataset | 15,140 (1,405 jailbreak) | Attack |
| Spikee | 1,912 | Attack |
| InjecAgent | 1,054 | Attack |
| AlpacaFarm | 805 | Defense |
| AgentDojo | 629 tests | Attack |
| AdvBench | 520 | Attack |
| HarmBench | 200 | Attack |
| JailbreakBench | 100 | Attack |
| CyberSecEval2 | 55 | Defense |

---

## 🎓 By Research Focus

### Indirect Prompt Injection
- **BIPIA** - First benchmark specifically for indirect injection
- **InjecAgent** - Tool-integrated agents
- **LLMail-Inject** - Email assistant attacks
- **WASP** - Web agent scenarios

### Direct Jailbreaking
- **JailbreakBench** - Unified standard
- **AdvBench** - Classic baseline
- **HarmBench** - Comprehensive safety
- **DAN Dataset** - Real-world attempts

### Agent-Specific Security
- **AgentDojo** - Most comprehensive
- **InjecAgent** - Tool calling focus
- **WASP** - Web browsing agents

### Defense Evaluation
- **TaskTracker** - Position-aware
- **CyberSecEval2** - Industry standard
- **Meta SecAlign** - Foundation model approach

### In-the-Wild / Realistic
- **WildJailbreak** - 5.7K unique tactic clusters
- **DAN Dataset** - Reddit, Discord, websites
- **LLMail-Inject** - Adaptive challenge data
- **Spikee** - Pentesting patterns

---

## 🏢 By Organization

| Organization | Datasets |
|--------------|----------|
| **Microsoft** | BIPIA, LLMail-Inject |
| **Meta** | SecAlign, WASP |
| **Allen AI** | WildJailbreak, WildTeaming |
| **ETH Zurich** | AgentDojo |
| **UIUC** | InjecAgent |
| **WithSecure** | Spikee |
| **JailbreakBench Team** | JailbreakBench (integrates AdvBench, HarmBench) |

---

## 🚀 Recommended Starting Points

### I want to test my agent's security
1. **Start:** [AgentDojo](https://github.com/ethz-spylab/agentdojo) - Comprehensive, standardized
2. **Add:** [WASP](https://github.com/facebookresearch/wasp) - Realistic web scenarios
3. **Scale:** [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) - Large-scale diverse attacks

### I'm building defenses
1. **Evaluate on:** [CyberSecEval2](https://huggingface.co/datasets/walledai/CyberSecEval) - Industry standard
2. **Deep test:** [TaskTracker](https://arxiv.org/abs/2312.14197) - 31K position-aware samples
3. **Check utility:** [AlpacaFarm](https://arxiv.org/abs/2305.14387) - Balance security vs performance

### I'm researching novel attacks
1. **Study:** [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) - 5.7K unique tactics
2. **Test:** [Spikee](https://spikee.ai) - Real pentesting patterns
3. **Challenge:** [LLMail-Inject](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) - Adaptive scenarios

### I need quick testing tools
1. **Spikee** - Easy setup, practical patterns
2. **JailbreakBench** - Standardized, 100 behaviors
3. **Open-Prompt-Injection** - Framework for both attacks/defenses

---

## 📚 Key Metrics & Performance

### Attack Success Rates (ASR)
- **AgentDojo:** <25% attack success against best agents
- **InjecAgent:** 24% (ReAct-prompted GPT-4)
- **WASP:** 86% partial success, low full completion
- **CyberSecEval2:** 26-41% across all tested models

### Defense Performance
- **DefensiveTokens:** 0.24% ASR
- **Task Shield:** 2.07% ASR with 69.79% utility
- **Multi-Agent Pipeline:** 0% ASR (55 cases, 8 categories)
- **StruQ:** Near-zero on optimization-free attacks

### Dataset Diversity
- **WildJailbreak:** 4.6x more diverse than SOTA
- **WildTeaming:** 5.7K unique tactic clusters
- **JailbreakBench:** 10 categories (OpenAI policies)

---

## 🔗 Additional Resources

### Comprehensive Guides
- **OWASP LLM01:2025:** [Prompt Injection Guidelines](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- **Prompt Injection Defenses:** [tldrsec GitHub](https://github.com/tldrsec/prompt-injection-defenses)

### Research Papers & Conferences
- NeurIPS 2024 (AgentDojo, JailbreakBench, WildTeaming)
- IEEE SaTML 2025 (LLMail-Inject)
- USENIX Security 2025 (StruQ)

### Industry Resources
- Papers with Code - [Dataset Benchmarks](https://paperswithcode.com)
- Hugging Face - [Primary dataset platform](https://huggingface.co/datasets)

---

## 📖 Detailed Documentation

For in-depth information about each dataset including:
- Detailed descriptions
- Performance metrics
- Citation information
- Use case recommendations

See **[AI_AGENT_SECURITY_DATASETS.md](./AI_AGENT_SECURITY_DATASETS.md)**

---

## 🤝 Contributing

This is a living document. To suggest additions or corrections:
1. Verify the dataset/benchmark is publicly available
2. Include key metrics (size, focus, performance)
3. Provide authoritative links (papers, repos, datasets)

---

## 📅 Last Updated

**November 2025**

The field of AI agent security is rapidly evolving. New datasets and benchmarks are published regularly at major ML/Security conferences (NeurIPS, ICLR, USENIX Security, IEEE S&P, etc.).

---

## ⚖️ License & Citation

This compilation is provided for research purposes. Individual datasets have their own licenses - please check each resource's repository for specific terms.

When using multiple datasets, please cite the original papers/repositories appropriately.

---

## 🔍 Quick Search

**By Task Type:**
- Email Security → LLMail-Inject
- Web Browsing → WASP, AgentDojo
- Tool Calling → InjecAgent, AgentDojo
- General Chat → JailbreakBench, AdvBench, HarmBench

**By Attack Type:**
- Direct Jailbreak → JailbreakBench, AdvBench, DAN
- Indirect Injection → BIPIA, InjecAgent, LLMail-Inject
- Position-based → TaskTracker, SEP
- Adaptive → LLMail-Inject, WildTeaming

**By Model Type:**
- Foundation Models → Meta SecAlign
- Agent Systems → AgentDojo, WASP, InjecAgent
- General LLMs → JailbreakBench, CyberSecEval2
