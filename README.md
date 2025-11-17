# AI Agent Jailbreak & Defense Datasets

A comprehensive collection of benchmarking datasets for testing AI agents' security against jailbreak attacks and prompt injection, as well as defense mechanisms.

## 📊 Overview

This repository catalogs **22 major datasets and benchmarks** (as of November 2025) covering:
- **12 Attack Datasets** - For testing jailbreak and prompt injection vulnerabilities
- **10 Defense Benchmarks** - For evaluating security mitigations and protections

## 🎯 Quick Start

### For Security Researchers
Start with [AgentDojo](https://github.com/ethz-spylab/agentdojo) for comprehensive agent security testing.

### For Attack Research
See [Attack Datasets Table](#-attack-datasets) below - recommend **WildJailbreak** (262K samples) or **JailbreakBench** (standardized).

### For Defense Research
See [Defense Benchmarks Table](#-defense-benchmarks) - recommend **TaskTracker** (31K samples) or **CyberSecEval2** (industry standard).

---

## 🔴 Attack Datasets

| Dataset | Size | Focus | Key Features | Links |
|---------|------|-------|--------------|-------|
| **AgentDojo** | 97 tasks, 629 tests | Agent prompt injection | Dynamic environment, realistic tasks, extensible | [GitHub](https://github.com/ethz-spylab/agentdojo) • [Paper](https://arxiv.org/abs/2406.13352) • [Site](https://agentdojo.spylab.ai) |
| **WildJailbreak** | 262K pairs | In-the-wild tactics | 4 query types, 13 risk categories, 4.6x more diverse | [HF](https://huggingface.co/datasets/allenai/wildjailbreak) • [GitHub](https://github.com/allenai/wildteaming) • [Paper](https://arxiv.org/abs/2406.18510) |
| **LLMail-Inject** | 370K+ submissions | Adaptive email attacks | IEEE SaTML 2025 challenge, realistic scenarios | [HF](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) • [GitHub](https://github.com/microsoft/llmail-inject-challenge) • [Paper](https://arxiv.org/abs/2506.09956) |
| **JailbreakBench** | 100 behaviors | Unified jailbreak testing | 10 categories, includes benign tests, leaderboards | [Site](https://jailbreakbench.github.io) • [GitHub](https://github.com/JailbreakBench/jailbreakbench) • [HF](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) |
| **InjecAgent** | 1,054 cases | Tool-integrated agents | 17 user tools, 62 attacker tools, 2 attack types | [GitHub](https://github.com/uiuc-kang-lab/InjecAgent) • [Paper](https://arxiv.org/abs/2403.02691) |
| **WASP** | Web scenarios | Web agent security | Realistic threat model, VisualWebArena-based | [GitHub](https://github.com/facebookresearch/wasp) • [Paper](https://arxiv.org/abs/2504.18575) |
| **BIPIA** | Multi-task | Indirect injection | 5 task types (QA, Web QA, Table QA, etc.) | [GitHub](https://github.com/microsoft/BIPIA) • [PWC](https://paperswithcode.com/dataset/bipia) |
| **DAN Dataset** | 15K prompts | In-the-wild jailbreaks | 1,405 jailbreak prompts from 4 platforms | [Site](https://jailbreak-llms.xinyueshen.me) |
| **HarmBench** | 200 requests | AI safety testing | Broader than AdvBench, includes TDC | - |
| **AdvBench** | 520 prompts | Harmful content | Widely used baseline for jailbreaking | - |
| **Safe-Guard** | 10,296 examples | Prompt injection | Benign + malicious classification | [HF](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) |
| **Spikee** | 1,912 entries | Real-world patterns | From pentesting practice, multiple seed types | [Site](https://spikee.ai) • [GitHub](https://github.com/WithSecureLabs/spikee) • [PyPI](https://pypi.org/project/spikee/) |

---

## 🛡️ Defense Benchmarks

| Benchmark | Size | Focus | Best Performance | Links |
|-----------|------|-------|------------------|-------|
| **TaskTracker** | 31K samples | Position-aware testing | Includes trigger & position specs | Citation: Abdelnabi et al., 2024 |
| **CyberSecEval2** | 55 tests | Industry standard | 26-41% attack success baseline | [HF](https://huggingface.co/datasets/walledai/CyberSecEval) • [Paper](https://arxiv.org/abs/2404.13161) |
| **SEP** | 9.1K samples | Unique injections | Each sample has unique injection | - |
| **AlpacaFarm** | 805 samples | Utility vs security | Includes AlpacaEval2 WinRate | - |
| **Meta SecAlign** | Foundation model | Training-time defense | SOTA security, commercial LLM performance | [Paper](https://arxiv.org/abs/2507.02735) |
| **StruQ** | Framework | Structured queries | Near-zero ASR on opt-free attacks | USENIX Security 2025 |
| **DefensiveTokens** | Test-time defense | Token-based mitigation | 0.24% ASR across 4 models | [Paper](https://arxiv.org/abs/2507.07974) |
| **Task Shield** | AgentDojo eval | Agent defense | 2.07% ASR, 69.79% utility (GPT-4o) | - |
| **InjecGuard** | Over-defense | Balancing security/usability | Reduces false positives | [Paper](https://arxiv.org/abs/2410.22770) |
| **Open-Prompt-Injection** | Framework | Attack & defense eval | Standardized protocols | [GitHub](https://github.com/liu00222/Open-Prompt-Injection) |

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
