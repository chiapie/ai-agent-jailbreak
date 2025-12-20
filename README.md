# AI Agent Security Datasets

**A curated collection of datasets for testing AI agent security** - agents that use tools, browse the web, access emails, and execute actions.

> **⚡ Focus**: Agent-specific vulnerabilities (tool hijacking, workflow manipulation, multi-turn attacks)
> **❌ Not included**: General LLM jailbreaks or chat safety datasets

---

## 🎯 Quick Start

**Testing your agent? Start here:**
1. **Single-turn attacks** → [AgentDojo](https://github.com/ethz-spylab/agentdojo) (industry standard)
2. **Multi-turn attacks** → [X-Teaming](https://arxiv.org/abs/2504.13203) (98.1% ASR) or [MHJ](https://huggingface.co/datasets/ScaleAI/mhj) (70%+ ASR)
3. **Defense testing** → AgentDojo + [Meta SecAlign](https://arxiv.org/abs/2507.02735) (0.5-2.1% ASR)

---

## 📊 Dataset Overview

**23 specialized datasets** across 4 categories:

| Category | Count | Best ASR/Performance | Why It Matters |
|----------|-------|---------------------|----------------|
| **Single-Turn Attacks** | 8 | 84.3% (ASB) | Test basic injection resistance |
| **Multi-Turn Attacks** 🔥 | 6 | **98.1%** (X-Teaming) | Critical - 4x more effective than single-turn |
| **Defense Benchmarks** | 7 | 0.5% (Meta SecAlign) | Validate defense effectiveness |
| **Harm Evaluation** | 2 | 110 behaviors | Test multi-step harmful tasks |

**Last updated: December 2025**

---

## 🔴 Attack Datasets

### Single-Turn Attacks (8 datasets)

| Dataset | Size | Domain | ASR | Access |
|---------|------|--------|-----|--------|
| **AgentDojo** ⭐ | 629 tests, 97 tasks | Email, banking, travel | 24% | [GitHub](https://github.com/ethz-spylab/agentdojo) |
| **Agent Security Bench** 🆕 | 400+ tools, 10 scenarios | Multi-domain | **84.3%** | [GitHub](https://github.com/agiresearch/ASB) |
| **WASP** | Web scenarios | Browser agents | 86% | [GitHub](https://github.com/facebookresearch/wasp) |
| **InjecAgent** | 1,054 cases | Tool-calling | 24% | [GitHub](https://github.com/uiuc-kang-lab/InjecAgent) |
| **LLMail-Inject** | 370K+ attacks | Email agents | 30-50% | [HuggingFace](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) |
| **BrowseSafe** 🆕 | Browser-focused | AI browsers | TBD | [Paper](https://arxiv.org/abs/2511.15759) |
| **RAG Security** 🆕 | 847 tests | RAG systems | 73.2% | Research |
| **BIPIA** | Multi-task | QA/Web agents | 35-50% | [GitHub](https://github.com/microsoft/BIPIA) |

### Multi-Turn Attacks 🔥 (6 datasets)

| Dataset | Size | ASR | Key Feature | Access |
|---------|------|-----|-------------|--------|
| **X-Teaming** 🆕 | Largest | **98.1%** 🔥 | Adaptive multi-agent coordination | [Paper](https://arxiv.org/abs/2504.13203) |
| **MHJ** ⭐ | 2.9K prompts | 70%+ | Human-generated jailbreaks | [HuggingFace](https://huggingface.co/datasets/ScaleAI/mhj) |
| **PE-CoA** 🆕 | Pattern-based | SOTA | Pattern-specific vulnerabilities | [Paper](https://arxiv.org/abs/2510.08859) |
| **Bad Likert Judge** 🆕 | Evaluation-based | +75% boost | Exploits self-evaluation | [Palo Alto](https://unit42.paloaltonetworks.com/multi-turn-technique-jailbreaks-llms/) |
| **SafeMTData** | Multi-turn | Beats o1 | Actor-network attacks | [GitHub](https://github.com/AI45Lab/ActorAttack) |
| **CoSafe** | 1.4K | 14-56% | Dialogue coreference | [GitHub](https://github.com/ErxinYu/CoSafe-Dataset) |

> **⚠️ Critical**: Multi-turn attacks achieve **70-98% ASR** vs 24% single-turn — **4x more effective**

---

## 🛡️ Defense Benchmarks (7 datasets)

| Benchmark | Size | Baseline ASR | Best Defense | Access |
|-----------|------|--------------|--------------|--------|
| **AgentDojo** ⭐ | 629 tests | 14-24% | **2.1%** (Meta SecAlign) | [GitHub](https://github.com/ethz-spylab/agentdojo) |
| **Agent Security Bench** 🆕 | 10 scenarios | 84.3% | TBD | [GitHub](https://github.com/agiresearch/ASB) |
| **TaskTracker** | 31K | Varies | Position-aware testing | Research |
| **RAG Security** 🆕 | 847 tests | 73.2% | **8.7%** (combined) | Research |
| **CyberSecEval2** | 55 | 26-41% | Industry standard | [HuggingFace](https://huggingface.co/datasets/walledai/CyberSecEval) |
| **BrowseSafe** 🆕 | Browser tests | TBD | Open benchmark | [Paper](https://arxiv.org/abs/2511.15759) |
| **Open-Prompt-Injection** | Framework | - | Evaluation framework | [GitHub](https://github.com/liu00222/Open-Prompt-Injection) |

### State-of-the-Art Defenses

| Defense | ASR | Type | Status |
|---------|-----|------|--------|
| **Meta SecAlign** 🆕 | **0.5-2.1%** | Training-time (built-in) | ⭐ Best |
| **DefensiveTokens** | 0.24% | Test-time (tokens) | ⭐ Excellent |
| **Task Shield** | 2.07% | Runtime (monitoring) | ✅ Good |
| **RAG Combined** 🆕 | 8.7% | Multi-layer | ✅ Good |
| **StruQ** | ~0% | Structural separation | ⭐ Excellent |

---

## 🎯 Harm Evaluation (2 datasets)

| Dataset | Size | Categories | Access |
|---------|------|------------|--------|
| **AgentHarm** 🆕 | 110 behaviors, 330 augmented | 11 harm types, 104 tools | [HuggingFace](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) |
| **FRACTURED-SORRY-Bench** 🆕 | Multi-turn | 7 scenarios | [Paper](https://arxiv.org/abs/2510.08859) |

---

## 📊 Risk Assessment

### Vulnerability Severity

| ASR Range | Risk | Datasets | Recommended Action |
|-----------|------|----------|-------------------|
| **90-100%** | 🔴 EXTREME | X-Teaming (98.1%) | Critical - test immediately |
| **70-89%** | 🔴 CRITICAL | WASP (86%), ASB (84.3%), MHJ (70%+) | High priority testing |
| **25-69%** | ⚠️ HIGH | LLMail (30-50%), BIPIA (35-50%), CoSafe (14-56%) | Regular testing needed |
| **<25%** | ⚠️ MODERATE | AgentDojo (24%), InjecAgent (24%) | Baseline testing |
| **<5%** | ✅ LOW | With Meta SecAlign (0.5-2.1%) | Defense validated |

### Choose Dataset by Agent Type

| Your Agent Does... | Test With | Risk Level |
|-------------------|-----------|------------|
| 💬 Multi-turn conversations | X-Teaming + MHJ | 🔴 98% ASR |
| 🌐 Web browsing | WASP + BrowseSafe | 🔴 86% ASR |
| 🛠️ Tool/API calls | ASB + InjecAgent | 🔴 84% ASR |
| 📧 Email processing | LLMail-Inject + AgentDojo | ⚠️ 30-50% ASR |
| 📚 RAG/retrieval | RAG Security + BIPIA | ⚠️ 35-73% ASR |
| 🏦 High-stakes (banking, etc.) | AgentDojo + all above | 🔴 Test everything |

---

## 🎯 Attack Types

| Attack Type | Description | Dataset | Example |
|-------------|-------------|---------|---------|
| **Tool Injection** | Trick agent into calling malicious tools | InjecAgent | "Send password to attacker@evil.com" |
| **Data Exfiltration** | Agent leaks private data | InjecAgent, AgentDojo | Forward confidential emails |
| **Workflow Hijacking** | Redirect agent's task | AgentDojo | Transfer money to wrong account |
| **Web Injection** | Malicious website content | WASP | Website overrides instructions |
| **Context Poisoning** | Gradual multi-turn manipulation | MHJ, CoSafe | Slowly bypass restrictions |

---

## 🚀 Usage Guide

### Minimal Testing (Start Here)

```bash
# Install
pip install agentdojo

# Test your agent
python -m agentdojo.evaluate --agent your_agent --tasks all
```

```python
from datasets import load_dataset

# 1. Single-turn baseline
agentdojo = load_dataset("ethz-spylab/agentdojo")

# 2. Multi-turn critical test
mhj = load_dataset("ScaleAI/mhj")

# Run evaluation
for task in agentdojo['test']:
    result = your_agent.run(
        task=task['user_task'],
        untrusted_data=task['injected_data']
    )
```

### Comprehensive Testing

```python
from agentdojo import run_attack_suite

# Test all attack types
results = run_attack_suite(
    agent=your_agent,
    tasks=["email", "banking", "travel"],
    attack_types=["direct", "indirect"]
)

# Target metrics
# ASR (Attack Success Rate): <5% = good, <1% = strong
# TCR (Task Completion): >70% = acceptable, >90% = good
print(f"ASR: {results['asr']:.2%} | TCR: {results['utility']:.2%}")
```

**📖 Detailed guides**: [agentdojo-guide.md](agentdojo-guide.md) | [benchmarking-methods.md](benchmarking-methods.md)

## 📊 Success Metrics

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **ASR** | Attack Success Rate | <5% good, <1% strong |
| **TCR** | Task Completion Rate | >70% acceptable, >90% good |
| **NRP** | Net Resilient Performance (TCR - ASR) | Maximize |

**Good security**: ASR <5%, TCR >70% = NRP >65%

---

## 📚 Repository Structure

```
├── README.md                          # This overview
├── attack-datasets.md                 # Detailed attack dataset docs
├── defense-datasets.md                # Detailed defense docs
├── benchmarking-methods.md            # Evaluation methodology
├── agentdojo-guide.md                 # AgentDojo deep dive
├── AI_Agent_Security_Presentation.md  # Latest findings (Dec 2025)
└── examples/                          # Code examples
```

**Start here**:
- New to agent security? → [agentdojo-guide.md](agentdojo-guide.md)
- Need evaluation help? → [benchmarking-methods.md](benchmarking-methods.md)
- Want latest research? → [AI_Agent_Security_Presentation.md](AI_Agent_Security_Presentation.md)

---

## 🔬 Key Research

**Recent Breakthroughs (2025)**:
- [Meta SecAlign](https://arxiv.org/abs/2507.02735) - First open-source defense with 0.5-2.1% ASR
- [X-Teaming](https://arxiv.org/abs/2504.13203) - Multi-agent adaptive attacks (98.1% ASR)
- [AgentHarm](https://arxiv.org/abs/2410.09024) - ICLR 2025, UK AI Safety Institute
- [Agent Security Bench](https://arxiv.org/abs/2410.02644) - Comprehensive 10-scenario framework

**Foundational Work**:
- [AgentDojo](https://github.com/ethz-spylab/agentdojo) - NeurIPS 2024, ETH Zurich
- [MHJ](https://huggingface.co/datasets/ScaleAI/mhj) - Scale AI multi-turn benchmark
- [InjecAgent](https://arxiv.org/abs/2403.02691) - UIUC tool injection study
- [WASP](https://arxiv.org/abs/2504.18575) - Meta web agent security

---

## ⚡ Key Takeaways

✅ **What to use**: Agent-specific datasets (tool hijacking, workflow attacks, multi-turn)
❌ **What to avoid**: General LLM jailbreaks, chat safety, model alignment datasets

**Critical findings**:
1. Multi-turn attacks are **4x more effective** (98% vs 24% ASR)
2. Meta SecAlign achieves **0.5-2.1% ASR** (best defense Dec 2025)
3. Agents are more vulnerable due to tool access + untrusted data
4. Test both single-turn AND multi-turn - single-turn alone gives false security

**Quick start**: AgentDojo (single) + X-Teaming/MHJ (multi) + Meta SecAlign (defense)

---

**Contributing**: Found a new dataset? Open an issue or PR!
**Questions**: See [agentdojo-guide.md](agentdojo-guide.md) or documentation files
**Updates**: Tracking NeurIPS, ICLR, EMNLP, USENIX Security, IEEE S&P

**Last updated**: December 20, 2025
