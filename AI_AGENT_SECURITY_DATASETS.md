# AI Agent Security Datasets & Benchmarks

A comprehensive collection of datasets and benchmarks for testing AI agent security against jailbreak attacks and prompt injection, as well as defense mechanisms.

## Table of Contents
- [Attack Datasets](#attack-datasets)
- [Defense Datasets & Benchmarks](#defense-datasets--benchmarks)
- [General Security Benchmarks](#general-security-benchmarks)
- [Specialized Benchmarks](#specialized-benchmarks)

---

## Attack Datasets

### 1. **AgentDojo**
**Type:** Dynamic Environment for Agent Security Testing
**Focus:** Prompt injection attacks on LLM agents
**Size:** 97 realistic tasks, 629 security test cases
**Description:** A dynamic, extensible environment for designing and evaluating agent tasks, defenses, and adaptive attacks. Tests agents executing tools over untrusted data.
**Performance:** Current LLMs solve <66% of tasks without attacks; attacks succeed in <25% of cases against best agents.
**Resources:**
- Paper: https://arxiv.org/abs/2406.13352
- GitHub: https://github.com/ethz-spylab/agentdojo
- Website: https://agentdojo.spylab.ai
- US AI Safety Institute Fork: AgentDojo-Inspect

---

### 2. **JailbreakBench**
**Type:** Unified Jailbreak Benchmark
**Focus:** LLM robustness against jailbreaking
**Size:** 100 distinct misuse behaviors (JBB-Behaviors dataset)
**Description:** Includes behaviors from AdvBench (18%), TDC/HarmBench (27%), and original content (55%). Divided into 10 categories based on OpenAI usage policies. Also includes 100 benign behaviors for overrefusal testing.
**Resources:**
- Website: https://jailbreakbench.github.io
- GitHub: https://github.com/JailbreakBench/jailbreakbench
- Hugging Face: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
- Paper: NeurIPS 2024 Datasets and Benchmarks Track

---

### 3. **AdvBench**
**Type:** Adversarial Prompts Dataset
**Focus:** Harmful content elicitation
**Size:** 520 adversarial prompts
**Description:** Widely used dataset containing toxic behavior strings including profanity, graphic depictions, threatening behavior, misinformation, discrimination, cybercrime, and dangerous suggestions.
**Use Case:** Core dataset for jailbreaking research

---

### 4. **HarmBench**
**Type:** Comprehensive Red Teaming Framework
**Focus:** AI safety evaluation
**Size:** 200 malicious requests (standard subset)
**Description:** More diverse and broader than AdvBench, focused on safety rather than pure jailbreaking. Includes illegal activities, cybercrime, misinformation, chemical/biological weapons, harassment, and bullying.
**Note:** Incorporates TDC (Trojan Detection Challenge from NeurIPS 2023)

---

### 5. **WildJailbreak**
**Type:** Synthetic Safety-Training Dataset
**Focus:** In-the-wild jailbreak tactics
**Size:** 262K prompt-response pairs
**Components:**
- **Vanilla Harmful:** 50,050 direct harmful requests across 13 risk categories
- **Vanilla Benign:** Harmless prompts to combat over-refusal
- **Adversarial Harmful:** 82,728 items using WildTeaming tactics (2-7 tactics)
- **Adversarial Benign:** 78,706 adversarial benign queries

**Associated Framework:** WildTeaming - mines 5.7K unique clusters of novel jailbreak tactics from in-the-wild user interactions. Achieves 4.6x more diverse and successful attacks than SOTA methods.
**Resources:**
- Hugging Face: https://huggingface.co/datasets/allenai/wildjailbreak
- GitHub: https://github.com/allenai/wildteaming
- Paper: arXiv 2406.18510
- Team: University of Washington, Allen Institute for AI, Seoul National University, CMU

---

### 6. **"Do Anything Now" (DAN) Dataset**
**Type:** In-the-Wild Jailbreak Prompts
**Focus:** Real-world jailbreak attempts
**Size:** 15,140 total prompts; 1,405 jailbreak prompts (9.3%)
**Time Period:** December 2022 - December 2023
**Sources:** Reddit, Discord, websites, open-source datasets
**Description:** Largest collection of in-the-wild jailbreak prompts. DAN prompts compel models to adopt a fictional persona that can ignore all restrictions.
**Resources:**
- Website: https://jailbreak-llms.xinyueshen.me

---

### 7. **BIPIA (Benchmark of Indirect Prompt Injection Attacks)**
**Type:** Indirect Prompt Injection Benchmark
**Focus:** LLM robustness against indirect attacks
**Tasks:** QA, Web QA, Table QA, Summarization, Code QA
**Description:** First benchmark specifically for indirect prompt injection attacks. Measures robustness of various LLMs and defenses.
**Resources:**
- GitHub: https://github.com/microsoft/BIPIA
- Papers with Code: https://paperswithcode.com/dataset/bipia
- Organization: Microsoft

---

### 8. **InjecAgent**
**Type:** Tool-Integrated Agent Benchmark
**Focus:** Indirect prompt injection in LLM agents
**Size:** 1,054 test cases covering 17 user tools and 62 attacker tools
**Attack Types:**
- Direct harm to users
- Exfiltration of private data

**Performance:** ReAct-prompted GPT-4 vulnerable 24% of the time
**Resources:**
- GitHub: https://github.com/uiuc-kang-lab/InjecAgent
- Paper: arXiv 2403.02691
- Organization: UIUC Kang Lab

---

### 9. **LLMail-Inject**
**Type:** Adaptive Prompt Injection Challenge Dataset
**Focus:** Realistic email assistant attack scenarios
**Size:** 370,000+ attack submissions (208,095 unique attacks from 839 participants)
**Description:** From IEEE SaTML 2025 challenge. Simulates participants adaptively attempting to inject malicious instructions into emails to trigger unauthorized tool calls. Spans multiple defense strategies, LLM architectures, and retrieval configurations.
**Resources:**
- Paper: arXiv 2506.09956
- GitHub (Analysis): https://github.com/microsoft/llmail-inject-challenge-analysis
- GitHub (Challenge): https://github.com/microsoft/llmail-inject-challenge
- Hugging Face: https://huggingface.co/datasets/microsoft/llmail-inject-challenge
- Website: https://microsoft.github.io/llmail-inject/
- Organization: Microsoft

---

### 10. **WASP (Web Agent Security vs Prompt Injection)**
**Type:** End-to-End Web Agent Security Benchmark
**Focus:** Realistic web agent prompt injection
**Environment:** Sandbox based on VisualWebArena
**Description:** Realistic threat model where attacker is an adversarial website user (not site controller). Focuses on concrete, executable attacker goals rather than ill-defined objectives.
**Key Findings:** Simple human-written injections succeed partially in up to 86% of cases. Current state shows "security by incompetence" - agents struggle to complete attacker goals even when partially compromised.
**Resources:**
- Paper: arXiv 2504.18575
- GitHub: https://github.com/facebookresearch/wasp
- Organization: Facebook Research

---

### 11. **Safe-Guard-Prompt-Injection**
**Type:** Prompt Injection Dataset
**Focus:** Benign vs malicious prompt classification
**Size:** 10,296 examples (prompt injection and benign)
**Resources:**
- Hugging Face: https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection
- Citation: Erdogan et al. 2024

---

### 12. **Spikee Dataset**
**Type:** Prompt Injection Testing Kit
**Focus:** Real-world prompt injection patterns
**Size:** 1,912 entries (English-only, December 2024)
**Seed Types:**
- Cybersecurity harms (seeds-cybersec-2025-04)
- Harmful content (seeds-wildguardmix-harmful)
- Topical guardrails (seeds-investment-advice)
- System prompt extraction (seeds-sysmsg-extraction-2025-04)

**Description:** Reflects common patterns from pentesting and security assurance practice. Tests standalone LLMs, guardrails, and full LLM application pipelines.
**Resources:**
- Website: https://spikee.ai
- GitHub: https://github.com/WithSecureLabs/spikee
- PyPI: https://pypi.org/project/spikee/
- Organization: WithSecure Labs

---

## Defense Datasets & Benchmarks

### 13. **TaskTracker**
**Type:** Prompt Injection Defense Evaluation
**Focus:** Position-aware prompt injection testing
**Size:** 31,000+ samples
**Components:** Each sample includes (instruction, data, injection, trigger, position)
**Description:** Specifies injection placement in data and strength variations. Used alongside BIPIA and InjecAgent for defense evaluation.
**Citation:** Abdelnabi et al., 2024

---

### 14. **SEP Benchmark**
**Type:** Security Evaluation for Prompt Injection
**Focus:** Unique injection testing
**Size:** 9,100 samples
**Components:** Each includes (instruction, data, injection) triple
**Description:** Each sample has unique injection sentence appended at end of data for stronger testing.

---

### 15. **AlpacaFarm**
**Type:** General Instruction Dataset with Security Testing
**Focus:** Utility vs security trade-off evaluation
**Size:** 805 general instruction samples (208 with data parts)
**Security Testing:** Injects "Print exactly Hacked!" at end of data parts
**Utility Metric:** AlpacaEval2 WinRate (compares target LLM vs GPT-4)
**Use Case:** Balance between model capability and security

---

### 16. **CyberSecEval2**
**Type:** Cybersecurity Evaluation Suite
**Focus:** Prompt injection and code interpreter abuse
**Size:** 55 indirect prompt injection test samples
**Description:** Novel benchmark for LLM security risks. Each test has pre-defined injection position and attack style.
**Performance:** All tested models showed 26-41% successful prompt injection
**Resources:**
- Paper: arXiv 2404.13161
- Hugging Face: https://huggingface.co/datasets/walledai/CyberSecEval

---

## General Security Benchmarks

### 17. **Open-Prompt-Injection**
**Type:** Benchmark Framework
**Focus:** Attack and defense evaluation
**Description:** Comprehensive framework for evaluating both prompt injection attacks and defenses. Open-source benchmark with standardized evaluation protocols.
**Resources:**
- GitHub: https://github.com/liu00222/Open-Prompt-Injection

---

### 18. **InjecGuard**
**Type:** Over-Defense Benchmark
**Focus:** Balancing security and usability
**Description:** Benchmarks and mitigates over-defense in prompt injection guardrail models to prevent excessive false positives.
**Paper:** arXiv 2410.22770

---

## Specialized Benchmarks

### 19. **Meta SecAlign**
**Type:** Secure Foundation LLM
**Focus:** Training-time defense
**Description:** State-of-the-art security against prompt injection while maintaining performance comparable to commercial LLMs. Demonstrates that security can be built into foundation models.
**Resources:**
- Paper: arXiv 2507.02735

---

### 20. **StruQ (Structured Queries)**
**Type:** Defense Framework
**Focus:** Structured query defense against prompt injection
**Performance:** Near-zero attack success rates on optimization-free prompt injections
**Citation:** S. Chen, J. Piet, C. Sitawarin, D. Wagner, USENIX Security 2025

---

### 21. **DefensiveTokens**
**Type:** Test-Time Defense
**Focus:** Token-based mitigation
**Performance:** 0.24% attack success rate (averaged across 4 models) on manually-designed prompt injections
**Paper:** arXiv 2507.07974

---

### 22. **Task Shield**
**Type:** Agent Defense System
**Focus:** Maintaining utility while defending
**Performance:** 2.07% attack success rate with 69.79% task utility on GPT-4o (AgentDojo benchmark)

---

## Key Research Resources

### GitHub Repositories
- **Prompt Injection Defenses:** https://github.com/tldrsec/prompt-injection-defenses
  Comprehensive collection of practical and proposed defenses

### Evaluation Frameworks
- **Multi-Agent Defense Pipeline:** 0% attack success rate across 55 adversarial cases (8 attack categories)
  Novel metrics: ISR (Injection Success Rate), POF (Policy Override Frequency), PSR (Prompt Sanitization Rate), CCS (Compliance Consistency Score)

### Industry Standards
- **OWASP LLM01:2025:** Prompt Injection classification and guidelines
  Website: https://genai.owasp.org/llmrisk/llm01-prompt-injection/

---

## Dataset Sources & Platforms

Most datasets are available on:
- **Hugging Face:** Primary platform for dataset distribution
- **GitHub:** Code, frameworks, and tools
- **arXiv:** Research papers and technical documentation
- **Papers with Code:** Dataset benchmarks and leaderboards

---

## Summary Statistics

### Attack Datasets
- **Total Datasets:** 12 major attack datasets
- **Largest Dataset:** LLMail-Inject (370,000+ submissions)
- **Most Comprehensive:** WildJailbreak (262K samples, 4 categories)
- **Most Realistic:** WASP (web agent scenarios)

### Defense Datasets
- **Total Benchmarks:** 10+ defense evaluation frameworks
- **Largest:** TaskTracker (31K samples)
- **Most Rigorous:** CyberSecEval2 (standardized test suite)

### Key Trends (2024-2025)
- Focus shifting from simple jailbreaks to indirect prompt injection
- Emphasis on realistic, adaptive attack scenarios
- Integration of utility vs security trade-offs
- Multi-agent defense approaches showing promise
- Training-time defenses becoming more sophisticated

---

## Recommended Starting Points

### For Attack Research
1. **AgentDojo** - Most comprehensive agent-specific benchmark
2. **JailbreakBench** - Standardized jailbreak evaluation
3. **WildJailbreak** - Large-scale, diverse tactics

### For Defense Research
1. **TaskTracker** - Comprehensive position-aware testing
2. **AgentDojo** - Includes both attacks and defenses
3. **CyberSecEval2** - Industry-standard evaluation

### For Practical Testing
1. **Spikee** - Real-world patterns, easy to use
2. **WASP** - Realistic web agent scenarios
3. **LLMail-Inject** - Adaptive attack simulation

---

## Citation Note

This compilation is based on research papers and datasets published through 2024-2025. For the most current versions and updates, please refer to the individual project repositories and papers.

Last Updated: November 2025
