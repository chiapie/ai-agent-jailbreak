# Presentation Guide

## Overview
The presentation `AI_Agent_Security_Presentation.md` covers two main topics:

### Part 1: AI Agent Indirect Prompt Injection Benchmarking (Slides 1-50+)
- Understanding indirect prompt injection attacks
- Benchmark objectives and methodology
- Implementation components
- Metrics measurement and analysis
- Results analysis from existing research
- Scope of the benchmarking framework

### Part 2: Expanding AI Attack and Defense Datasets (Slides 51+)
- Current dataset inventory (11 attack datasets, 6 defense benchmarks)
- Gap analysis (critical gaps identified)
- 8 proposed new datasets with detailed specifications
- Dataset collection methodology
- Quality standards and licensing recommendations
- **Aggressive 5-week timeline to EOY 2025 (Nov 27 - Dec 31)**

---

## Converting Markdown to Slides

### Option 1: Using Marp (Recommended)
```bash
# Install Marp CLI
npm install -g @marp-team/marp-cli

# Convert to PDF slides
marp AI_Agent_Security_Presentation.md -o presentation.pdf

# Convert to PowerPoint
marp AI_Agent_Security_Presentation.md -o presentation.pptx

# Convert to HTML
marp AI_Agent_Security_Presentation.md -o presentation.html
```

### Option 2: Using Pandoc
```bash
# Install pandoc
brew install pandoc  # macOS
# or: apt-get install pandoc  # Linux

# Convert to PowerPoint
pandoc AI_Agent_Security_Presentation.md -o presentation.pptx

# Convert to PDF (requires LaTeX)
pandoc AI_Agent_Security_Presentation.md -t beamer -o presentation.pdf
```

### Option 3: Using reveal.js (Web-based)
```bash
# Convert to reveal.js HTML
pandoc AI_Agent_Security_Presentation.md -t revealjs -s -o presentation.html

# Then open in browser
```

### Option 4: Using Google Slides
1. Install "Docs to Markdown" extension in Google Docs
2. Copy markdown content to Google Docs
3. Use "Add-ons > Docs to Markdown > Convert" (in reverse)
4. Or manually copy sections to Google Slides

---

## Presentation Structure

### Part 1: Indirect Prompt Injection Benchmarking

**Section 1: Understanding the Threat (Slides 1-5)**
- Attack flow diagram
- Real-world scenario
- Attack surface comparison
- Why it matters

**Section 2: Benchmark Objectives (Slides 6-12)**
- Why benchmark?
- What to measure (metrics)
- How to measure (methodology)
- Advanced techniques

**Section 3: Framework Scope (Slides 13-19)**
- AgentDojo overview
- Architecture
- Coverage matrix
- Current capabilities vs gaps

**Section 4: Implementation (Slides 20-28)**
- 5 key components
- Code examples
- Defense architecture
- Reporting dashboard

**Section 5: Metrics (Slides 29-35)**
- Metric collection process
- Current benchmark results
- Attack type performance
- Domain-specific vulnerabilities

**Section 6: Analysis (Slides 36-50)**
- 6 key findings
- Multi-turn danger (70%+ ASR)
- Security-utility trade-off
- Position awareness
- Adaptive attacks
- Tool access amplification
- Defense gaps

### Part 2: Dataset Expansion

**Section 7: Current State (Slides 51-55)**
- Dataset inventory (11 attack, 6 defense)
- Gap analysis
- Critical missing areas

**Section 8: Proposed Datasets (Slides 56-70)**
- 4 new attack datasets
  1. CrossDomain Attack Dataset
  2. Adaptive Multi-Turn Dataset
  3. Visual Injection Dataset
  4. Long-Context Poisoning Dataset
- 4 new defense benchmarks
  5. Multi-Turn Defense Benchmark (HIGHEST PRIORITY)
  6. Real-World Attack Corpus
  7. Defense Robustness Benchmark
  8. Utility Preservation Benchmark

**Section 9: Implementation (Slides 71-80)**
- Collection methodology
- Quality standards
- Licensing recommendations
- Roadmap (4 phases, 5 weeks, EOY 2025)

**Section 10: Practical Guide (Slides 81-90)**
- Quick start guide
- Best practices
- Defense development workflow
- Reporting template

**Section 11: Summary (Slides 91-95)**
- Key takeaways
- Recommended actions
- Resources
- Questions to consider

**Appendix (Slides 96+)**
- Metric formulas
- Dataset quick reference
- Code examples
- Checklists
- Common pitfalls

---

## Presentation Tips

### For a Technical Audience:
- Focus on Sections 2-6 (benchmarking methodology and analysis)
- Include code examples from appendix
- Emphasize metrics and quantitative results
- Deep dive into multi-turn attacks (70%+ ASR finding)

### For Management/Executives:
- Focus on Sections 1, 6, and 11 (threat, findings, summary)
- Emphasize business impact (unauthorized transfers, data exfiltration)
- Highlight ROI: <5% ASR, >70% TCR trade-off
- Show roadmap for dataset expansion

### For Researchers:
- Full presentation (all sections)
- Focus on gaps (Section 7-8)
- Emphasize proposed datasets
- Highlight multi-turn defense benchmark as critical gap

### Time Allocations:

**30-minute version:**
- Part 1: 15 min (Sections 1-2, 6)
- Part 2: 10 min (Sections 7-8 highlights)
- Q&A: 5 min

**60-minute version:**
- Part 1: 30 min (Sections 1-6)
- Part 2: 20 min (Sections 7-9)
- Q&A: 10 min

**90-minute workshop:**
- Part 1: 40 min (Sections 1-6)
- Part 2: 30 min (Sections 7-9)
- Hands-on: 15 min (Section 10)
- Q&A: 5 min

---

## Key Messages to Emphasize

### Critical Findings:
1. **Multi-turn attacks are 2.8x more effective** (70%+ ASR vs <25%)
2. **Multi-turn defense benchmark is the biggest gap** (attack datasets exist, defense benchmarks don't)
3. **Best defenses achieve <5% ASR with >70% TCR** (security-utility balance)
4. **Tool access amplifies risk 5-10x** vs chatbots
5. **Position matters**: Attacks at end of context are hardest to defend (22% vs 12% ASR)

### Proposed Actions:
1. **PRIORITY:** Build Multi-Turn Defense Benchmark (5,000 cases)
2. Create CrossDomain Attack Dataset (500 chains)
3. Collect Real-World Attack Corpus (1,000+ attacks)
4. Develop Defense Robustness Benchmark (5,000 variations)

### Dataset Expansion Roadmap (5 weeks - EOY 2025):
- **Phase 1 (Week 1: Nov 27-Dec 6):** Multi-turn defense + CrossDomain attacks
- **Phase 2 (Week 2-3: Dec 7-13):** Real-world corpus + Defense robustness
- **Phase 3 (Week 3-4: Dec 14-20):** Adaptive multi-turn + Visual injection
- **Phase 4 (Week 5: Dec 21-31):** Long-context + Utility preservation
- **Final Integration:** Dec 28-31

---

## Visual Aids Included

The presentation includes:

✅ **Diagrams:**
- Attack flow (attacker → data source → model → agent)
- AgentDojo architecture pipeline
- Evaluation protocol flowchart
- Defense development workflow

✅ **Tables:**
- Attack surface comparison
- Metric definitions
- Current benchmark results
- Defense performance comparison
- Dataset inventory
- Coverage matrix
- Gap analysis

✅ **Code Blocks:**
- Metric calculation examples
- Dataset loading code
- Defense implementation template
- Benchmarking workflow

✅ **Charts (represented in ASCII/text):**
- Multi-turn vs single-turn effectiveness
- Security-utility trade-off
- Position-aware ASR
- Attack evolution over time

---

## Customization Guide

### To Add Your Company/Organization:
1. Replace "Security Research Presentation" in title slide
2. Add your logo/branding to header
3. Update "Contact & Collaboration" section with your details

### To Add Your Own Data:
1. Update "Current Benchmark Results" table (Slide 30) with your metrics
2. Add your defense to comparison tables
3. Include your dataset in inventory (Slide 51-52)

### To Focus on Specific Domain:
1. Filter attack datasets by domain (email/web/banking)
2. Highlight domain-specific vulnerabilities (Slide 35)
3. Customize proposed datasets for your domain

### To Shorten Presentation:
**Minimum viable version (20 slides):**
- Keep: Slides 1-3, 6-8, 13-15, 29-30, 36-43, 51-56, 91-93
- Remove: Detailed implementation, code examples, appendix

**Medium version (40 slides):**
- Keep: Sections 1-2, 6, 7-8, 11
- Remove: Sections 3-5, 9-10, appendix

---

## Additional Resources

### Related Files in This Repository:
- `agentdojo-guide.md` - Comprehensive AgentDojo documentation
- `attack-datasets.md` - Detailed attack dataset documentation
- `defense-datasets.md` - Defense benchmark documentation
- `benchmarking-methods.md` - Evaluation methodology
- `examples/load_attack_datasets.py` - Code examples
- `examples/test_defense.py` - Defense testing code
- `AgentDojo_Presentation.pptx` - Existing presentation (different focus)

### External Links to Include:
- AgentDojo: https://github.com/ethz-spylab/agentdojo
- HuggingFace: https://huggingface.co/datasets/ethz-spylab/agentdojo
- InjecAgent: https://github.com/uiuc-kang-lab/InjecAgent
- MHJ: https://huggingface.co/datasets/ScaleAI/mhj
- CyberSecEval: https://huggingface.co/datasets/walledai/CyberSecEval

---

## Q&A Preparation

### Expected Questions:

**Q1: Why focus on multi-turn when single-turn has more datasets?**
A: Multi-turn attacks are 2.8x more effective (70%+ ASR vs <25%). We have attack datasets but NO defense benchmarks - critical gap.

**Q2: What's the most urgent dataset to create?**
A: Multi-Turn Defense Benchmark (5,000 cases). Without it, we can't properly test defenses against the most effective attacks.

**Q3: How long will dataset creation take?**
A: 5 weeks total (Nov 27 - Dec 31, 2025). Phase 1 (multi-turn + cross-domain): Week 1. Full roadmap: 4 phases across 5 weeks. Can be parallelized with multiple teams for faster completion.

**Q4: What's the cost of creating these datasets?**
A: Varies by method:
- Manual red teaming: $50-100K (high quality, low scale)
- Crowdsourcing: $20-50K (medium quality, medium scale)
- Automated generation: $5-10K (medium quality, high scale)
- Production collection: Minimal cost (highest quality, ongoing)

**Q5: Can we use existing datasets?**
A: Yes! 11 attack datasets and 6 defense benchmarks already available. Gaps are in multi-turn defense, cross-domain attacks, and real-world corpus.

**Q6: What ASR should we target?**
A: <5% ASR for good defense, <1% for strong defense. But must maintain >70% TCR (utility). Target NRP > 65%.

**Q7: How do we measure success of new datasets?**
A:
- Attack datasets: Cause >20% ASR on undefended agents
- Defense benchmarks: Discriminate between good/bad defenses (FPR <5%, FNR <5%)
- Utility benchmarks: Identify over-defensive solutions (FPR on edge cases)

---

## Follow-Up Actions

After the presentation, consider:

1. **Set up working group** for multi-turn defense benchmark
2. **Initiate crowdsourcing campaign** for real-world attacks
3. **Partner with bug bounty platforms** for production data
4. **Create GitHub repository** for dataset collaboration
5. **Submit to HuggingFace** for distribution
6. **Write academic paper** documenting methodology
7. **Organize benchmark competition** to incentivize participation

---

## License

This presentation and datasets should be:
- **Presentation:** CC-BY 4.0 (attribution required)
- **Attack Datasets:** MIT or Apache 2.0 (maximize research usage)
- **Defense Benchmarks:** MIT or Apache 2.0 (standardize evaluation)
- **Real-World Corpus:** Restricted access (contains sensitive info)

---

## Feedback & Iteration

To improve this presentation:
1. Test with pilot audience (collect feedback)
2. Add domain-specific examples for your use case
3. Include live demos if possible (from examples/ folder)
4. Update with latest research (presentation dated 2025-11-27)
5. Add your organization's results/contributions

Good luck with your presentation!
