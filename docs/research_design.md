# Research Design: Multi-Agent LLM Simulation of Medical Student Clinical Reasoning

**Target Journal**: Computers in Human Behavior (SSCI Q1, IF ~9.0)
**Positioning**: Methodological study — evaluating LLM simulation fidelity
**Precedent**: Vogelsmeier et al. (2025, CHB) — LLM-generated survey response validation

---

## Title (Working)

**"Can Multi-Agent LLMs Faithfully Simulate Medical Students' Clinical Reasoning?
A Methodological Evaluation of Knowledge-Constrained Generation"**

---

## Research Questions

**RQ1**: To what extent can knowledge-constrained LLM agents simulate medical students
at different competence levels (Year 1–5) during clinical reasoning dialogues?

**RQ2**: Does a multi-agent architecture (separate Tutor, Patient, Assessment agents)
produce more educationally realistic interactions compared to a single-agent LLM?

**RQ3**: What are the systematic failure modes of LLM-simulated students, and how do
they differ from documented patterns of real student errors?

---

## Theoretical Framework

### 1. Community of Inquiry (CoI) — Garrison et al. (2000)
- **Teaching Presence**: How the Tutor Agent scaffolds learning
- **Cognitive Presence**: Depth of reasoning in simulated student responses
- **Social Presence**: (limited in simulation — acknowledged as limitation)

### 2. Bloom's Revised Taxonomy — Anderson & Krathwohl (2001)
- Coding simulated student responses by cognitive level:
  Remember → Understand → Apply → Analyze → Evaluate → Create
- Hypothesis: Lower-level simulated students should cluster at Remember/Understand;
  higher-level at Apply/Analyze

### 3. Competence Paradox — Scarlatos et al. (2026)
- Core tension: Powerful LLMs struggle to convincingly simulate partial knowledge
- Our contribution: Systematic evaluation of knowledge-constrained generation methods
  for resolving this paradox

### 4. Conceptual Change Theory — Posner et al. (1982)
- Medical misconceptions have documented typologies (e.g., "chest pain = heart attack")
- We inject known misconceptions and measure whether the tutoring system corrects them

---

## Method

### Study Design

3 (interaction condition) × 4 (student competence level) × 30 (clinical cases)
= 360 unique dialogue configurations, each repeated with 5 random seeds
= **1,800 total dialogues**

### Independent Variables

**IV1: Interaction Condition**
| Condition | Architecture | Description |
|-----------|-------------|-------------|
| Multi-Agent | Tutor + Patient + Assessment + SimStudent | Full architecture |
| Single-Agent | One LLM plays all roles | Baseline comparison |
| Direct QA | Student asks, LLM answers directly | No pedagogical scaffolding |

**IV2: Simulated Student Competence Level**
| Level | Medical Knowledge Scope | Reasoning Capability |
|-------|------------------------|---------------------|
| Novice (Y1) | Anatomy + Physiology only | Pattern matching only |
| Intermediate (Y3) | + Pathology + Pharmacology basics | 1-hop reasoning |
| Advanced (Y4) | + Clinical medicine | Multi-hop, some gaps |
| Expert (Y5) | Full curriculum | Systematic reasoning |

Each level implemented via:
1. **Knowledge Boundary Specification**: Explicit list of known/unknown concepts
2. **Confusion Tuples**: Specified concept pairs the student confuses
3. **Reasoning Depth Limit**: Maximum inference chain length

### Clinical Cases (30 total)
- Source: MedQA-USMLE case vignettes (already on MUSICA)
- Distribution: 10 internal medicine, 5 surgery, 5 emergency, 5 pediatrics, 5 OB/GYN
- Complexity: 10 single-hop, 10 two-hop, 10 three-hop reasoning required

### Dependent Variables (Fidelity Metrics)

**Linguistic Fidelity**
- Vocabulary complexity (type-token ratio) by student level
- Medical terminology density
- Question formulation sophistication

**Behavioral Fidelity**
- Response latency patterns (simulated via token count)
- Help-seeking frequency
- Error self-correction rate
- Dialogue turn count to reach diagnosis

**Cognitive Fidelity** (primary outcome)
- Bloom's taxonomy level distribution per student level
- CoI Cognitive Presence coding (Triggering → Exploration → Integration → Resolution)
- Misconception persistence rate (injected misconceptions that survive tutoring)
- Diagnostic accuracy trajectory across dialogue turns

**Educational Interaction Quality**
- Scaffolding appropriateness (does Tutor adjust to student level?)
- Socratic questioning ratio (questions vs direct answers)
- Multi-agent coordination quality (no contradictions, no loops)

### Analysis Plan

1. **Descriptive**: Distribution of fidelity metrics across conditions and levels
2. **ANOVA**: Condition × Level interaction effects on cognitive fidelity
3. **Qualitative coding**: Manual annotation of 100 randomly sampled dialogues
   (2 independent coders, Cohen's kappa for reliability)
4. **Failure mode taxonomy**: Systematic categorization of simulation breakdowns
5. **Comparison with literature**: Align simulated patterns against published
   medical education interaction studies (e.g., Barrows 1986, Hmelo-Silver 2004)

---

## Contribution to CHB

1. **First systematic evaluation** of multi-agent LLM simulation fidelity for
   medical education interactions (no prior CHB/SSCI paper addresses this)
2. **Knowledge-constrained generation framework** for resolving the Competence Paradox
3. **Failure mode taxonomy** providing actionable guidelines for future simulation studies
4. **Methodological foundation** for subsequent human-participant studies
   (simulation as pre-study design tool)

---

## Limitations (to acknowledge upfront)

1. No real human validation (positioned as future work)
2. Social Presence dimension of CoI cannot be authentically simulated
3. LLM behavior may not generalize across model families
4. English-language medical cases only

---

## Timeline

| Week | Task |
|------|------|
| 1 | System development (Multi-Agent architecture) |
| 2 | Knowledge constraint mechanism + case preparation |
| 3 | Generate 1,800 dialogues on MUSICA |
| 4 | Coding + quantitative analysis |
| 5-6 | Paper writing |
| 7 | Triple review (AI detection + factual + integrity) |
| 8 | Submit to CHB |

---

## References (Key)

- Garrison, D. R., Anderson, T., & Archer, W. (2000). Critical inquiry in a text-based
  environment: Computer conferencing in higher education. The Internet and Higher Education.
- Scarlatos, A., et al. (2026). Simulated Students in Tutoring Dialogues: Substance or Illusion?
  arXiv:2601.04025.
- Vogelsmeier, L. G., et al. (2025). Delving into the psychology of machines. CHB, 108769.
- Posner, G. J., et al. (1982). Accommodation of a scientific conception. Science Education.
- Anderson, L. W., & Krathwohl, D. R. (2001). A Taxonomy for Learning, Teaching, and Assessing.
