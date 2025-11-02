"""
Prompt templates for LLM operations.

Includes prompts for:
- Synthetic Q&A generation
- Evaluation
- Classification
"""

# System prompt for Q&A generation
QA_GENERATION_SYSTEM = """You are an expert in experimental design, A/B testing, and statistical analysis. Your task is to generate high-quality question-answer pairs for training a language model on these topics.

Guidelines:
- Generate questions that practitioners would actually ask
- Provide clear, accurate, and helpful answers
- Include step-by-step reasoning when appropriate
- Keep answers concise but comprehensive (100-300 words)
- Focus on practical understanding, not just theory
- Avoid code examples (focus on concepts and methodology)
"""

# Prompt for generating Q&A from ArXiv paper abstracts
ARXIV_ABSTRACT_TO_QA = """Based on the following research paper abstract about experimentation and statistics, generate {num_questions} diverse question-answer pairs.

**Paper Title:** {title}

**Abstract:**
{abstract}

Generate questions that cover:
1. What problem does this research address?
2. What methodology or approach is proposed?
3. What are the key findings or contributions?
4. When would a practitioner use this technique?
5. How does this relate to A/B testing or experimentation?

For each Q&A pair, output in this exact format:
---
Q: [Clear, specific question]
A: [Comprehensive answer with reasoning]
---

Generate {num_questions} Q&A pairs now:
"""

# Prompt for generating Q&A from paper conclusions
ARXIV_CONCLUSION_TO_QA = """Based on the following research paper conclusion/summary, generate {num_questions} practical question-answer pairs.

**Paper Title:** {title}

**Conclusion:**
{conclusion}

Generate questions about:
- Practical implications
- When to apply these findings
- Comparisons to other methods
- Limitations and considerations
- Real-world applications

For each Q&A pair, output in this exact format:
---
Q: [Clear, specific question]
A: [Comprehensive answer with reasoning]
---

Generate {num_questions} Q&A pairs now:
"""

# Prompt for different question types
QUESTION_TYPE_PROMPTS = {
    "definition": """Generate a definition-style question and answer about: {concept}

The question should ask "What is X?" or "Define X" and the answer should:
- Provide a clear definition
- Explain why it matters in experimentation
- Give a brief example or context

Format:
---
Q: [Question]
A: [Answer]
---
""",
    "methodology": """Generate a methodology question and answer about: {concept}

The question should ask "How do I X?" or "What's the process for X?" and the answer should:
- Provide step-by-step guidance
- Explain the reasoning
- Mention common pitfalls or considerations

Format:
---
Q: [Question]
A: [Answer]
---
""",
    "comparison": """Generate a comparison question and answer about: {concept1} and {concept2}

The question should ask about differences, trade-offs, or when to use each. The answer should:
- Compare the two approaches clearly
- Explain trade-offs
- Provide guidance on when to use each

Format:
---
Q: [Question]
A: [Answer]
---
""",
    "troubleshooting": """Generate a troubleshooting question and answer about: {problem}

The question should describe a common problem and the answer should:
- Identify the root cause
- Provide clear solutions
- Explain how to prevent it in the future

Format:
---
Q: [Question]
A: [Answer]
---
""",
}

# Prompt for topic classification
CLASSIFY_TOPIC = """Classify the following question into one or more experimentation topics.

**Question:** {question}

**Available Topics:**
- hypothesis_testing
- p_values
- confidence_intervals
- statistical_power
- effect_size
- significance
- sample_size
- randomization
- stratification
- test_duration
- metric_selection
- variance_reduction
- multiple_testing
- sequential_testing
- bayesian_methods
- causal_inference
- heterogeneous_treatment_effects
- network_effects
- general
- other

Return ONLY the topic names as a comma-separated list, e.g.: "hypothesis_testing, p_values"
"""

# Prompt for answer quality evaluation
EVALUATE_ANSWER_QUALITY = """Evaluate the quality of this answer for training a language model on experimentation and statistics.

**Question:** {question}
**Answer:** {answer}

Rate the answer on these criteria (1-5 scale):
1. **Accuracy**: Is the information correct?
2. **Completeness**: Does it fully answer the question?
3. **Clarity**: Is it easy to understand?
4. **Practicality**: Is it useful for practitioners?
5. **Conciseness**: Is it appropriately detailed without being verbose?

Provide scores and brief justification:
Accuracy: [1-5] - [why]
Completeness: [1-5] - [why]
Clarity: [1-5] - [why]
Practicality: [1-5] - [why]
Conciseness: [1-5] - [why]
Overall: [Accept/Reject] - [brief summary]
"""


def format_arxiv_abstract_prompt(
    title: str, abstract: str, num_questions: int = 5
) -> str:
    """Format prompt for generating Q&A from ArXiv abstract."""
    return ARXIV_ABSTRACT_TO_QA.format(
        title=title, abstract=abstract, num_questions=num_questions
    )


def format_arxiv_conclusion_prompt(
    title: str, conclusion: str, num_questions: int = 3
) -> str:
    """Format prompt for generating Q&A from paper conclusion."""
    return ARXIV_CONCLUSION_TO_QA.format(
        title=title, conclusion=conclusion, num_questions=num_questions
    )


def format_question_type_prompt(
    question_type: str, **kwargs
) -> str:
    """
    Format prompt for specific question type.

    Args:
        question_type: One of 'definition', 'methodology', 'comparison', 'troubleshooting'
        **kwargs: Variables for the prompt (concept, concept1, concept2, problem, etc.)

    Returns:
        Formatted prompt string
    """
    if question_type not in QUESTION_TYPE_PROMPTS:
        raise ValueError(
            f"Unknown question type: {question_type}. "
            f"Choose from: {list(QUESTION_TYPE_PROMPTS.keys())}"
        )

    return QUESTION_TYPE_PROMPTS[question_type].format(**kwargs)
