"""
Format Q&A pairs into ChatML format for Llama 3.2 training.
"""

from typing import Optional

from ..models import QAPair


class ChatMLFormatter:
    """
    Format Q&A pairs into ChatML format for Llama 3.2.

    ChatML format:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    {answer}<|eot_id|>
    """

    # Special tokens for Llama 3.2
    BEGIN_OF_TEXT = "<|begin_of_text|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"
    EOT = "<|eot_id|>"
    END_OF_TEXT = "<|end_of_text|>"

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert in experimentation, statistics, and A/B testing. "
        "Provide clear, accurate, and helpful explanations. "
        "When appropriate, explain your reasoning step-by-step and cite sources."
    )

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize formatter.

        Args:
            system_prompt: Custom system prompt (uses default if None)
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def format_qa_pair(
        self,
        qa_pair: QAPair,
        include_reasoning: bool = True,
        include_citations: bool = True,
    ) -> str:
        """
        Format a single Q&A pair into ChatML format.

        Args:
            qa_pair: QAPair to format
            include_reasoning: Include reasoning in answer if available
            include_citations: Include citations in answer if available

        Returns:
            Formatted ChatML string
        """
        # Build answer text
        answer_text = qa_pair.answer

        # Add reasoning if requested and available
        if include_reasoning and qa_pair.reasoning:
            answer_text = f"{qa_pair.reasoning}\n\n{answer_text}"

        # Add citations if requested and available
        if include_citations and qa_pair.citations:
            citations_text = "\n\nSources:\n" + "\n".join(
                f"- {citation}" for citation in qa_pair.citations
            )
            answer_text = f"{answer_text}{citations_text}"

        # Build ChatML format
        messages = [
            self._format_message("system", self.system_prompt),
            self._format_message("user", qa_pair.question),
            self._format_message("assistant", answer_text),
        ]

        return self.BEGIN_OF_TEXT + "".join(messages)

    def format_multi_turn(
        self, qa_pairs: list[QAPair], conversation_id: str
    ) -> Optional[str]:
        """
        Format multiple Q&A pairs as a multi-turn conversation.

        Args:
            qa_pairs: List of QAPair objects that form a conversation
            conversation_id: ID to identify the conversation

        Returns:
            Formatted ChatML string or None if invalid
        """
        if not qa_pairs:
            return None

        # Sort by parent_id to ensure correct order
        # (assuming parent_id indicates the conversation flow)
        sorted_pairs = sorted(qa_pairs, key=lambda x: x.collected_at)

        # Build multi-turn conversation
        messages = [self._format_message("system", self.system_prompt)]

        for qa_pair in sorted_pairs:
            messages.append(self._format_message("user", qa_pair.question))
            messages.append(self._format_message("assistant", qa_pair.answer))

        return self.BEGIN_OF_TEXT + "".join(messages)

    def _format_message(self, role: str, content: str) -> str:
        """
        Format a single message in ChatML format.

        Args:
            role: Role (system, user, assistant)
            content: Message content

        Returns:
            Formatted message string
        """
        return (
            f"{self.START_HEADER}{role}{self.END_HEADER}\n"
            f"{content.strip()}{self.EOT}"
        )

    def format_batch(
        self,
        qa_pairs: list[QAPair],
        include_reasoning: bool = True,
        include_citations: bool = True,
    ) -> list[str]:
        """
        Format multiple Q&A pairs.

        Args:
            qa_pairs: List of QAPair objects
            include_reasoning: Include reasoning if available
            include_citations: Include citations if available

        Returns:
            List of formatted ChatML strings
        """
        formatted = []

        for qa_pair in qa_pairs:
            try:
                formatted_text = self.format_qa_pair(
                    qa_pair,
                    include_reasoning=include_reasoning,
                    include_citations=include_citations,
                )
                formatted.append(formatted_text)
            except Exception as e:
                # Log error but continue
                print(f"Error formatting {qa_pair.id}: {e}")
                continue

        return formatted

    def to_jsonl(self, qa_pairs: list[QAPair], output_file: str) -> None:
        """
        Save formatted Q&A pairs to JSONL file for training.

        Each line is a JSON object with:
        {
            "text": "<formatted ChatML string>",
            "id": "qa_pair_id",
            "source": "data_source"
        }

        Args:
            qa_pairs: List of QAPair objects
            output_file: Path to output JSONL file
        """
        import json
        from pathlib import Path

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for qa_pair in qa_pairs:
                formatted_text = self.format_qa_pair(qa_pair)

                entry = {
                    "text": formatted_text,
                    "id": qa_pair.id,
                    "source": qa_pair.source,
                    "topics": qa_pair.topics,
                }

                f.write(json.dumps(entry) + "\n")

        print(f"Saved {len(qa_pairs)} formatted examples to {output_file}")
