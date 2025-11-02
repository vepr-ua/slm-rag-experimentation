"""
Synthetic Q&A generation from source materials.

Generates training Q&A pairs from:
- ArXiv papers (abstracts and conclusions)
- Textbooks (future)
- Blog posts (future)
"""

import json
import re
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

from ...llm.claude_client import ClaudeClient
from ...llm.prompts import (
    QA_GENERATION_SYSTEM,
    format_arxiv_abstract_prompt,
    format_arxiv_conclusion_prompt,
)
from ..models import DataSource, QAPair


class SyntheticQAGenerator:
    """Generate synthetic Q&A pairs from source materials using Claude."""

    def __init__(self, claude_client: Optional[ClaudeClient] = None):
        """
        Initialize generator.

        Args:
            claude_client: ClaudeClient instance (creates new if None)
        """
        self.claude = claude_client or ClaudeClient()

    def generate_from_arxiv_paper(
        self,
        arxiv_id: str,
        title: str,
        abstract: str,
        num_questions: int = 5,
    ) -> list[QAPair]:
        """
        Generate Q&A pairs from an ArXiv paper.

        Args:
            arxiv_id: ArXiv paper ID
            title: Paper title
            abstract: Paper abstract
            num_questions: Number of Q&A pairs to generate

        Returns:
            List of QAPair objects
        """
        logger.info(f"Generating {num_questions} Q&A pairs from paper {arxiv_id}")

        # Generate Q&A using Claude
        prompt = format_arxiv_abstract_prompt(title, abstract, num_questions)

        try:
            response = self.claude.generate(
                prompt=prompt,
                system_prompt=QA_GENERATION_SYSTEM,
                temperature=0.8,  # Some creativity
            )

            # Parse response into Q&A pairs
            qa_pairs = self._parse_qa_response(response, arxiv_id, title)

            logger.info(f"Generated {len(qa_pairs)} Q&A pairs from {arxiv_id}")
            return qa_pairs

        except Exception as e:
            logger.error(f"Failed to generate Q&A for {arxiv_id}: {e}")
            return []

    def generate_from_arxiv_metadata(
        self, metadata_file: Path, max_papers: Optional[int] = None
    ) -> list[QAPair]:
        """
        Generate Q&A pairs from ArXiv metadata file.

        Args:
            metadata_file: Path to arxiv_metadata.json
            max_papers: Maximum number of papers to process (None for all)

        Returns:
            List of all generated QAPair objects
        """
        logger.info(f"Loading ArXiv metadata from {metadata_file}")

        # Load metadata
        with open(metadata_file, "r") as f:
            data = json.load(f)

        papers = data.get("papers", [])

        if max_papers:
            papers = papers[:max_papers]

        logger.info(f"Processing {len(papers)} papers")

        # Estimate cost
        self.claude.estimate_cost(
            num_requests=len(papers),
            avg_tokens_per_request=1500,  # Smaller than default for abstracts
        )

        all_qa_pairs = []

        for paper in tqdm(papers, desc="Generating Q&A from papers"):
            try:
                qa_pairs = self.generate_from_arxiv_paper(
                    arxiv_id=paper["arxiv_id"],
                    title=paper["title"],
                    abstract=paper["abstract"],
                    num_questions=5,  # Generate 5 Q&A per paper
                )
                all_qa_pairs.extend(qa_pairs)

            except Exception as e:
                logger.warning(f"Skipping paper {paper.get('arxiv_id')}: {e}")
                continue

        logger.info(f"Generated {len(all_qa_pairs)} total Q&A pairs")
        return all_qa_pairs

    def _parse_qa_response(
        self, response: str, arxiv_id: str, title: str
    ) -> list[QAPair]:
        """
        Parse Claude's response into QAPair objects.

        Expected format:
        ---
        Q: Question text
        A: Answer text
        ---

        Args:
            response: Generated text from Claude
            arxiv_id: ArXiv paper ID
            title: Paper title

        Returns:
            List of QAPair objects
        """
        qa_pairs = []

        # Split by delimiter
        sections = re.split(r"---+", response)

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # Extract question and answer
            q_match = re.search(r"Q:\s*(.+?)(?=A:)", section, re.DOTALL)
            a_match = re.search(r"A:\s*(.+)", section, re.DOTALL)

            if q_match and a_match:
                question = q_match.group(1).strip()
                answer = a_match.group(1).strip()

                # Create QAPair
                qa_pair = QAPair(
                    id=f"synthetic_arxiv_{arxiv_id}_{i}",
                    question=question,
                    answer=answer,
                    source=DataSource.SYNTHETIC,
                    source_url=f"https://arxiv.org/abs/{arxiv_id}",
                    source_id=arxiv_id,
                    citations=[f"ArXiv: {arxiv_id} - {title}"],
                )

                qa_pairs.append(qa_pair)

        return qa_pairs

    def save_generated_qa(
        self, qa_pairs: list[QAPair], output_file: Path
    ) -> None:
        """
        Save generated Q&A pairs to JSON file.

        Args:
            qa_pairs: List of QAPair objects
            output_file: Output file path
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(
                [qa.model_dump() for qa in qa_pairs],
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {output_file}")
