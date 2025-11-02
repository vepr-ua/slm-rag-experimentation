"""
Collector for Cross Validated (StackExchange for Statistics) Q&A pairs.
"""

import time
from typing import Optional

from loguru import logger
from stackapi import StackAPI
from tqdm import tqdm

from ..config import CollectorConfig, DataCollectionConfig
from ..models import DataSource, QAPair


class CrossValidatedCollector:
    """Collect Q&A pairs from Cross Validated (stats.stackexchange.com)."""

    def __init__(
        self,
        config: Optional[DataCollectionConfig] = None,
        collector_config: Optional[CollectorConfig] = None,
    ):
        self.config = config or DataCollectionConfig()
        self.collector_config = collector_config or CollectorConfig()

        # Initialize StackAPI client
        self.client = StackAPI(self.collector_config.stackexchange_site)
        if self.collector_config.stackexchange_api_key:
            self.client.key = self.collector_config.stackexchange_api_key

        self.client.max_pages = 100  # Safety limit
        self.client.page_size = 100  # Max allowed by API

    def collect_by_tags(self, max_questions: Optional[int] = None) -> list[QAPair]:
        """
        Collect questions with specific tags from Cross Validated.

        Args:
            max_questions: Maximum number of questions to collect (None for config default)

        Returns:
            List of QAPair objects
        """
        max_questions = max_questions or self.config.stackexchange_max_questions
        qa_pairs = []

        # Warn about API limits
        if not self.collector_config.stackexchange_api_key:
            logger.warning(
                "⚠️  No StackExchange API key configured. "
                "Rate limit: 300 requests/day. "
                "Add STACKEXCHANGE_API_KEY to .env for 10,000 requests/day."
            )
            if max_questions > 100:
                logger.warning(
                    f"⚠️  Collecting {max_questions} questions without API key may hit rate limits. "
                    "Consider using --max-questions 100 or adding an API key."
                )

        logger.info(
            f"Collecting up to {max_questions} questions with tags: {self.config.stackexchange_tags}"
        )

        # Fetch questions with relevant tags
        questions = self._fetch_questions(max_questions)
        logger.info(f"Fetched {len(questions)} questions")

        # Process each question and its answers
        for question in tqdm(questions, desc="Processing questions"):
            try:
                pairs = self._process_question(question)
                qa_pairs.extend(pairs)
                time.sleep(1 / (self.config.requests_per_minute / 60))  # Rate limit
            except Exception as e:
                logger.warning(f"Error processing question {question.get('question_id')}: {e}")
                continue

        logger.info(f"Collected {len(qa_pairs)} Q&A pairs from Cross Validated")
        return qa_pairs

    def _fetch_questions(self, max_questions: int) -> list[dict]:
        """
        Fetch questions from StackExchange API.

        Queries each tag individually to get diverse results (OR logic),
        then deduplicates and sorts by score.
        """
        all_questions = {}  # Use dict to deduplicate by question_id

        logger.info(f"Fetching questions for {len(self.config.stackexchange_tags)} tags...")

        for tag in self.config.stackexchange_tags:
            try:
                logger.debug(f"Fetching questions tagged '{tag}'")

                # Fetch questions for this tag
                response = self.client.fetch(
                    "questions",
                    tagged=tag,
                    sort="votes",
                    order="desc",
                    filter="withbody",  # Include question body
                    min=self.config.stackexchange_min_score,
                )

                if "items" in response:
                    for question in response["items"]:
                        # Deduplicate by question_id
                        qid = question["question_id"]
                        if qid not in all_questions:
                            all_questions[qid] = question

                # Small delay between tag queries to be nice to API
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error fetching questions for tag '{tag}': {e}")
                continue

        # Convert to list and sort by score
        questions = list(all_questions.values())
        questions.sort(key=lambda q: q.get("score", 0), reverse=True)

        # Limit to max_questions
        questions = questions[:max_questions]

        logger.info(f"Fetched {len(questions)} unique questions across all tags")

        return questions

    def _process_question(self, question: dict) -> list[QAPair]:
        """
        Process a single question and its answers into QAPair objects.

        Args:
            question: Question data from StackExchange API

        Returns:
            List of QAPair objects (one per answer)
        """
        qa_pairs = []
        question_id = question["question_id"]

        # Basic quality filters
        if len(question.get("title", "")) < self.config.min_question_length:
            return qa_pairs

        # Fetch answers for this question
        try:
            answers_response = self.client.fetch(
                f"questions/{question_id}/answers",
                sort="votes",
                order="desc",
                filter="withbody",
            )

            answers = answers_response.get("items", [])

            # Filter answers
            if self.config.stackexchange_accepted_only:
                answers = [a for a in answers if a.get("is_accepted", False)]

            if len(answers) < self.config.stackexchange_min_answers:
                return qa_pairs

            # Process top answer(s)
            for answer in answers[:3]:  # Take up to 3 best answers
                qa_pair = self._create_qa_pair(question, answer)
                if qa_pair:
                    qa_pairs.append(qa_pair)

        except Exception as e:
            logger.warning(f"Error fetching answers for question {question_id}: {e}")

        return qa_pairs

    def _create_qa_pair(self, question: dict, answer: dict) -> Optional[QAPair]:
        """
        Create a QAPair from question and answer data.

        Args:
            question: Question data from API
            answer: Answer data from API

        Returns:
            QAPair object or None if invalid
        """
        # Extract text
        question_title = question.get("title", "")
        question_body = question.get("body", "")
        answer_body = answer.get("body", "")

        # Quality checks
        if len(answer_body) < self.config.min_answer_length:
            return None
        if len(answer_body) > self.config.max_answer_length:
            return None

        # Combine title and body for question
        question_text = f"{question_title}\n\n{question_body}".strip()

        # Clean HTML tags (StackExchange uses HTML)
        question_text = self._clean_html(question_text)
        answer_text = self._clean_html(answer_body)

        # Create QAPair
        qa_pair = QAPair(
            id=f"cv_{question['question_id']}_{answer['answer_id']}",
            question=question_text,
            answer=answer_text,
            source=DataSource.CROSS_VALIDATED,
            source_url=question.get("link"),
            source_id=str(question["question_id"]),
            score=float(answer.get("score", 0)),
            upvotes=answer.get("up_vote_count", 0),
            view_count=question.get("view_count", 0),
            topics=[],  # Will be classified later
        )

        return qa_pair

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text().strip()
