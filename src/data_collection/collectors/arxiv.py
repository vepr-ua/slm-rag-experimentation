"""
Collector for ArXiv papers on experimentation and statistics.

Note: ArXiv papers are collected as source material for synthetic Q&A generation,
not as direct Q&A pairs.
"""

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import arxiv
from loguru import logger
from tqdm import tqdm

from ..config import CollectorConfig, DataCollectionConfig


class ArXivPaper:
    """Represents an ArXiv paper for processing."""

    def __init__(self, arxiv_result: arxiv.Result):
        self.arxiv_id = arxiv_result.entry_id.split("/")[-1]
        self.title = arxiv_result.title
        self.authors = [author.name for author in arxiv_result.authors]
        self.abstract = arxiv_result.summary
        self.published = arxiv_result.published
        self.updated = arxiv_result.updated
        self.categories = arxiv_result.categories
        self.pdf_url = arxiv_result.pdf_url
        self.primary_category = arxiv_result.primary_category

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "categories": self.categories,
            "primary_category": self.primary_category,
            "pdf_url": self.pdf_url,
        }


class ArXivCollector:
    """Collect papers from ArXiv about experimentation and statistics."""

    def __init__(
        self,
        config: Optional[DataCollectionConfig] = None,
        collector_config: Optional[CollectorConfig] = None,
    ):
        self.config = config or DataCollectionConfig()
        self.collector_config = collector_config or CollectorConfig()

        # Initialize ArXiv client
        self.client = arxiv.Client()

    def collect_papers(
        self, max_results_per_query: Optional[int] = None
    ) -> list[ArXivPaper]:
        """
        Collect papers from ArXiv based on configured search queries.

        Args:
            max_results_per_query: Max results per search query

        Returns:
            List of ArXivPaper objects
        """
        max_results = max_results_per_query or self.config.arxiv_max_results_per_query
        all_papers = []
        seen_ids = set()

        # Warn about API etiquette
        total_requests = len(self.config.arxiv_search_queries) * max_results
        estimated_time = (total_requests * self.collector_config.arxiv_delay_seconds) / 60
        logger.info(
            f"ℹ️  ArXiv API etiquette: {self.collector_config.arxiv_delay_seconds}s delay between requests. "
            f"Estimated time: ~{estimated_time:.1f} minutes for {total_requests} requests."
        )

        logger.info(
            f"Collecting ArXiv papers for {len(self.config.arxiv_search_queries)} queries "
            f"(max {max_results} per query)"
        )

        for query in tqdm(self.config.arxiv_search_queries, desc="ArXiv queries"):
            try:
                papers = self._search_arxiv(query, max_results)

                # Deduplicate
                for paper in papers:
                    if paper.arxiv_id not in seen_ids:
                        all_papers.append(paper)
                        seen_ids.add(paper.arxiv_id)

                # Rate limiting
                time.sleep(self.collector_config.arxiv_delay_seconds)

            except Exception as e:
                logger.warning(f"Error searching ArXiv for '{query}': {e}")
                continue

        logger.info(f"Collected {len(all_papers)} unique papers from ArXiv")
        return all_papers

    def _search_arxiv(self, query: str, max_results: int) -> list[ArXivPaper]:
        """
        Search ArXiv for papers matching query.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            List of ArXivPaper objects
        """
        papers = []

        try:
            # Build search with category filters if specified
            search_query = query
            if self.config.arxiv_categories:
                category_filter = " OR ".join(
                    [f"cat:{cat}" for cat in self.config.arxiv_categories]
                )
                search_query = f"({query}) AND ({category_filter})"

            # Create search
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            # Execute search
            for result in self.client.results(search):
                papers.append(ArXivPaper(result))

        except Exception as e:
            logger.error(f"Error executing ArXiv search for '{query}': {e}")

        return papers

    def download_pdfs(
        self, papers: list[ArXivPaper], output_dir: Optional[Path] = None
    ) -> dict[str, Path]:
        """
        Download PDFs for collected papers.

        Args:
            papers: List of ArXivPaper objects
            output_dir: Directory to save PDFs (default: data/raw/arxiv)

        Returns:
            Dictionary mapping arxiv_id to downloaded file path
        """
        output_dir = output_dir or Path(self.config.raw_data_dir) / "arxiv" / "pdfs"
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = {}

        for paper in tqdm(papers, desc="Downloading PDFs"):
            try:
                # Create filename
                filename = f"{paper.arxiv_id.replace('/', '_')}.pdf"
                filepath = output_dir / filename

                # Skip if already downloaded
                if filepath.exists():
                    logger.debug(f"Skipping {paper.arxiv_id} - already exists")
                    downloaded[paper.arxiv_id] = filepath
                    continue

                # Download - arxiv library expects directory, not full path
                paper_obj = next(
                    arxiv.Search(id_list=[paper.arxiv_id]).results()
                )

                # Download to directory (arxiv creates its own filename)
                downloaded_path = paper_obj.download_pdf(dirpath=str(output_dir))

                # Rename to our simplified format if needed
                if downloaded_path != str(filepath):
                    shutil.move(downloaded_path, filepath)

                downloaded[paper.arxiv_id] = filepath
                logger.info(f"Downloaded {paper.arxiv_id} to {filepath}")

                # Rate limiting
                time.sleep(self.collector_config.arxiv_delay_seconds)

            except Exception as e:
                logger.warning(f"Error downloading PDF for {paper.arxiv_id}: {e}")
                continue

        logger.info(f"Downloaded {len(downloaded)}/{len(papers)} PDFs")
        return downloaded

    def save_metadata(
        self, papers: list[ArXivPaper], output_file: Optional[Path] = None
    ) -> Path:
        """
        Save paper metadata to JSON file.

        Args:
            papers: List of ArXivPaper objects
            output_file: Output file path (default: data/raw/arxiv/metadata.json)

        Returns:
            Path to saved file
        """
        import json

        output_file = output_file or (
            Path(self.config.raw_data_dir) / "arxiv" / "metadata.json"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        papers_data = [paper.to_dict() for paper in papers]

        # Save
        with open(output_file, "w") as f:
            json.dump(
                {
                    "collected_at": datetime.utcnow().isoformat(),
                    "total_papers": len(papers),
                    "papers": papers_data,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved metadata for {len(papers)} papers to {output_file}")
        return output_file
