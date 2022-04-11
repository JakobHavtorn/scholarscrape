from __future__ import annotations

import argparse
import logging
import os
import re
import time

from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue

from typing import List

import rich
import pandas as pd
import numpy as np

from tqdm import tqdm

from scholarscrape import semanticscholar_api as ssapi


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)7s %(thread)d | %(message)s")
LOGGER = logging.getLogger(__name__)

DATA_DIR = "./data"


def camel2snake(txt: str) -> str:
    """Convert a CamelCasedString to a snake_cased_string"""
    txt = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", txt)
    txt = re.sub("__([A-Z])", r"_\1", txt)
    txt = re.sub("([a-z0-9])([A-Z])", r"\1_\2", txt)
    return txt.lower()


class SemanticCrawler:
    """Crawl across SemanticScholar to obtain all papers and authors while writing to database."""

    def __init__(
        self,
        search_seeds: List[str],
        priority_key: str = "citation_count",
        timeout: int = ssapi.REQUEST_TIMEOUT,
        api_key: str = None,
        api_url: str = None,
    ):
        self.search_seeds = search_seeds
        self.priority_key = priority_key
        self.api = ssapi.SemanticScholarAPI(timeout, api_key, api_url)
        self.in_queue = PriorityQueue()  # prioritize authors by citation count to get the most bang for the buck
        raise NotImplementedError()

    def submit_author(self, author_id):
        raise NotImplementedError()

    def submit_paper(self, paper_id: str):
        raise NotImplementedError()

    def work(self):
        raise NotImplementedError()


def papers2df(papers: List[ssapi.Paper]):
    papers_df = pd.DataFrame(papers)
    papers_df.dropna(axis=1, how="all", inplace=True)  # Drop all columns that have only NA values
    papers_df.columns = [camel2snake(str(x)) for x in papers_df.columns]  # Convert camelCase columns to snake_case
    return papers_df


def author2df(author: ssapi.Author):
    author_dict = author.to_dict()
    author_dict.pop("papers")
    author_df = pd.DataFrame([author_dict])
    author_df.columns = [camel2snake(str(x)) for x in author_df.columns]  # Convert camelCase columns to snake_case
    return author_df


def get_author_details(author_id: str, ss: ssapi.SemanticScholarAPI):
    fields = [f"papers.{f}" for f in ssapi.FIELDS_PAPER_BASIC] + ssapi.FIELDS_AUTHOR_BASIC
    author = ss.author_details(author_id, fields=fields)
    papers_df = papers2df(author.papers)
    author_df = author2df(author)
    return author, author_df, papers_df


def compute_self_citation_count(author_id: str, ss: ssapi.SemanticScholarAPI = None) -> int:
    if ss is None:
        ss = ssapi.SemanticScholarAPI()  # default

    author, author_df, papers_df_trunc = get_author_details(author_id, ss)

    papers = ss.author_papers(author_id, fields=ssapi.FIELDS_PAPER_BASIC)
    papers_df = papers2df(papers)

    citations_per_paper = ss.paper_citations_threaded(papers_df.paper_id, ssapi.FIELDS_PAPER_BASIC)
    citations_per_paper = {k: v for k, v in zip(papers_df.paper_id, citations_per_paper)}

    co_author_ids = set(a.author_id for paper in author.papers for a in paper.authors)

    is_pseudo_author = lambda paper: bool(co_author_ids.intersection(a.author_id for a in paper.authors))
    is_self_author = lambda paper: any(author.author_id == a.author_id for a in paper.authors)
    self_cites = dict()
    pseudo_self_cites = dict()
    for paper_id, citations in tqdm(citations_per_paper.items()):
        self_cites[paper_id] = sum([1 for c in citations if is_self_author(c.citing_paper)])
        pseudo_self_cites[paper_id] = sum([1 for c in citations if is_pseudo_author(c.citing_paper)])

    papers_df["self_cites"] = papers_df.paper_id.map(self_cites)
    papers_df["pseudo_self_cites"] = papers_df.paper_id.map(pseudo_self_cites)
    papers_df["self_cite_ratio"] = papers_df.self_cites / papers_df.citation_count
    papers_df["pseudo_self_cite_ratio"] = papers_df.pseudo_self_cites / papers_df.citation_count

    papers_df["citation_count_without_self_cites"] = papers_df.citation_count - papers_df.self_cites
    papers_df["citation_count_without_pseudo_self_cites"] = papers_df.citation_count - papers_df.pseudo_self_cites

    author_df["self_cites"] = sum(papers_df.self_cites)
    author_df["pseudo_self_cites"] = sum(papers_df.pseudo_self_cites)

    author_df["h_index_without_self_cites"] = compute_h_index(papers_df.citation_count_without_self_cites)
    author_df["h_index_without_pesudo_self_cites"] = compute_h_index(papers_df.citation_count_without_self_cites)

    author_df["s_index"] = compute_h_index(papers_df.self_cites)
    author_df["ps_index"] = compute_h_index(papers_df.pseudo_self_cites)

    author_df["i10_index"] = compute_i_k_index(papers_df.citation_count, 10)
    author_df["i10_index_without_self_cites"] = compute_i_k_index(papers_df.citation_count_without_self_cites, 10)
    author_df["i10_index_without_pseudo_self_cites"] = compute_i_k_index(
        papers_df.citation_count_without_pseudo_self_cites, 10
    )

    author_df["self_cite_ratio"] = author_df["self_cites"] / author.citation_count
    author_df["pseudo_self_cite_ratio"] = author_df["pseudo_self_cites"] / author.citation_count
    author_df["co_author_ids"] = ", ".join(co_author_ids)
    author_df["co_author_count"] = len(co_author_ids)
    return author_df, papers_df


def compute_h_index(citations: List[int]):
    """Given a list of citations (integers) compute the h-index."""
    citations = np.asarray(citations)
    n = citations.shape[0]
    index = np.arange(1, n + 1)
    citations = np.sort(citations)[::-1]  # reverse sorting
    h_idx = np.max(np.minimum(citations, index))  # intersection of citations and k
    return h_idx


def compute_i_k_index(citations: List[int], k: int = 10):
    """Given a list of citations (integers) compute the i-k-index (default i10)."""
    citations = np.asarray(citations)
    i_k_index = (citations > k).sum()
    return i_k_index


def handle_author(author_id: str, ss: ssapi.SemanticScholarAPI = None):
    """Request author info from API and dump results to disk. Read from disk if avaiable and less than a week old."""
    author_dir = os.path.join(DATA_DIR, f"{author_id}_author.pkl")
    papers_dir = os.path.join(DATA_DIR, f"{author_id}_papers_dir.pkl")
    if (
        os.path.exists(author_dir)
        and os.path.exists(papers_dir)
        and time.time() - os.path.getmtime(author_dir) < 60 * 60 * 24 * 7
    ):
        LOGGER.info(f"Skipping {author_id} since fresh data is available on disk.")
        return pd.read_pickle(author_dir), pd.read_pickle(papers_dir)
    else:
        LOGGER.info(f"Requesting data for author {author_id} from API.")
        author_df, papers_df = compute_self_citation_count(author_id, ss=ss)

    author_df.to_pickle(author_dir)
    papers_df.to_pickle(papers_dir)
    return author_df, papers_df


if __name__ == "__main__":
    os.makedirs(f"{DATA_DIR}", exist_ok=True)

    parser = argparse.ArgumentParser(description="Compute self citation counts for Semantic Scholar authors.")
    parser.add_argument("--authors", type=str, nargs="+", default=[], help="Author names to search for.")
    parser.add_argument("--author_ids", type=str, nargs="+", default=[], help="Author ids to search for.")
    parser.add_argument("--num_threads", default=25, type=int, help="Number of threads for concurrent requests.")

    args = parser.parse_args()

    ss = ssapi.SemanticScholarAPI(num_threads=args.num_threads)

    LOGGER.info("Searching for authors...")
    authors = [ss.author_search(author_name, fields=ssapi.FIELDS_AUTHOR_BASIC) for author_name in args.authors]
    authors = [sorted(results, key=lambda a: a.citation_count, reverse=True) for results in authors]
    authors = [authors_[0] for authors_ in authors]
    authors += [ssapi.Author(author_id) for author_id in args.author_ids]  # add author ids
    LOGGER.info(f"Found authors:")
    rich.print(authors)

    author_df = dict()
    papers_df = dict()
    for author in authors:
        author_df[author.author_id], papers_df[author.author_id] = handle_author(author.author_id, ss=ss)

    for author_id, df in author_df.items():
        name = df.name.values[0] + f" ({author_id})"
        cites = f"citations={df.citation_count.item():<5d}"
        self_cites = f"sc={df.self_cites.item():<5d}"
        pseudo_self_cites = f"psc={df.pseudo_self_cites.item():<5d}"
        h_index = f"h={df.h_index.item():<3d}"
        h_index_corrected = f"h_corrected={df.h_index_without_self_cites.item():<3d}"
        i_index = f"i10={df.i10_index.item():<3d}"
        i_index_corrected = f"i10_corrected={df.i10_index_without_self_cites.item():<3d}"
        s_index = f"s={df.s_index.item():<3d}"
        ps_index = f"ps={df.ps_index.item():<3d}"
        scr = f"scr={df.self_cite_ratio.item():5.3f}"
        pscr = f"pscr={df.pseudo_self_cite_ratio.item():5.3f}"
        rich.print(
            f"{name:<30s} {cites} {self_cites} {pseudo_self_cites} {h_index} {h_index_corrected} {i_index} {i_index_corrected} {s_index} {ps_index} {scr} {pscr}"
        )
