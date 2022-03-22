"""
https://www.semanticscholar.org/product/api

https://api.semanticscholar.org/graph/v1

https://blog.allenai.org/building-a-better-search-engine-for-semantic-scholar-ea23a0b661e7

https://github.com/danielnsilva/semanticscholar/blob/master/semanticscholar/SemanticScholar.py

https://tenacity.readthedocs.io/en/latest/
"""

from __future__ import annotations


import argparse
import logging
import os
import re

from dataclasses import dataclass
from functools import partial
import time
from typing import Callable, Dict, List, Optional, Union
from tenacity import retry, wait_fixed, retry_if_exception_type, stop_after_attempt

import requests
import rich
import pandas as pd
import numpy as np

from tqdm import tqdm
from dataclass_wizard import JSONWizard


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)7s %(thread)d | %(message)s")
LOGGER = logging.getLogger(__name__)


RETRY_WAIT_SECONDS = 30
NUM_RETRY_ATTEMPTS = 100
REQUEST_TIMEOUT = 30

DATA_DIR = "./data"

API_URL = "https://api.semanticscholar.org/graph/v1/"
PARTNER_API_URL = "https://partner.semanticscholar.org/v1"

PAPER_SEARCH = "paper/search"  # Get list of papers that match the query (batched)
PAPER_DETAILS = "paper/{paper_id}"  # Get details about this paper.
PAPER_AUTHORS = "paper/{paper_id}/authors"  # Get details about the authors of the this paper (batched)
PAPER_CITATIONS = "paper/{paper_id}/citations"  # Get details about the papers that cite this paper (batched)
PAPER_REFERENCES = "paper/{paper_id}/references"  # Get details about the papers cited by this paper (batched)

AUTHOR_SEARCH = "author/search"  # Get list of authors that match the query.
AUTHOR_DETAILS = "author/{author_id}"  # Get details about this author (batched).
AUTHOR_PAPERS = "author/{author_id}/papers"  # Get details about the papers of this author (batched).


# Collections of sets of fields
FIELDS_AUTHOR_BASIC = ["authorId", "name", "affiliations", "paperCount", "citationCount", "hIndex"]
FIELDS_PAPER_BASIC = [
    "title",
    "venue",
    "year",
    "citationCount",
    "referenceCount",
    "influentialCitationCount",
    "authors",
]
FIELDS_PAPER_WITHOUT_CITATIONS = [
    "citationCount",
    "externalIds",
    "url",
    "title",
    "abstract",
    "venue",
    "year",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
    "isOpenAccess",
    # "fieldsOfStudy",
    # "s2FieldsOfStudy",
    "authors",
]
FIELDS_PAPER_WITH_CITATIONS = FIELDS_PAPER_WITHOUT_CITATIONS + ["citations", "references"]


def camel2snake(txt: str) -> str:
    """Convert a CamelCasedString to a snake_cased_string"""
    txt = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", txt)
    txt = re.sub("__([A-Z])", r"_\1", txt)
    txt = re.sub("([a-z0-9])([A-Z])", r"\1_\2", txt)
    return txt.lower()


def to_params(fields: List[str] = None, offset: int = None, limit: int = None, **kwargs) -> Dict[str, str]:
    """Form parameters dictionary for API get requests. See https://api.semanticscholar.org/graph/v1."""
    params = dict()
    if fields:
        params["fields"] = ",".join(fields)
    if offset:
        params["offset"] = str(offset)
    if limit:
        params["limit"] = str(limit)
    params.update(**kwargs)
    return params


class APIError(Exception):
    """Base exception for all SemanticScholar API-related errors."""

    def __init__(self, response: requests.Response):
        error_code = response.status_code
        url = response.url
        data = response.json()
        error_msg = ", ".join((f"{key}: {value}" for key, value in data.items()))
        if not error_msg:
            error_msg = "No error message provided"
        error_message = f"HTTP {error_code} for request {url}: {error_msg}"
        super().__init__(error_message)


class BadQueryError(APIError):
    """HTTP 400 Bad Query c.f. https://api.semanticscholar.org/graph/v1"""


class BadIDError(APIError):
    """HTTP 404 Bad ID c.f. https://api.semanticscholar.org/graph/v1"""


class TooManyRequests(APIError):
    """HTTP 429 Too many requests https://api.semanticscholar.org/graph/v1"""


class Non200Warning(APIError):
    """Non-errror non-200 HTTP response"""


@retry(
    wait=wait_fixed(RETRY_WAIT_SECONDS),
    retry=retry_if_exception_type(
        (
            TooManyRequests,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
        )
    ),
    stop=stop_after_attempt(NUM_RETRY_ATTEMPTS),
)
def get_and_parse(url: str, params: dict = None, headers: dict = None, timeout: float = None, **kwargs) -> dict:
    """Get data from url in dictionary with `requests.get` with automatic retries connection errors"""
    # Make get request
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as e:
        LOGGER.error(f"Got a {e} error.")
        raise e
    except requests.exceptions.HTTPError as e:
        LOGGER.error(f"Got a {e} error.")
        raise e
    except requests.exceptions.TooManyRedirects as e:
        LOGGER.error(f"Got a {e} error.")
        raise e
    except requests.exceptions.ReadTimeout as e:
        LOGGER.error(f"Got a {e} error. Retrying in {RETRY_WAIT_SECONDS}s.")
        raise e
    except requests.exceptions.Timeout as e:
        LOGGER.error(f"Got a {e} error. Retrying in {RETRY_WAIT_SECONDS}s.")
        raise e

    # Parse response
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Handle specific know error codes, raise otherwise
        if response.status_code == 400:
            raise BadQueryError(response)

        if response.status_code == 404:
            raise BadIDError(response)

        if response.status_code == 429:
            error = TooManyRequests(response)
            LOGGER.error(f"Got a {error}. Retrying in {RETRY_WAIT_SECONDS}s.")
            raise error

        raise e

    if response.status_code != 200:
        warning = Non200Warning(response)
        LOGGER.warning(f"Got a {warning} for request: {url}.")

    return response.json()


@dataclass
class BaseContainer(JSONWizard):
    """BaseContainer is a base data container that is used to hold and parse data returned by the Semantic Scholar API.
    
    Using `BaseContainer.from_dict(data)` where `data = request.json()` will parse the `data` dictionary and return the
    corresponding dataclass container object.
    
    We define specific dataclasses for the different endpoints below. Some return `data` is shared between endpoints.
    """
    def __repr__(self):
        kvs = [f"{k}={self[k]}" for k in self.__dataclass_fields__ if self[k] is not None]
        return f"{self.__class__.__name__}({kvs})"

    def __str__(self):
        return repr(self)


@dataclass
class PaperEmbedding(BaseContainer):
    """Vector-space embedding of a paper."""
    model: str
    vector: List[float]


@dataclass
class PaperTLDR(BaseContainer):
    """Model-generated TLDR summary of a paper."""
    model: str
    text: str


@dataclass
class PaperWithoutCitations(BaseContainer):
    """A paper without citations (can be used recursively within `Paper`)."""
    paper_id: str  # Always included
    url: str = None
    title: str = None  # Included if no fields are specified
    venue: str = None
    year: int = None
    authors: List[AuthorWithoutPapers] = None


@dataclass
class Paper(BaseContainer):
    """A full paper with citations and references."""
    paper_id: str  # Always included
    external_ids: Dict[str, str] = None
    url: str = None
    title: str = None  # Included if no fields are specified
    abstract: str = None
    venue: str = None
    year: int = None
    reference_count: int = None
    citation_count: int = None
    influential_citation_count: int = None
    is_open_access: bool = None
    fields_of_study: Optional[List[str]] = None
    s2_fields_of_study: Optional[List[Dict[str, str]]] = None
    authors: List[AuthorWithoutPapers] = None  # Returns max 500. Alternatively: /paper/{paper_id}/authors endpoint
    citations: List[
        PaperWithoutCitations
    ] = None  # Returns max 1000. Alternatively: /paper/{paper_id}/citations endpoint
    references: List[
        PaperWithoutCitations
    ] = None  # Returns max 1000. Alternatively: /paper/{paper_id}/references endpoint
    embedding: PaperEmbedding = None  # Vector embedding of paper content from the SPECTER model
    tldr: PaperTLDR = None  # Auto-generated short summary of the paper from the SciTLDR model


@dataclass
class Citation(BaseContainer):
    """A citation of a paper by the given paper. Returned by `PAPER_CITATIONS` endpoint."""
    citing_paper: Paper  # The citing paper
    contexts: List[str] = None  # List of sentences in which the citation(s) were made.
    intents: List[str] = None
    is_influential: bool = None  # Did the cited paper have a significant impact on the citing paper?


@dataclass
class Reference(BaseContainer):
    """A reference from the given paper to a paper. Returned by `PAPER_REFERENCES` endpoint."""
    contexts: List[str]  # List of sentences in which the citation(s) were made.
    intents: List[str]
    is_influential: bool  # Did the cited paper have a significant impact on the citing paper?
    cited_paper: Paper  # The cited paper


@dataclass
class AuthorWithoutPapers(BaseContainer):
    """An author without papers (can be used recursively within `Author`)."""
    author_id: str
    name: str = None
    external_ids: str = None
    url: str = None
    aliases: str = None
    affiliations: str = None
    homepage: str = None
    paper_count: int = None
    citation_count: int = None
    h_index: int = None


@dataclass
class Author(AuthorWithoutPapers):
    """An author with papers."""
    papers: List[Paper] = None


@dataclass
class Batch(BaseContainer):
    """A batch of Papers, Authors, Citations or References. Returned by any batched endpoint."""
    data: Union[List[Author], List[Paper], List[Citation], List[Reference]]  # list of objects returned in this batch.
    offset: int = None  # starting index for this batch. Required.
    next: int = None  # starting index for next batch. Missing if no more data exists.
    total: int = None  # total number of search matches, only returned by `AUTHOR_SEARCH` and `PAPER_SEARCH`.


@dataclass
class PaperBatch(Batch):
    data: List[Paper]


@dataclass
class AuthorBatch(Batch):
    data: List[Author]


@dataclass
class CitationBatch(Batch):
    data: List[Citation]


@dataclass
class ReferenceBatch(Batch):
    data: List[Reference]


def query_exhaustively(
    endpoint: Callable,
    query_or_id: str,
    fields: Dict[str, str] = None,
    offset: int = None,
    limit: int = None,
    api_url: str = API_URL,
    **kwargs,
) -> Union[List[Author], List[Paper], List[Citation], List[Reference]]:
    """Keep requesting the `endpoint` with `query_or_id` and return the concatenated `Batch.data`."""
    batch = endpoint(query_or_id, fields, offset, limit, api_url, **kwargs)
    data = batch.data
    while batch.next is not None:
        offset = batch.next
        batch = endpoint(query_or_id, fields, offset, limit, api_url, **kwargs)
        data.extend(batch.data)
    return data


def author_search_batch(
    query: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> AuthorBatch:
    """"""
    params = to_params(fields, offset, limit, query=query)
    data = get_and_parse(api_url + AUTHOR_SEARCH, params=params, **kwargs)
    return AuthorBatch.from_dict(data)


def author_search(
    query: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> List[Author]:
    """"""
    return query_exhaustively(author_search_batch, query, fields, offset, limit, api_url, **kwargs)


def author_details(author_id: str, fields: List[str] = None, api_url: str = API_URL, **kwargs) -> Author:
    """"""
    params = to_params(fields)
    data = get_and_parse(api_url + AUTHOR_DETAILS.format(author_id=author_id), params=params, **kwargs)
    return Author.from_dict(data)


def author_papers_batch(
    author_id: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> PaperBatch:
    """"""
    params = to_params(fields, offset, limit)
    data = get_and_parse(api_url + AUTHOR_PAPERS.format(author_id=author_id), params=params, **kwargs)
    return PaperBatch.from_dict(data)


def author_papers(
    author_id: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> List[Paper]:
    """"""
    return query_exhaustively(author_papers_batch, author_id, fields, offset, limit, api_url, **kwargs)


def paper_search_batch(
    query: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> PaperBatch:
    """"""
    params = to_params(fields, offset, limit, query=query)
    data = get_and_parse(api_url + PAPER_SEARCH, params=params, **kwargs)
    return PaperBatch.from_dict(data)


def paper_search(
    query: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> List[Paper]:
    """"""
    return query_exhaustively(paper_search_batch, query, fields, offset, limit, api_url, **kwargs)


def paper_details(paper_id: str, fields: List[str] = None, api_url: str = API_URL, **kwargs) -> Paper:
    """"""
    params = to_params(fields)
    data = get_and_parse(api_url + PAPER_DETAILS.format(paper_id=paper_id), params=params, **kwargs)
    return Paper.from_dict(data)


def paper_authors_batch(
    paper_id: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> AuthorBatch:
    """"""
    params = to_params(fields, offset, limit)
    data = get_and_parse(api_url + PAPER_AUTHORS.format(paper_id=paper_id), params=params, **kwargs)
    return AuthorBatch.from_dict(data)


def paper_authors(
    paper_id: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> List[Author]:
    """"""
    return query_exhaustively(paper_authors_batch, paper_id, fields, offset, limit, api_url, **kwargs)


def paper_citations_batch(
    paper_id: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> CitationBatch:
    """"""
    params = to_params(fields, offset, limit)
    data = get_and_parse(api_url + PAPER_CITATIONS.format(paper_id=paper_id), params=params, **kwargs)
    return CitationBatch.from_dict(data)


def paper_citations(
    paper_id: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> List[Citation]:
    """"""
    return query_exhaustively(paper_citations_batch, paper_id, fields, offset, limit, api_url, **kwargs)


def paper_references_batch(
    paper_id: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> ReferenceBatch:
    """"""
    params = to_params(fields, offset, limit)
    data = get_and_parse(api_url + PAPER_REFERENCES.format(paper_id=paper_id), params=params, **kwargs)
    return ReferenceBatch.from_dict(data)


def paper_references(
    paper_id: str, fields: List[str] = None, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> List[Reference]:
    """"""
    return query_exhaustively(paper_references_batch, paper_id, fields, offset, limit, api_url, **kwargs)


class SemanticScholarAPI:
    """Class that facilitates interacting with the SemanticsScholar API.
    
    Non-partners are limited to 100 queries per 5 minutes. Will automatically keep retrying until whitelisted again.
    """
    DEFAULT_API_URL = API_URL
    DEFAULT_PARTNER_API_URL = PARTNER_API_URL

    def __init__(self, timeout: int = REQUEST_TIMEOUT, api_key: str = None, api_url: str = None):
        self.timeout = timeout
        self.api_url = api_url if api_url else self.DEFAULT_API_URL
        self.auth_header = {"x-api-key": api_key} if api_key else {}
        self.kwargs = dict(api_url=self.api_url, timeout=timeout, headers=self.auth_header)

        self.executor = ThreadPoolExecutor(max_workers=100)

        self.author_search = partial(author_search, **self.kwargs)
        self.author_details = partial(author_details, **self.kwargs)
        self.author_papers = partial(author_papers, **self.kwargs)

        self.author_search_batch = partial(author_search_batch, **self.kwargs)
        self.author_papers_batch = partial(author_papers_batch, **self.kwargs)

        self.author_search_threaded = partial(self.argument_parallel_requests, author_search)
        self.author_papers_threaded = partial(self.argument_parallel_requests, author_papers)

        self.paper_search = partial(paper_search, **self.kwargs)
        self.paper_details = partial(paper_details, **self.kwargs)
        self.paper_authors = partial(paper_authors, **self.kwargs)
        self.paper_citations = partial(paper_citations, **self.kwargs)
        self.paper_references = partial(paper_references, **self.kwargs)

        self.paper_search_batch = partial(paper_search_batch, **self.kwargs)
        self.paper_authors_batch = partial(paper_authors_batch, **self.kwargs)
        self.paper_citations_batch = partial(paper_citations_batch, **self.kwargs)
        self.paper_references_batch = partial(paper_references_batch, **self.kwargs)

        self.paper_search_threaded = partial(self.argument_parallel_requests, paper_search)
        self.paper_authors_threaded = partial(self.argument_parallel_requests, paper_authors)
        self.paper_citations_threaded = partial(self.argument_parallel_requests, paper_citations)
        self.paper_references_threaded = partial(self.argument_parallel_requests, paper_references)

    def argument_parallel_requests(
        self,
        endpoint: Callable,
        queries_or_ids: Union[str, List[str]],
        fields: List[str],
        progressbar: bool = True,
        **kwargs,
    ) -> Union[List[List[Author]], List[List[Paper]], List[List[Citation]], List[List[Reference]]]:
        """Submit many `queries_or_ids` arguments to the API `endpoint` to be executed concurrently.

        The return values are in the same order as `queries_or_ids`.
        """
        if isinstance(queries_or_ids, str):
            queries_or_ids = [queries_or_ids]
        
        if progressbar:
            desc = kwargs.pop("desc", f"{endpoint.__name__}")
            pbar = partial(tqdm, desc=desc)
        else:
            pbar = lambda x: x

        futures = [self.executor.submit(endpoint, qid, fields, **self.kwargs, **kwargs) for qid in queries_or_ids]
        results = [future.result() for future in pbar(futures)]
        return results


from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue


class SemanticCrawler:
    """Crawl across SemanticScholar to obtain all papers and authors while writing to database."""

    def __init__(
        self,
        search_seeds: List[str],
        priority_key: str = "citation_count",
        timeout: int = REQUEST_TIMEOUT,
        api_key: str = None,
        api_url: str = None,
    ):
        self.search_seeds = search_seeds
        self.priority_key = priority_key
        self.api = SemanticScholarAPI(timeout, api_key, api_url)
        self.in_queue = PriorityQueue()  # prioritize authors by citation count to get the most bang for the buck

    def submit_author(self, author_id):
        raise NotImplementedError

    def submit_paper(self, paper_id: str):
        raise NotImplementedError

    def work(self):
        raise NotImplementedError


def papers2df(papers: List[Paper]):
    papers_df = pd.DataFrame(papers)
    papers_df.dropna(axis=1, how="all", inplace=True)  # Drop all columns that have only NA values
    papers_df.columns = [camel2snake(str(x)) for x in papers_df.columns]  # Convert camelCase columns to snake_case
    return papers_df


def authors2df(authors: List[Author]):
    author_dict = author.to_dict()
    author_dict.pop("papers")
    author_df = pd.DataFrame([author_dict])
    author_df.columns = [camel2snake(str(x)) for x in author_df.columns]  # Convert camelCase columns to snake_case
    return author_df


def get_author_details(author_id: str, ss: SemanticScholarAPI):
    fields = [f"papers.{f}" for f in FIELDS_PAPER_BASIC] + FIELDS_AUTHOR_BASIC
    author = ss.author_details(author_id, fields=fields)
    papers_df = papers2df(author.papers)
    author_df = authors2df(author)
    return author, author_df, papers_df


def compute_self_citation_count(author_id: str, ss: SemanticScholarAPI = None) -> int:
    if ss is None:
        ss = SemanticScholarAPI()  # default

    author, author_df, papers_df_trunc = get_author_details(author_id, ss)
    
    papers = ss.author_papers(author_id, fields=FIELDS_PAPER_BASIC)
    papers_df = papers2df(papers)

    citations_per_paper = ss.paper_citations_threaded(papers_df.paper_id, FIELDS_PAPER_BASIC)
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
    author_df["i10_index_without_pseudo_self_cites"] = compute_i_k_index(papers_df.citation_count_without_pseudo_self_cites, 10)
    
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


def compute_h_index_without_self_cites(author_id: str, ss: SemanticScholarAPI = None):
    """Subtract self cites from citation count of each paper and then compute h-index."""
    if ss is None:
        ss = SemanticScholarAPI()  # default


def concat_paper_dfs(paper_dfs: List[pd.DataFrame], author_ids: List[str]) -> pd.DataFrame:
    """Concatenate multiple paper dataframes."""
    for author_id, df in zip(author_ids, paper_dfs):
        df["author_id"] = author_id
    return pd.concat(paper_dfs)


def handle_author(author_id: str, ss: SemanticScholarAPI = None):
    """Request author info from API and dump results to disk. Read from disk if avaiable and less than a week old."""
    author_dir = os.path.join(DATA_DIR, f"{author_id}_author.pkl")
    papers_dir = os.path.join(DATA_DIR, f"{author_id}_papers_dir.pkl")
    if os.path.exists(author_dir) and os.path.exists(papers_dir) and \
       time.time() - os.path.getmtime(author_dir) < 60 * 60 * 24 * 7:
        LOGGER.info(f"Skipping {author_id} since fresh data is available on disk.")
        return pd.read_pickle(author_dir), pd.read_pickle(papers_dir)
    else:
        LOGGER.info(f"Requesting data for author {author_id} from API.")
        author_df, papers_df = compute_self_citation_count(author.author_id, ss=ss)

    author_df.to_pickle(author_dir)
    papers_df.to_pickle(papers_dir)
    return author_df, papers_df


if __name__ == "__main__":
    os.makedirs(f"{DATA_DIR}", exist_ok=True)

    parser = argparse.ArgumentParser(description="Compute self citation counts for Semantic Scholar authors.")
    parser.add_argument("--authors", type=str, nargs="+", help="Author names to search for.")

    args = parser.parse_args()

    ss = SemanticScholarAPI()
    
    LOGGER.info("Searching for authors...")
    authors = [ss.author_search(author_name, fields=FIELDS_AUTHOR_BASIC) for author_name in args.authors]
    authors = [sorted(results, key=lambda a: a.citation_count, reverse=True) for results in authors]
    authors = [authors_[0] for authors_ in authors]
    LOGGER.info(f"Found authors:")
    rich.print(authors)

    author_df = dict()
    papers_df = dict()
    for author in authors:
        author_df[author.name], papers_df[author.name] = handle_author(author.author_id, ss=ss)

    for name, df in author_df.items():
        cites = f"citations={df.citation_count.item():<5d}"
        self_cites = f"SC={df.self_cites.item():<5d}"
        pseudo_self_cites = f"SC={df.pseudo_self_cites.item():<5d}"
        h_index = f"h={df.h_index.item():<3d}"
        h_index_corrected = f"h_corrected={df.h_index_without_self_cites.item():<3d}"
        i_index = f"i10={df.i10_index.item():<3d}"
        i_index_corrected = f"i10_corrected={df.i10_index_without_self_cites.item():<3d}"
        s_index =  f"s={df.s_index.item():<3d}"
        ps_index = f"ps={df.ps_index.item():<3d}"
        scf = f"scf={df.self_cite_ratio.item():5.3f}"
        pscf = f"pscf={df.pseudo_self_cite_ratio.item():5.3f}"
        rich.print(f"{name:<30s} {cites} {self_cites} {pseudo_self_cites} {h_index} {h_index_corrected} {i_index} {i_index_corrected} {s_index} {ps_index} {scf} {pscf}")
        # rich.print(f"{name:30s}: {scf} {pscf} {h_index} {h_index_corrected} {i_index} {i_index_corrected} {cites} {self_cites} {pseudo_self_cites}")
