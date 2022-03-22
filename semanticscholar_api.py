"""
https://www.semanticscholar.org/product/api

https://api.semanticscholar.org/graph/v1

https://blog.allenai.org/building-a-better-search-engine-for-semantic-scholar-ea23a0b661e7

https://github.com/danielnsilva/semanticscholar/blob/master/semanticscholar/SemanticScholar.py

https://tenacity.readthedocs.io/en/latest/
"""

from __future__ import annotations

import logging

from concurrent.futures import ThreadPoolExecutor

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Union
from tenacity import retry, wait_fixed, retry_if_exception_type, stop_after_attempt

import requests

from tqdm import tqdm
from dataclass_wizard import JSONWizard


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)7s %(thread)d | %(message)s")
LOGGER = logging.getLogger(__name__)


RETRY_WAIT_SECONDS = 30
NUM_RETRY_ATTEMPTS = 100
REQUEST_TIMEOUT = 30

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
