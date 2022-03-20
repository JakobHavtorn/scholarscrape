"""
https://www.semanticscholar.org/product/api

https://api.semanticscholar.org/graph/v1#tag/paper

https://blog.allenai.org/building-a-better-search-engine-for-semantic-scholar-ea23a0b661e7

https://github.com/danielnsilva/semanticscholar/blob/master/semanticscholar/SemanticScholar.py

https://tenacity.readthedocs.io/en/latest/
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Union
from typing_extensions import Self  # TODO Import from typing in Python 3.11
from tenacity import retry, wait_fixed, retry_if_exception_type, stop_after_attempt

import requests

from dataclass_wizard import JSONWizard, fromdict, asdict


class BadIDError(Exception):
    """HTTP 404"""

    pass


RETRY_WAIT_SECONDS = 30
NUM_RETRY_ATTEMPTS = 10


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


def to_params(fields: Dict[str, str] = None, offset: int = None, limit: int = None, **kwargs) -> Dict[str, str]:
    kvs = [("offset", offset), ("limit", limit), ("fields", fields)]
    out = {k: v for k, v in kvs if v is not None}
    out.update(**kwargs)
    return out


@retry(
    wait=wait_fixed(RETRY_WAIT_SECONDS),
    retry=retry_if_exception_type([ConnectionRefusedError, requests.ConnectTimeout, requests.ConnectionError]),
    stop=stop_after_attempt(NUM_RETRY_ATTEMPTS),
)
def get_and_parse(url: str, params: dict = None, headers: dict = None, timeout: float = None, **kwargs) -> dict:
    """Get data from url in dictionary with `requests.get` with automatic retries connection errors"""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout, **kwargs)
    except requests.ConnectionError as e:
        raise e
    except requests.HTTPError as e:
        raise e
    except requests.TooManyRedirects as e:
        raise e
    except requests.Timeout as e:
        raise e
    return response.json()


@dataclass
class PaperEmbedding(JSONWizard):
    model: str
    vector: List[float]


@dataclass
class PaperTLDR(JSONWizard):
    model: str
    text: str


@dataclass
class Paper(JSONWizard):
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
    fields_of_study: List[str] = None
    s2_fields_of_study: List[Dict[str, str]] = None
    authors: List[AuthorWithoutPapers] = None  # Returns max 500. Alternatively: /paper/{paper_id}/authors endpoint
    citations: List[Self] = None  # Returns max 1000. Alternatively: /paper/{paper_id}/citations endpoint
    references: List[Self] = None  # Returns max 1000. Alternatively: /paper/{paper_id}/references endpoint
    embedding: PaperEmbedding = None  # Vector embedding of paper content from the SPECTER model
    tldr: PaperTLDR = None  # Auto-generated short summary of the paper from the SciTLDR model


# @dataclass
# class PaperBatch(JSONWizard):
#     offset: int
#     next: int
#     data: List[Paper]
#     total: int = None


# class PaperBatchIterator:
#     def __init__(self, action: str, query: str, offset: int, limit: int, fields: Dict[str, str]):
#         self.action = action
#         self.query = query
#         self.offset = offset
#         self.limit = limit
#         self.fields = fields

#     def __iter__(self):
#         return self

#     def __next__(self):
#         data = paper_query(self.action, self.query, self.offset, self.limit, self.fields)
#         self.offset = data.next
#         return data


@dataclass
class Citation(JSONWizard):
    contexts: List[str]  # List of sentences in which the citation(s) were made.
    intents: List[str]
    is_influential: bool  # Did the cited paper have a significant impact on the citing paper?
    citing_paper: Paper  # The citing paper


@dataclass
class Reference(JSONWizard):
    contexts: List[str]  # List of sentences in which the citation(s) were made.
    intents: List[str]
    is_influential: bool  # Did the cited paper have a significant impact on the citing paper?
    cited_paper: Paper  # The cited paper


@dataclass
class AuthorWithoutPapers(JSONWizard):
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
    papers: List[Paper] = None


@dataclass
# class AuthorBatch(JSONWizard):
#     offset: int
#     data: List[Author]
#     next: int = None  # missing if no more data exists
#     total: int = None  # only returned by `AUTHOR_SEARCH`


@dataclass
class Batch(JSONWizard):
    offset: int  # starting index for this batch.
    data: Union[List[Author], List[Paper], List[Citation], List[Reference]]  # list of objects returned in this batch.
    next: int = None  # starting index for next batch. Missing if no more data exists.
    total: int = None  # total number of search matches, only returned by `AUTHOR_SEARCH` and `PAPER_SEARCH`.


# class AuthorBatchIterator:
#     def __init__(self, action: str, query: str, offset: int, limit: int, fields: Dict[str, str]):
#         self.action = action
#         self.query = query
#         self.offset = offset
#         self.limit = limit
#         self.fields = fields

#     def __iter__(self):
#         return self

#     def __next__(self):
#         data = author_query(self.action, self.query, self.offset, self.limit, self.fields)
#         self.offset = data.next
#         return data


def author_search(
    query: str, fields: str, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> Batch[Author]:
    """"""
    params = to_params(fields, offset, limit, query=query)
    data = get_and_parse(api_url + AUTHOR_SEARCH, params=params, **kwargs)
    return Batch.from_dict(data)


def author_details(author_id: str, fields: str, api_url: str = API_URL, **kwargs) -> Author:
    """"""
    params = to_params(fields)
    data = get_and_parse(api_url + AUTHOR_DETAILS.format(author_id=author_id), params=params, **kwargs)
    return Author.from_dict(data)


def author_papers(
    author_id: str, fields: str, offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> Batch[Paper]:
    """"""
    params = to_params(fields, offset, limit)
    data = get_and_parse(api_url + AUTHOR_SEARCH.format(author_id=author_id), params=params, **kwargs)
    return Batch.from_dict(data)


def paper_search(
    query: str, fields: Dict[str, str], offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> Batch[Paper]:
    """"""
    params = to_params(fields, offset, limit, query=query)
    data = get_and_parse(api_url + PAPER_SEARCH, params=params, **kwargs)
    return Batch.from_dict(data)


def paper_details(paper_id: str, fields: Dict[str, str], api_url: str = API_URL, **kwargs) -> Paper:
    """"""
    params = to_params(fields)
    data = get_and_parse(api_url + PAPER_DETAILS.format(paper_id=paper_id), params=params, **kwargs)
    return Paper.from_dict(data)


def paper_authors(
    paper_id: str, fields: Dict[str, str], offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> Batch[Author]:
    """"""
    params = to_params(fields, offset, limit)
    data = get_and_parse(api_url + PAPER_AUTHORS.format(paper_id=paper_id), params=params, **kwargs)
    return Batch.from_dict(data)


def paper_citations(
    paper_id: str, fields: Dict[str, str], offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> Batch[Citation]:
    """"""
    params = to_params(fields, offset, limit)
    data = get_and_parse(api_url + PAPER_CITATIONS.format(paper_id=paper_id), params=params, **kwargs)
    return Batch.from_dict(data)


def paper_references(
    paper_id: str, fields: Dict[str, str], offset: int = None, limit: int = None, api_url: str = API_URL, **kwargs
) -> Batch[Reference]:
    """"""
    params = to_params(fields, offset, limit)
    data = get_and_parse(api_url + PAPER_REFERENCES.format(paper_id=paper_id), params=params, **kwargs)
    return Batch.from_dict(data)


AUTHOR_METHODS = dict(
    search=author_search,
    details=author_details,
    papers=author_papers,
)


PAPER_METHODS = dict()


def author_query(query: str, action: str, offset: int, limit: int, fields: Dict[str, str]):
    """query (author_id or query for search)"""
    if not action in AUTHOR_METHODS:
        raise ValueError(f"Invalid action: {action}")

    method = AUTHOR_METHODS[action]
    return method(query, fields, offset, limit)


def paper_query(paper_id: str, endpoint: str, fields: dict = None, limit: int = None):
    """
    paper_id (str): The paper ID. Any one of
                    - <sha> - a Semantic Scholar ID, e.g. 649def34f8be52c8b66281af98ae884c09aef38b
                    - CorpusId:<id> - Semantic Scholar numerical ID, e.g. 215416146
                    - DOI:<doi> - a Digital Object Identifier, e.g. DOI:10.18653/v1/N18-3011
                    - ARXIV:<id> - arXiv.rg, e.g. ARXIV:2106.15928
                    - MAG:<id> - Microsoft Academic Graph, e.g. MAG:112218234
                    - ACL:<id> - Association for Computational Linguistics, e.g. ACL:W12-3903
                    - PMID:<id> - PubMed/Medline, e.g. PMID:19872477
                    - PMCID:<id> - PubMed Central, e.g. PMCID:2323736
                    - URL:<url> - URL from one of the sites listed below, e.g. URL:https://arxiv.org/abs/2106.15928v1

                    URLs are recognized from the following sites:
                    - semanticscholar.org
                    - arxiv.org
                    - aclweb.org
                    - acm.org
                    - biorxiv.org
    """
    if not endpoint in PAPER_METHODs:
        raise ValueError(f"Invalid endpoint: {endpoint}")

    json = dict(fields=fields, limit=limit)
    r = requests.get(API_URL, json=json)
    return r.json()


class SemanticScholarAPI:
    DEFAULT_API_URL = API_URL
    DEFAULT_PARTNER_API_URL = PARTNER_API_URL

    def __init__(self, timeout: int = 2, api_key: str = None, api_url: str = None):
        self.timeout = timeout
        self.api_url = api_url if api_url else self.DEFAULT_API_URL
        self.auth_header = {"x-api-key": api_key} if api_key else {}
        self.kwargs = dict(api_url=self.api_url, timeout=timeout, headers=self.auth_header)

    def author_search_batch(self, query: str, fields: str = None, offset: int = None, limit: int = None) -> Batch[Author]:
        return author_search(query, fields, offset, limit, **self.kwargs)

    def author_search():
        """Call author_search_batch iteratively until no more batches are available. Return one large Batch"""
        raise NotImplementedError()

    def author_details(self, author_id: str, fields: str = None) -> Author:
        return author_details(author_id, fields, **self.kwargs)

    def author_papers_batch(self, author_id: str, fields: str = None, offset: int = None, limit: int = None) -> Batch[Paper]:
        return author_papers(author_id, fields, offset, limit, **self.kwargs)

    def paper_search_batch(self, query: str, fields: str = None, offset: int = None, limit: int = None) -> Batch[Paper]:
        return paper_search(query, fields, offset, limit, **self.kwargs)

    def paper_details(self, paper_id: str, fields: str = None) -> Paper:
        return paper_details(paper_id, fields, **self.kwargs)

    def paper_authors_batch(self, paper_id: str, fields: str = None, offset: int = None, limit: int = None) -> Batch[Author]:
        return paper_authors(paper_id, fields, offset, limit, **self.kwargs)

    def paper_citations_batch(self, paper_id: str, fields: str = None, offset: int = None, limit: int = None) -> Batch[Citation]:
        return paper_citations(paper_id, fields, offset, limit, **self.kwargs)

    def paper_references_batch(self, paper_id: str, fields: str = None, offset: int = None, limit: int = None) -> Batch[Reference]:
        return paper_references(paper_id, fields, offset, limit, **self.kwargs)



if __name__ == "__main__":

    ss = SemanticScholarAPI()
    
    FIELDS_AUTHOR_STATS = "citationCount,paperCount,hIndex"

    result = ss.author_search("Jakob Havtorn", fields=FIELDS_AUTHOR_STATS)  # "citation_count,paper_count,homepage,h_index"

    # r = requests.get(PAPER_DETAILS.format(paper_id="649def34f8be52c8b66281af98ae884c09aef38b"), params={})
    # data = get_request(PAPER_DETAILS.format(paper_id="649def34f8be52c8b66281af98ae884c09aef38b"))
    # paper = Paper.from_dict(data)

    import IPython

    IPython.embed(using=False)

    # fields = ["name", "externalIds", "url", "aliases", "affiliations", "homepage", "paperCount", "citationCount", "hIndex", "papers"]
    #                                                                               'name,external_ids,url,aliases,affiliations,homepage,paper_count,citation_count,h_index,papers'
    # r = requests.get(AUTHOR_SEARCH, params={"query": "Diederik Kingma", "fields": "name,externalIds,url,aliases,affiliations,homepage,paperCount,citationCount,hIndex,papers"})
    # r = requests.get(AUTHOR_SEARCH, params={"query": "Diederik Kingma", "fields": fields})
    # author_batch = Batch.from_dict(r.json())

    # r = requests.get(PAPER_SEARCH, params={"query": "Auto-encoding variational bayes", "fields": ""})
