"""
https://www.semanticscholar.org/product/api

https://api.semanticscholar.org/graph/v1#tag/paper

https://blog.allenai.org/building-a-better-search-engine-for-semantic-scholar-ea23a0b661e7
"""


from typing import Dict
import requests


API_URL = "https://api.semanticscholar.org/graph/v1/"

PAPER_SEARCH = API_URL + "paper/search"
PAPER_DETAILS = API_URL + "paper/{paper_id}"
PAPER_AUTHORS = API_URL + "paper/{paper_id}/authors"
PAPER_CITATIONS = API_URL + "paper/{paper_id}/citations"
PAPER_REFERENCES = API_URL + "paper/{paper_id}/references"

PAPER_ENDPOINTS = dict(
    search=PAPER_SEARCH,
    details=PAPER_DETAILS,
    authors=PAPER_AUTHORS,
    citations=PAPER_CITATIONS,
    references=PAPER_REFERENCES,
)

AUTHOR_SEARCH = API_URL + "author/search"  # Search for authors by name return
AUTHOR_DETAILS = API_URL + "author/{author_id}"  # Returns details about an author
AUTHOR_PAPERS = API_URL + "author/{author_id}/papers"  # Returns the papers of an author in batches.

AUTHOR_ENDPOINTS = dict(
    search=AUTHOR_SEARCH,
    details=AUTHOR_DETAILS,
    papers=AUTHOR_PAPERS,
)


def make_json(offset: int, limit: int, fields: Dict[str, str]):
    kvs = [("offset", offset), ("limit", limit), ("fields", fields)]
    return {k: v for k, v in kvs if v is not None}


def author_query(author_id: str, endpoint: str, fields: Dict[str, str], limit: int = None):
    """
    """
    if not endpoint in AUTHOR_ENDPOINTS:
        raise ValueError(f"Invalid endpoint: {endpoint}")

    json = dict(offset=offset, fields=fields, limit=limit)
    response = requests.post(API_URL, json=json)
    return response.json()


def author_search(query: str, offset: int, limit: int, fields: Dict[str, str]) -> Dict[str, str]:
    json = make_json(offset, limit, fields)
    json.update(query=query)
    response = requests.post(AUTHOR_SEARCH, json=json)
    return response.json()


def author_details(author_id: str, fields: Dict[str, str]) -> Dict[str, str]:
    json = make_json(offset, limit, fields)
    json.update(query=query)
    response = requests.post(AUTHOR_SEARCH, json=json)
    return response.json()


def author_papers(author_id: str, offset: int, limit: int, fields: Dict[str, str]) -> Dict[str, str]:
    json = make_json(offset, limit, fields)
    json.update(query=query)
    response = requests.post(AUTHOR_SEARCH, json=json)
    return response.json()



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
    if not endpoint in PAPER_ENDPOINTS:
        raise ValueError(f"Invalid endpoint: {endpoint}")

    json = dict(fields=fields, limit=limit)
    response = requests.post(API_URL, json=json)
    return response.json()


def paper_search(endpoint: str, fields: dict = None, limit: int = None):
    """
    endpoint (str): A endpoint string.
    fields (dict): A dictionary of fields to return.
    limit (int): The maximum number of results to return.
    """
    json = dict(fields=fields, limit=limit)
    requests.post(API_URL, json=json)




if __name__ == "__main__":
    import IPython
    IPython.embed(using=False)
    
    
    

