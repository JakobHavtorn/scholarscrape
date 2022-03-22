# ScholarScrape

## Installation
```bash
conda create -y -n scholarscrape python==3.10
conda activate scholarscrape
pip install --upgrade --editable .
```


## Run
You can look up a single author by inputting their name
```bash
python --authors "Firstname Middlename Lastname"
```

You can also input several authors and they will all be looked up.
```bash
python --authors "First person" "Second person"
```

The string search matching is done by the SemanticScholar API and given multiple matches, we select the author with the most citations.


## Outputs
> `sc`:   self citations (total number of citations where the person is an author of both the citing and cited works).

> `psc`:  pseudo-self citations (total number of citations where the person, or a historical co-author, is an author of both the citing and cited works).

> `s`:    self citation s-index computed as the h-index but on self-citations.

> `ps`:   pseudo-self citation ps-index computed as the h-index but on pseudo-self citations.


> `scr`:  self citation ratio (citations / SC).


> `pscr`: pseudo-self citation ratio (citations / PSC).

