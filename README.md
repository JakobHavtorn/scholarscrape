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
python --authors "Firstname Middlename Lastname""
```

You can also input several authors and they will all be looked up.
```bash
python --authors "First person" "Second person"
```

The string search matching is done by the SemanticScholar API and given multiple matches, we select the author with the most citations.
