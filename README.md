# LLMs as a Crowd

## Setup
Python 3.8, venv or conda recommended.

Setup with venv:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Entity Resolution

The list of configured experiments is in `config/entity_resolution.yaml`, which contains all the experiments necessary to reproduce our results in Sections 5.1 and 5.2. As an example, below is how to run the `er_basic` experiment:
```
export OPENAI_API_KEY=<API key>
./scripts/download_er_datasets.sh
python llm_crowd/entity_resolution/query.py er_basic
```

Then, to produce F1 scores and other metrics:
```
python llm_crowd/lib/analyze.py entity_resolution
```

See the `out` directory for the results.

## Joinability

All code necessary to reproduce our joinability experiments in Section 5.3 are in the folder `llm_crowd/joinability`.

First, download the ChemBL 21 database from here: https://chembl.gitbook.io/chembl-interface-documentation/downloads#chembl-database-release-dois

And extract its FKs to a csv file ``chembl_groundtruth.csv`` with schema: ``Input Table, FK Table, Input Column, FK Column``.
Then, extract each table from the database as a csv file to directory ``../chembl_csvs``.

To run the joinability experiment where we test the LLM's ability to determine whether two tables are joinable, first run 
```
python llm_crowd/generate_50_50.py
```
to generate the 100 ChemBL FKs, and 100 incorrect pairs of tables which comprise the dataset in our experiment. Then run

```
python llm_crowd/join_raw.py
```
To run the joinability experiment where we test the LLM's ability to postprocess join candidates returned by an existing system (Aurum), first run 
```
python llm_crowd/store_aurumjoins.py
```
to generate the joins found by Aurum over ChemBL. Then run

```
python llm_crowd/joinability/join_experiments.py
```
For both experiments, you can add the option ``--shots`` to add the few-shot examples described in the paper.

## Unionability
All code necessary to reproduce our joinability experiments in Section 5.3 are in the folder `llm_crowd/joinability`.

First, follow instructions to generate SANTOS results for ``santos_small`` and ``TUS`` datasets here: https://github.com/northeastern-datalab/santos

To run our unionability experiment, first run the script:
```
python llm_crowd/unionability/gather_santos.py
```
This will use SANTOS's saved output to store SANTOS's candidates. You can then run our unionability experiment in Section 5.3 by running:

```
python llm_crowd/unionability/union_experiments.py
```
We provide the options ``--dataset`` to run on either the ``santos_small`` or ``TUS`` dataset, and ``--shots`` for few-shot examples.


