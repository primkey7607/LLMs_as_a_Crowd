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

The list of configured experiments is in `config/entity_resolution.yaml`. Below is how to run the `er_basic` experiment:
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
