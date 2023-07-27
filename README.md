# LLMs_as_a_Crowd

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


Still a work-in-progress. The idea is to share as much code as possible in libs, and any task-specific code can go in a folder like entity_resolution.

Notice config/entity_resolution.yaml. The idea is to have a script that takes in an experiment name, and then looks at this config file to figure out what prompt templates / parameters to use for that experiment - I will have this done soon.

In the meantime, you can start on adding subclasses to PromptTemplate for the other tasks.

Most of the downstream stuff - combining the many csvs into one, calculating crowd methods - will be shared code.
