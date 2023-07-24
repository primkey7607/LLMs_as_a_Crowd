# LLMs_as_a_Crowd
Still a work-in-progress. The idea is to share as much code as possible in libs, and any task-specific code can go in a folder like entity_resolution.

Notice the experiments_conf.yaml file under entity_resolution. The idea is to have a script that takes in an experiment name, and then looks at this config file to figure out what prompt templates / parameters to use for that experiment - I will have this done soon.

In the meantime, you can start on adding subclasses to PromptTemplate for the other tasks.
