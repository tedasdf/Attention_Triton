- Build and scale training pipelines for large language models and other architectures
- Triton
- build out mutli attention variant with triton or pytorch 
- CI/CD pipeline 
- MLFlow or wandb for observfaibility into training and inference
- optimise 



Syntehtic data yippe :?

2. Versioning the "Recipe," Not Just the Data
With synthetic data, the most important thing to version isn't just the .jsonl file—it's the Seed and the Prompt.

In your DVC project, you should track:

seeds/math_basics.csv: Small human-written examples.

prompts/generator_v1.txt: The system prompt used to tell the "Teacher" how to write the textbooks.

data/synthetic_corpus.jsonl.dvc: The massive output file (tracked in S3).


. Implementing the "Filter" Stage
The biggest risk with synthetic data in 2026 is "Model Collapse" (where the model learns the teacher's mistakes and amplifies them). To prevent this, your pipeline needs a Judge