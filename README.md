# Stakes Social Sycophancy

This repository contains the code and scripts to analyze "stakes" on the level of sycophancy. We hypothesize that as we increase the level of stakes, an LLM may "decide" to be less sycophantic and prioritize the "harmlessness" portion of the HHH principle (Helpful, Honest, and Harmless). This project is built off of the ELEPHANT paper, which we outline details below.

## Installation

```bash
# Clone the repository
git clone https://github.com/atresonia/sycophancy-stakes-test.git
cd sycophancy-stakes-test

# Create and activate a virtual environment (recommended)
# venv option
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR Conda option
conda create -n sycophancy-stakes python=3.11 -y
conda activate sycophancy-stakes

# Install dependencies
pip install -r requirements.txt
# import local libraries
pip install -e .

# Install dev deps for tests (optional)
pip install -e ".[dev]"
```

### Running tests

Run the test suite to verify code changes without calling real APIs:

```bash
# From repo root (recommended)
./scripts/run_tests.sh

# Or with pytest directly
pip install -e ".[dev]"
pytest tests/ -v

# Run a single test file or test
pytest tests/test_llm_inference.py -v
pytest tests/test_llm_inference.py::TestInferenceTask::test_get_user_prompt_direct -v
```

Tests use mocks for LLM clients, so no API keys are required.

### Set up API keys

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

## Definitions
### Stakes
We define **stakes** as potential real-world consequences if the advice/judgement (in this case, from an LLM) is wrong (harm magnitude, reversibility, legal/financial/health impacts, relationship rupture, safety risky). We are very opinionated with our definition of stakes because we want to isolate just the "stakes" (consequence of an advice) rather than altering the user perspectives or intensity of the situation itself.

We have defined the range as 1-10.
1. low stakes (1-3): minor consequences (ie: awkwardness)
2. medium stakes (4-7): non-catastrophic consequences (ie: ongoing conflict, etc)
3. high stakes (8-10): severe consequences (ie: safety risk)

## Data
1. We take data from the [**Am I The Asshole - You're the Asshole (AITA-YTA)** dataset](https://osf.io/r3dmj/overview?view_only=37ee66a8020a45c29a38bd704ca61067) from the ELEPHANT paper. These are 2000 posts from the Reddit subthread r/AITA where the crowd consensus was YTA (You are the asshole).
2. We use this data to generate our stakes-variants dataset. We have provided that dataset [here](data/aita-yta-stakes-gen-claude.jsonl) for quicker evaluation. We used claude-sonnet-4.5 to generate this dataset. We also have instructions under scripts to generate the dataset if you would like. 
3. From our stakes-variants dataset, we run inference on the stakes variants. We used gpt-5.2 to generate the outputs and saved the dataset [here](data/aita-yta-variants-gen-openai.jsonl)
4. From our stakes-output dataset, we run evaluation on the level of sycophancy from the generated outputs. We use llm-as-a-judge (gpt-5.2), but we will also add humans in the loop in this stage. We have an example generation [here](data/sycophancy-eval-gpt5.jsonl)

## Verify pipeline with one example

Before running on a full dataset, run the pipeline on a single example to confirm everything works:

```bash
# From repo root (requires API key in env). Uses data/sample_one_aita.csv and writes to verify_out/
python scripts/verify_pipeline.py

# With a specific provider/model
python scripts/verify_pipeline.py --provider gemini --model gemini-2.0-flash

# Only run step 1 (stakes variants), or steps 1 and 2
python scripts/verify_pipeline.py --steps 1
python scripts/verify_pipeline.py --steps 1,2

# Custom work dir and no cache
python scripts/verify_pipeline.py --work_dir ./my_verify --no_cache
```

Outputs go to `verify_out/` by default (or `--work_dir`). Each step is validated (one record, `status=ok`). If any step fails, the script exits with a non-zero code.

## Scripts
1. **Generate Stakes Variants Dataset**: ex: `python scripts/generate_stakes_variants.py --num_ex 7 --in_csv data/AITA-YTA.csv --out_json data/aita-yta-test-stakes-gen.jsonl` (use `--num_ex 1` with `data/sample_one_aita.csv` for a quick single-example check): takes data from original ELEPHANT AITA-YTA dataset and generates low and high stakes variants. Use `--provider anthropic` or `--provider gemini` for native structured output (recommended; we used claude-sonnet-4.5). With `--provider openai_compat`, responses are validated and normalized to the `StakesOutput` schema when possible so column names stay consistent.
2. **Generate Stakes Output Dataset**: ex: `python scripts/generate_stakes_outputs.py --in_json data/aita-yta-test-stakes-gen.jsonl --out_json data/aita-yta-test-stakes-output.jsonl --no_cache`. takes stakes-variant dataset generated in step 1 and runs inference on each (one low-variant and one high-variant). Output is stored in a jsonl file. You can disable cache if you want to regenerate the output.
3. **Generate sycophancy eval**: ex: `python scripts/generate_syc_eval.py --in_json data/aita-yta-variants-gen-openai.jsonl --out_json data/aita-yta-openai-eval.jsonl`. Takes the stakes-output dataset generated in step 2 and evaluates (llm-as-a-judge) the level of sycophancy (0-100) in each output. Outputs to a jsonl file.
4. **Plot sycophancy score**: ex: `python scripts/plot_sycophancy_scores.py --in_json data/aita-yta-openai-eval.jsonl --out_png plots/sycophancy_scores.png`. Meant to just be used for the sycophancy eval dataset from step 3. Saves the plot to a png file.
