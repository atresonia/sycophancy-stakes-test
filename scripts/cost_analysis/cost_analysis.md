# Cost Analysis
## Essay Grading
### Token Sizes
1. `LLM_ESSAY_GRADE_SYSTEM_PROMPT`: 852 tokens
2. `USER_FRAMING_PREFIX`: 24 tokens
3. Average stakes variant: 12 tokens (mean  across 30 variants in `essay_grading_stakes.py`)
4. Average essay length: 421 tokens (mean from `persuasive_df.csv`)
5. Output per call: 30 tokens (grade + short justification estimate)

### Per-call Input Tokens
KB call = system prompt + essay = 852 + 421 = 1273 tokens
user_sycophancy call = system prompt + user_framing + sep (new line) + essay = 852 + 24 + 1 + 421 = 1298 tokens
stakes call = system prompt + stakes variant + sep (new line) + essay = 852 + 24 + 12 + 1 + 421 = 1310 tokens

#### Calls per essay
baseline = 2 batches * 10 probes -> 20 calls
user_sycophancy = 10 probes
stakes = 10 probes per level * 3 levels = 30 probes
total = 20 + 10 + 30 = 60 calls per essay

#### Total Input Tokens
KB = 1273 tokens * 60 calls = 25460 tokens
user_sycophancy = 1298 tokens * 60 calls = 12980 tokens
stakes = 1310 tokens * 30 calls = 39300 tokens
total = 25460 + 12980 + 39300 = 77740 tokens

#### Total Output Tokens
output = 30 tokens * 60 calls = 1800 tokens

#### Total Cost
N = 800 (conservative power analysis estimate)
Total input=77,740×800=62,192,000=62.19M
Total output=1,800×800=1,440,000=1.44M
#### OpenAI
gpt-4o: $2.50/M input, $10.00/M output
input cost = 62.19M * $2.50 / 1000000 = $155.475
output cost = 1.44M * $10.00 / 1000000 = $14.40
total cost = $155.475 + $14.40 = $169.88
N = 200 (power analysis estimate)
total cost $169.88 / 4 = $42.47

## MMLU Bet
### Token Sizes
1. `LLM_MMLU_BET_SYSTEM_PROMPT`: 47 tokens
2. `KB_USER_PROMPT`: 177 (mean of `kb_pilot_gemini.csv['prompt']`)
3. user_baseline user prompt: 231 (mean of `stakes_pilot_gemini.csv['user_baseline_prompt']`)
4. bet_baseline user prompt: 245 (mean of `stakes_pilot_gemini.csv['bet_baseline_prompt']`)
5. stakes user prompt: 249 (bet_baseline + 4 tokens for stakes-amount phrase)
6. output per call: 3 -> 1 token (letter) - using 3 for overestimate

### Per-call Input Tokens
1. KB = 47 + 177 = 224 tokens
2. user_baseline = 47 + 231 = 278 tokens
3. bet_baseline = 47 + 245 = 292 tokens
4. stakes = 47 + 249 = 296 tokens

#### Calls per question
1. KB: 10 probes
2. user_baseline: 1 call
3. bet_baseline: 1 call
4. stakes: 3 calls per level * 2 directions (incorrect/correct) * 10 variants = 60 calls
total calls per question = 10 + 1 + 1 + 60 = 72 calls

#### Total Input Tokens
KB = 224 tokens * 10 calls = 2240 tokens
user_baseline = 278 tokens * 1 call = 278 tokens
bet_baseline = 292 tokens * 1 call = 292 tokens
stakes = 296 tokens * 60 calls = 17760 tokens
total = 2240 + 278 + 292 + 17760 = 20570 tokens

#### Total Output Tokens
output = 3 tokens * 72 calls = 216 tokens

#### Total Cost
N = 300 (conservative power analysis estimate)
Total input=20,570×300=6,171,000=6.17M
Total output=216×300=64,800=0.065M
#### OpenAI
gpt-4o: $2.50/M input, $10.00/M output
input cost = 6.17M * $2.50 / 1000000 = $15.425
output cost = 0.065M * $10.00 / 1000000 = $0.65
total cost = $15.425 + $0.65 = $16.075

## ELEPHANT
### Token Sizes
1. AITA post prompt: 400 tokens (estimated mean)
2. Binary suffix (output only YTA/NTA): 9 tokens
3. Average stakes variant: 10.6 tokens (mean across 30 variants in elephant_base_experiment_flip.py)
4. output per call: 1 (YTA/NTA)

### Per-call Input Tokens
1. baseline call (post + binary suffix): 400 + 9 = 409 tokens
2. stakes call (post + stakes variant + sep (new line) + binary suffix): 400 + 10.6 + 1 + 9 = 420.6 tokens

#### Calls per question
1. moral_syc_baseline: 2 calls (original + flipped)
2. moral_syc_stakes("low", num_probes=10): 2 * 10 = 20 calls
3. moral_syc_stakes("medium", num_probes=10): 2 * 10 = 20 calls
4. moral_syc_stakes("high", num_probes=10): 2 * 10 = 20 calls

total calls per question = 2 + 20 + 20 + 20 = 62 calls

#### Total Tokens
input tokens = 409 tokens * 2 calls + 60 calls * 420.6 tokens = 26054 tokens
output tokens = 1 token * 62 calls = 62 tokens

#### Total Cost
N = 1000 (half of 2000 questions in elephant dataset)
Total input=26,054×1000=26,054,000=26.054M
Total output=62×1000=62,000=0.062M
#### OpenAI
gpt-4o: $2.50/M input, $10.00/M output
input cost = 26.054M * $2.50 / 1000000 = $65.135
output cost = 0.062M * $10.00 / 1000000 = $0.62
total cost = $65.135 + $0.62 = $65.75

