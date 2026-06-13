# Mentis: A Training-Free Mental World Model Baseline for Multiple-choice Situated Reasoning

Mentis is a training-free baseline for multiple-choice situated reasoning. Instead of asking an LLM to directly choose an answer from the question and options, Mentis explicitly simulates a Mental World Model (MWM): it parses the scene into a joint physical/mental state, renders the target agent's partial observation, simulates each candidate action, scores the resulting next states, and then selects the final action with a deterministic decision rule.

This repository contains the current implementation of the Mentis pipeline, evaluation scripts, ablations, and runnable sample data.

## Overview

Mentis follows the paper framing of a third-person world simulator with first-person target-agent observations. The main inference chain is:

```text
scene -> s_t
s_t + target_agent -> o_t
dataset options -> sampled_actions
sampled_actions -> candidate_actions
(a^phy_t, s^phy_t, s^ment_t) -> s^phy_{t+1}
(a^phy_t, a^ment_t, s^phy_t, s^ment_t) -> s^ment_{t+1}
s^phy_{t+1} + s^ment_{t+1} -> s_{t+1}
(s_t, o_t, action, s_{t+1}, question) -> score
score -> final_action
```

Key properties:

- Training-free: all reasoning is performed through configurable OpenAI-compatible LLM calls.
- Explicit state transition: each option is evaluated through predicted `s_{t+1}` rather than direct answer selection.
- Joint physical and mental state: `WorldState` contains both `physical_state` and `mental_state`.
- Partial observation: `o_t` represents what the target agent can see, hear, know, or infer.
- Deterministic final decision: scores are combined by code using configured weights and tie-break rules.
- Multimodal input support: text, image, and video stories are supported; video is converted into sampled image frames.

## Repository Structure

```text
mentis/
  clients/        OpenAI-compatible LLM client
  evaluation/     deterministic metrics, LLM judges, and evaluation CLI
  modules/        MWM pipeline modules
  prompts/        prompt builders
  utils/          JSON, media, tracing, runtime-input, and concurrency helpers
  config.py       configuration loading
  pipeline.py     end-to-end pipeline
  run.py          prediction CLI
  schema.py       Pydantic models, state templates, and schema normalization

configs/
  default.yaml

data/
  sample_input.jsonl
  sample_assets/

outputs/
```

## Installation

Create the recommended local environment:

```bash
conda create -n mentis python=3.11 -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate mentis
python -m pip install -r requirements.txt
```

Mentis currently supports OpenAI or OpenAI-compatible Responses API endpoints.

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
$env:OPENAI_BASE_URL="https://your-forwarder.example/v1"
```

On Linux/macOS:

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_BASE_URL="https://your-forwarder.example/v1"
```

`OPENAI_BASE_URL` is optional. Leave it unset or set it to an empty string to use the official OpenAI API. Mentis reads credentials from environment variables only; it does not load `.env`.

## Data Format

Mentis reads JSON or JSONL files. The included runnable sample is:

```text
data/sample_input.jsonl
data/sample_assets/
```

Each record follows this structure:

```json
{
  "sample_id": 1,
  "domain": "dormitory",
  "domain_category": "residential",
  "scene": "social_conflict",
  "num_of_characters": 2,
  "story": {
    "text": "story text ending at the current decision moment"
  },
  "target_agent": "Amy",
  "question": "What should Amy do next?",
  "options": [
    {
      "option_id": "A",
      "action_description": "..."
    }
  ],
  "golden_answer": {
    "current_state_s_t": {},
    "target_agent_observation_o_t": {
      "physical_observation": {
        "entity_and_attribute": {},
        "relations": {},
        "environment": {}
      },
      "mental_observation": {
        "mental_entity_and_attribute": {},
        "relations": {},
        "atmosphere_of_environment": ""
      }
    },
    "next_state_s_{t+1}": {},
    "score": {},
    "final_action": "C"
  }
}
```

`golden_answer` is used only for evaluation and explicit oracle ablations. Standard `full_mwm` runs do not feed gold fields into the model.

### Supported Story Modalities

```json
{"story": {"text": "continuous story text"}}
{"story": {"images": ["relative/or/absolute/image_path.png"]}}
{"story": {"video": "relative/or/absolute/video_path.mp4"}}
```

At runtime, media paths are resolved into internal `_media_scene.image_paths` or `_media_scene.video_path` fields. These runtime fields are not part of the dataset contract.

Video is not sent as `input_video`. Mentis samples local video frames and sends them as `input_image` items. If no local decoder is available, the sample fails early with diagnostics.

## Quick Start

Run Mentis on the sample JSONL:

```bash
python -m mentis.run
```

Run one or more selected samples:

```bash
python -m mentis.run --sample-id 1 51 76
```

By default, Mentis reads `data/sample_input.jsonl`, uses `configs/default.yaml`, and writes predictions under `outputs/`. The default output path is:

```text
outputs/gpt5_5_predictions.jsonl
```

When the requested output filename is a generic prediction filename, Mentis also prefixes it with the configured world model.

## Output Format

Each output record copies public input fields and appends `generated_results`. The prediction output does not include `golden_answer`.

```json
{
  "sample_id": 1,
  "generated_results": {
    "status": "success",
    "current_state_s_t": {
      "physical_state": {},
      "mental_state": {}
    },
    "target_agent_observation_o_t": {
      "physical_observation": {
        "entity_and_attribute": {},
        "relations": {},
        "environment": {}
      },
      "mental_observation": {
        "mental_entity_and_attribute": {},
        "relations": {},
        "atmosphere_of_environment": ""
      }
    },
    "candidate_actions": [
      {
        "option_id": "A",
        "raw_action_description": "...",
        "physical_action_description": "...",
        "mental_action_description": "..."
      }
    ],
    "next_state_s_{t+1}": {
      "A": {
        "physical_state": {},
        "mental_state": {}
      }
    },
    "score": {
      "A": {
        "mentally_consistent": 0.0,
        "physically_plausible": 0.0,
        "socially_appropriate": 0.0,
        "safety_legality_veto": false,
        "raw_value_score": 0.0,
        "overall_score": 0.0,
        "reasoning": "concise rationale"
      }
    },
    "final_action": "C",
    "decision_trace": {},
    "metadata": {}
  }
}
```

Failed samples are also written with `status: "failed"`, `error_type`, `error_message`, and any available metadata, so batch execution continues after per-sample failures.

## Evaluation

Run evaluation with deterministic metrics and the default LLM judge:

```bash
python -m mentis.evaluation.run \
  --pred outputs/gpt5_5_predictions.jsonl \
  --gold data/sample_input.jsonl \
  --output outputs/eval_report.json \
  --config configs/default.yaml
```

Run deterministic metrics only:

```bash
python -m mentis.evaluation.run \
  --pred outputs/gpt5_5_predictions.jsonl \
  --gold data/sample_input.jsonl \
  --output outputs/eval_report.json \
  --config configs/default.yaml \
  --skip-llm-judge
```

Deterministic metrics include:

- `final_action_accuracy`
- `failure_rate`
- score MAE
- next-state option coverage and schema diagnostics
- `per_sample`
- `per_task_type`

Schema completeness is a structural diagnostic, not a semantic quality score. The LLM judge evaluates semantic and causal quality for:

- `current_state_s_t`
- `target_agent_observation_o_t`
- `next_state_s_{t+1}`
- `score`

## Ablations

```bash
python -m mentis.run --input data/sample_input.jsonl --output outputs/full_mwm.jsonl --config configs/default.yaml --ablation full_mwm
python -m mentis.run --input data/sample_input.jsonl --output outputs/no_mental_information.jsonl --config configs/default.yaml --ablation no_mental_information
python -m mentis.run --input data/sample_input.jsonl --output outputs/no_physical_information.jsonl --config configs/default.yaml --ablation no_physical_information
python -m mentis.run --input data/sample_input.jsonl --output outputs/direct_answer_baseline.jsonl --config configs/default.yaml --ablation direct_answer_baseline
python -m mentis.run --input data/sample_input.jsonl --output outputs/oracle_state.jsonl --config configs/default.yaml --ablation oracle_state
python -m mentis.run --input data/sample_input.jsonl --output outputs/oracle_observation.jsonl --config configs/default.yaml --ablation oracle_observation
```

Supported ablations:

- `full_mwm`: the complete Mentis pipeline.
- `direct_answer_baseline`: direct answer baseline for comparison only.
- `no_mental_information`: hides mental state and mental observation from scoring/evaluation.
- `no_physical_information`: hides physical state and physical observation from scoring/evaluation.
- `oracle_state`: uses `golden_answer.current_state_s_t`.
- `oracle_observation`: uses `golden_answer.target_agent_observation_o_t`.

## Modules

| Module | Role |
| --- | --- |
| `StateParser` | Parses text/image/video scenes into `WorldState`. |
| `ObservationGenerator` | Generates target-agent partial observation `o_t`. |
| `TargetPseudoAgent` | Uses benchmark options as candidate actions; not a learned policy. |
| `ActionParser` | Splits each option into physical and mental action components. |
| `PhysicalTransitionModel` | Predicts `s^phy_{t+1}` for each candidate branch. |
| `MentalTransitionModel` | Predicts `s^ment_{t+1}` for each candidate branch. |
| `NextStateMerger` | Merges physical and mental transition outputs. |
| `ScoringModule` | Scores each branch from `s_t`, `o_t`, action, `s_{t+1}`, and question. |
| `DecisionModule` | Selects `final_action` with deterministic weights and tie-break rules. |
| `DirectAnswerBaseline` | Direct-answer comparison baseline. |

## Configuration

The default configuration is in `configs/default.yaml`:

```yaml
api:
  provider: "openai"
  base_url: ""
  timeout_seconds: 240
  max_retries: 4
  max_concurrent_requests: 1
  template_schema_max_retries: 1
  pydantic_schema_max_retries: 1
  retry_initial_delay_seconds: 2.0
  retry_max_delay_seconds: 60.0
models:
  parser_model: "gpt-5.5"
  world_model: "gpt-5.5"
  scoring_model: "gpt-5.5"
  direct_answer_model: "gpt-5.5"
  judge_model: "gpt-5.5"
generation:
  temperature: null
  max_output_tokens: 30000
  reasoning_effort: "xhigh"
  text_verbosity: "high"
video:
  enabled: true
  max_frames: 16
  frame_sampling: "uniform"
transition:
  max_concurrency: 6
  expected_action_count: 6
scoring:
  weights:
    mentally_consistent: 0.45
    physically_plausible: 0.35
    socially_appropriate: 0.2
  tie_break_order:
    - "physically_plausible"
    - "mentally_consistent"
    - "socially_appropriate"
    - "option_order"
evaluation:
  llm_judge_enabled: true
  judge_max_concurrency: 2
logging:
  save_raw_llm_outputs: true
  save_prompts: true
  output_dir: "outputs/logs"
```

`DecisionModule` does not call an LLM. It ranks by `overall_score` and then applies the configured `tie_break_order`.

## Logging

If logging is enabled, each run writes LLM traces to:

```text
outputs/logs/<run_id>/<sample_id>_llm_calls.jsonl
```

Each trace includes the prompt, raw output, parsed JSON, model, latency, token usage, request metadata, and warnings. Prediction files also summarize module metadata under `generated_results.metadata`.

## Citation

BibTeX will be added once the manuscript metadata is finalized.
