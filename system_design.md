# Mentis 当前系统设计

本文档描述当前仓库中的 Mentis 实现。当前系统是一个 training-free、OpenAI Responses API based 的 Mental World Model baseline，用于多选式情境推理样本。核心约束是：系统必须先显式构建状态、观察、候选动作和状态转移，再评分并选择答案；非 direct-answer 消融模式不得直接根据题目和选项猜答案。

## 1. 目标

Mentis 的主流程强制执行下面的推理链：

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

其中：

- `s_t` 是当前联合世界状态，包含 `physical_state` 和 `mental_state`。
- `o_t` 是 target agent 的第一人称局部观察，包含 `physical_observation` 和 `mental_observation`。
- 每个多选项被视为一个候选动作分支。
- 每个分支都先预测下一状态 `s_{t+1}`，再进行价值评分。
- `final_action` 由代码中的 `DecisionModule` 根据分数和 tie-break 规则确定。

## 2. 输入数据契约

运行入口支持 JSON 或 JSONL，单条样本字段如下：

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

`golden_answer` 只用于 evaluation 和显式 oracle ablation。普通 `full_mwm` 运行不会把 gold 字段传给主模型。

### 2.1 story 模态

当前实现要求 state parser 输入中只选择一个有效 story 模态。优先级为：

1. `story.images`
2. `story.video`
3. `story.text`

支持形式：

```json
{"story": {"text": "..."}}
{"story": {"images": ["relative/or/absolute/image_path.png"]}}
{"story": {"video": "relative/or/absolute/video_path.mp4"}}
```

运行时会派生内部字段 `_media_scene`：

- `image_paths`: 已解析的图片路径列表。
- `video_path`: 已解析的视频路径。
- `modality`: `text`、`image` 或 `video`。

这些字段是运行时内部数据，不属于数据集输入契约。相对媒体路径会先按输入文件所在目录解析，再按当前工作目录解析。

## 3. 输出数据契约

每条输出会复制输入中的公开字段，并追加 `generated_results`。输出不会包含 `golden_answer`。

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

失败样本也会写出，`status` 为 `failed`，并带有 `error_type`、`error_message` 和已产生的 metadata。batch 不会因为单条失败而中断。

## 4. 核心 schema

当前 schema 契约定义在 `mentis/schema.py`，包含 Pydantic 模型、状态模板和归一化逻辑。

### 4.1 WorldState

```json
{
  "physical_state": {
    "entity_and_attribute": {},
    "relations": {},
    "environment": {}
  },
  "mental_state": {
    "mental_entity_and_attribute": {},
    "relations": {},
    "atmosphere_of_environment": ""
  }
}
```

实现允许 LLM 返回模板中的扩展内容，但顶层必须归一化为：

- `physical_state`
- `mental_state`

### 4.2 TargetObservation

```json
{
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
}
```

`o_t` 是 target agent 的局部观察，不是全局状态复制。prompt 要求它体现 POMDP 风格的信息边界：target agent 可以看到、听到、知道、推断到什么，就只能写什么。
结构上，`physical_observation` 复用 `PhysicalState`，`mental_observation` 复用 `MentalState`。因此 `TargetObservation` 是 `WorldState` 的 target-view 局部投影，而不是另一套松散 schema。

### 4.3 CandidateAction

每个数据集选项先由 `TargetPseudoAgent` 抽取为 `SampledAction`，再由 `ActionParser` 分解为：

```json
{
  "option_id": "A",
  "raw_action_description": "...",
  "physical_action_description": "physical carrier / execution / direct physical effect",
  "mental_action_description": "semantic / intentional / emotional / social component"
}
```

如果某一部分不存在，使用空字符串。实现会把 `none`、`null`、`n/a` 等归一化为空字符串。

### 4.4 ScoreResult

```json
{
  "option_id": "A",
  "mentally_consistent": 0.7,
  "physically_plausible": 0.5,
  "socially_appropriate": 0.8,
  "safety_legality_veto": false,
  "reasoning": "concise rationale",
  "raw_value_score": 0.66,
  "overall_score": 0.66
}
```

LLM 负责给出前三个维度、veto 和简短理由。代码负责计算：

```text
raw_value_score =
  weights.mentally_consistent * mentally_consistent
  + weights.physically_plausible * physically_plausible
  + weights.socially_appropriate * socially_appropriate

overall_score = 0.0 if safety_legality_veto else raw_value_score
```

如果 LLM 返回 1 到 5 分量表，schema validator 会换算到 0 到 1。

## 5. 模块职责

当前主流程在 `mentis/pipeline.py` 的 `MentisPipeline` 中编排。

### 5.1 StateParser

文件：`mentis/modules/state_parser.py`

职责：

- 接收单模态 story 输入和 media scene。
- 调用 `models.parser_model`。
- 输出 `WorldState`。

### 5.2 ObservationGenerator

文件：`mentis/modules/observation_generator.py`

职责：

- 接收 `WorldState` 和 `target_agent`。
- 调用 `models.world_model`。
- 输出 target agent 的 `TargetObservation`。

### 5.3 TargetPseudoAgent

文件：`mentis/modules/target_pseudo_agent.py`

职责：

- 把 benchmark 的多选项作为 target agent 候选动作代理。
- 不是真实 policy，不根据 `o_t` 生成新动作。
- 默认期望 6 个动作；数量不符时只写 warning，不中断。

### 5.4 ActionParser

文件：`mentis/modules/action_parser.py`

职责：

- 并行处理所有 sampled actions。
- 调用 `models.parser_model`。
- 输出 `CandidateAction` 列表。

### 5.5 PhysicalTransitionModel

文件：`mentis/modules/physical_transition.py`

职责：

- 输入 `physical_state`、`mental_state` 和当前选项的 `physical_action_description`。
- 调用 `models.world_model`。
- 输出该分支的 `PhysicalState`。

### 5.6 MentalTransitionModel

文件：`mentis/modules/mental_transition.py`

职责：

- 输入 `physical_state`、`mental_state` 和完整 `CandidateAction`。
- 调用 `models.world_model`。
- 输出该分支的 `MentalState`。

### 5.7 NextStateMerger

文件：`mentis/modules/next_state_merger.py`

职责：

- 将同一选项分支的 `PhysicalState` 和 `MentalState` 合并为完整 `WorldState`。

### 5.8 ScoringModule

文件：`mentis/modules/scoring.py`

职责：

- 接收 `s_t`、`o_t`、`question`、`target_agent`、`CandidateAction`、`s_{t+1}`。
- 根据 ablation policy 决定是否隐藏 mental 或 physical 信息。
- 调用 `models.scoring_model` 得到评分维度和 veto。
- 由代码计算 `raw_value_score` 和 `overall_score`。

评分必须基于状态转移结果，不能直接让模型跳过 `s_{t+1}` 做选择题。

### 5.9 DecisionModule

文件：`mentis/modules/decision.py`

职责：

- 不调用 LLM。
- 按 `overall_score` 排序。
- 根据 `scoring.tie_break_order` 做确定性 tie-break。
- 输出 `final_action` 和 `decision_trace.ranked_options`。

默认 tie-break 顺序：

```text
physically_plausible
mentally_consistent
socially_appropriate
option_order
```

`option_order` 使用较早的选项作为最后兜底。

### 5.10 DirectAnswerBaseline

文件：`mentis/modules/direct_answer_baseline.py`

职责：

- 只在 `--ablation direct_answer_baseline` 时启用。
- 直接把 scene、question、options 交给 `models.direct_answer_model`。
- 输出 `final_action`。

这个模式是对照实验，不代表主系统设计。

## 6. 并发模型

当前实现有三层并发控制：

- `ActionParser` 对所有候选动作并行解析。
- `MentisPipeline._run_branches()` 对候选动作分支做 bounded concurrency，受 `transition.max_concurrency` 控制。
- 每个分支内部，physical transition 和 mental transition 使用 `asyncio.gather()` 并行执行。

OpenAI 客户端还有全局请求并发限制：

```yaml
api:
  max_concurrent_requests: 1
```

因此真实 API 并发上限同时受 pipeline 分支并发和 client semaphore 控制。

## 7. Prompt 组织

Prompt builders 位于 `mentis/prompts/` 包中，并通过 `mentis/prompts/__init__.py` 导出：

- `build_state_parser_prompt`
- `build_observation_generation_prompt`
- `build_action_parser_prompt`
- `build_physical_transition_prompt`
- `build_mental_transition_prompt`
- `build_scoring_prompt`
- `build_direct_answer_baseline_prompt`
- `build_state_judge_prompt`
- `build_observation_judge_prompt`
- `build_next_state_judge_prompt`
- `build_score_judge_prompt`

注意：当前实现不是单文件 `mentis/prompts.py`，而是 `mentis/prompts/` 包。

Prompt 的共同要求：

- 明确输入、任务和 JSON 输出 schema。
- 不输出 markdown。
- 不输出长篇解释。
- 所有模块输出都必须能被 JSON/Pydantic schema 校验。
- scoring prompt 必须基于 `s_t -> action -> s_{t+1}` 的模拟结果。

## 8. LLM 客户端与 API

文件：`mentis/clients/openai_client.py`

当前只支持：

```yaml
api:
  provider: "openai"
```

客户端使用 OpenAI Responses API，并支持：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- text input
- image input
- video-to-frames input
- JSON 修复和抽取
- Pydantic/schema 校验
- retry/backoff
- timeout
- token usage、latency、model、request metadata 记录

`OPENAI_BASE_URL` 环境变量会覆盖 `configs/default.yaml` 中的 `api.base_url`。

### 8.1 视频输入

系统不会发送 `input_video`。视频会在本地抽帧，再作为多个 `input_image` 发送给 Responses API。

抽帧优先尝试：

1. PIL animated image
2. OpenCV
3. imageio
4. ffmpeg

如果本地无法解码视频，会 fail early，并给出 decoder diagnostics。

## 9. 默认配置

当前默认配置在 `configs/default.yaml`：

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

## 10. 运行命令

普通 batch 运行：

```bash
python -m mentis.run
```

默认输入为 `data/sample_input.jsonl`，默认配置为 `configs/default.yaml`，默认输出目录为 `outputs/`。因此 `gpt-5.5` 会写为：

```text
outputs/predict_full_mwm_gpt5_5_all_20260614_predictions.jsonl
```

如果输出参数是目录，系统会用和 log 文件夹一致的 readable run id 命名 prediction 文件，并追加 `_predictions.jsonl`。如果显式输出名是通用的 `prediction.json`、`prediction.jsonl`、`predictions.json`、`predictions.jsonl`，或旧式 `{world_model}_predictions.jsonl`，系统同样会改写为 run-id 文件名。

单样本或多样本筛选运行：

```bash
python -m mentis.run --sample-id 1 51 76
```

评估：

```bash
python -m mentis.evaluation.run --pred outputs/predict_full_mwm_gpt5_5_samples-1-51-76_20260614_predictions.jsonl --gold data/sample_input.jsonl --config configs/default.yaml
```

只跑确定性指标，不调用 LLM judge：

```bash
python -m mentis.evaluation.run --pred outputs/predict_full_mwm_gpt5_5_samples-1-51-76_20260614_predictions.jsonl --gold data/sample_input.jsonl --config configs/default.yaml --skip-llm-judge
```

## 11. Evaluation

评估入口：`mentis/evaluation/run.py`

确定性指标在 `mentis/evaluation/metrics.py`：

- `num_samples`
- `final_action_accuracy`
- `failure_rate`
- `score_alignment.mental_mae`
- `score_alignment.physical_mae`
- `score_alignment.social_mae`
- `next_state` option coverage/schema validity diagnostics
- `per_sample`
- `per_task_type`

确定性 schema 指标只检查结构存在和分支覆盖，不代表语义质量。

LLM judge 在 `mentis/evaluation/judge.py`，默认启用，评估：

- `current_state_s_t`
- `target_agent_observation_o_t`
- `next_state_s_{t+1}`，逐 option judge
- `score`

judge 报告会给出：

- `state_physical_judge_score`
- `state_mental_judge_score`
- `observation_physical_judge_score`
- `observation_mental_judge_score`
- `next_state_physical_judge_score`
- `next_state_mental_judge_score`
- `transition_reasonableness_score`
- `coupling_validity_score`

这些分数用于语义、因果和耦合质量分析，不替代 `final_action_accuracy`。

## 12. Ablations

当前 ablation policy 定义在 `mentis/policies/ablation.py`：

- `full_mwm`: 完整 Mentis 主流程。
- `direct_answer_baseline`: 直接回答 baseline。
- `no_mental_information`: scoring/evaluator 输入隐藏 mental state 和 mental observation。
- `no_physical_information`: scoring/evaluator 输入隐藏 physical state 和 physical observation。
- `oracle_state`: 使用 `golden_answer.current_state_s_t` 作为当前状态。
- `oracle_observation`: 使用 `golden_answer.target_agent_observation_o_t` 作为 target observation。

示例：

```bash
python -m mentis.run --input data/sample_input.jsonl --output outputs/no_mental_information.jsonl --config configs/default.yaml --ablation no_mental_information
python -m mentis.run --input data/sample_input.jsonl --output outputs/no_physical_information.jsonl --config configs/default.yaml --ablation no_physical_information
python -m mentis.run --input data/sample_input.jsonl --output outputs/direct_answer_baseline.jsonl --config configs/default.yaml --ablation direct_answer_baseline
python -m mentis.run --input data/sample_input.jsonl --output outputs/oracle_state.jsonl --config configs/default.yaml --ablation oracle_state
python -m mentis.run --input data/sample_input.jsonl --output outputs/oracle_observation.jsonl --config configs/default.yaml --ablation oracle_observation
```

Oracle ablation 如果缺少所需 gold 字段，会返回失败样本。

## 13. 日志与 metadata

每次 prediction 或 evaluation judge 初始化会创建一个可读 `run_id`。如果日志开启，LLM 调用写入：

```text
outputs/logs/<readable_run_id>/<sample_id>_llm_calls.jsonl
```

Prediction run id 形如：

```text
predict_<ablation>_<world_model>_<sample_scope>_<YYYYMMDD>
```

Evaluation judge run id 形如：

```text
eval_<ablation>_judge_<judge_model>_<sample_scope>_<YYYYMMDD>
```

示例：

```text
outputs/logs/predict_full_mwm_gpt5_5_samples-1-51-76_20260614/
outputs/logs/eval_full_mwm_judge_gpt5_5_all_20260614/
```

如果跑全部样本，`sample_scope` 使用 `all`；如果只跑部分样本，使用 `samples-1-51-76` 这类短标签。每个 log 目录还会写入 `run_manifest.json`，记录 run type、模型、样本范围、输入/输出路径、config 路径和时间戳。

每条日志包含：

- `sample_id`
- `task`
- `prompt_hash`
- `prompt`
- `raw_output`
- `parsed`
- `metadata.model`
- `metadata.latency_ms`
- `metadata.token_usage`
- `metadata.warnings`
- `metadata.request`

最终输出中的 `generated_results.metadata` 汇总：

- `ablation`
- `run_id`
- `module_metadata`
- `models`
- `llm_call_count`
- `latency_ms`
- `token_usage`
- `requests`
- `warnings`

## 14. 当前工程结构

```text
mentis/
  __init__.py
  config.py
  pipeline.py
  run.py
  schema.py

  clients/
    __init__.py
    base.py
    openai_client.py

  evaluation/
    judge.py
    metrics.py
    package_builder.py
    run.py

  modules/
    action_parser.py
    base.py
    decision.py
    direct_answer_baseline.py
    mental_transition.py
    next_state_merger.py
    observation_generator.py
    physical_transition.py
    scoring.py
    state_parser.py
    target_pseudo_agent.py

  policies/
    ablation.py

  prompts/
    __init__.py
    action.py
    common.py
    direct_answer.py
    judge.py
    observation.py
    scoring.py
    state.py
    transition.py

  utils/
    concurrency_utils.py
    json_utils.py
    media_utils.py
    runtime_input.py
    tracing.py

configs/
  default.yaml

data/
  sample_input.jsonl
  sample_assets/

outputs/

README.md
requirements.txt
system_design.md
```

## 15. 当前实现边界

- 只支持 OpenAI 或 OpenAI-compatible Responses API。
- 主流程使用数据集 options 作为候选动作集合，不实现真实 target-agent action generation policy。
- `full_mwm` 不读取 gold；gold 仅用于 evaluation 和 oracle ablation。
- 视频通过本地抽帧转成 `input_image`，不使用直接 `input_video`。
- 状态和观察最终都以 JSON 文本结构表达，不生成图像或视频状态。
- `DecisionModule` 是确定性代码模块，不是 LLM。
- schema presence 不是语义质量；语义/因果/耦合质量主要靠 LLM judge 分析。

## 16. 最重要的实现原则

Mentis 的核心不是让 LLM 做选择题，而是把多选题拆成可检查的 world-modeling 分支：

```text
先模拟世界状态和状态转移，再评价动作。
```

因此，除 `direct_answer_baseline` 外，任何实现修改都应保持以下顺序：

1. 解析当前 scene 得到 `s_t`。
2. 从 target agent 视角生成 `o_t`。
3. 把每个 option 解析为 physical/mental action。
4. 对每个 option 预测 `s_{t+1}`。
5. 基于 `s_t`、`o_t`、action、`s_{t+1}` 和 question 评分。
6. 用确定性决策模块选择 `final_action`。

如果系统绕过状态转移，直接根据 question 和 options 选择答案，就不符合 Mentis 主系统目标。
