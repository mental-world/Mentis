# full_MWM 的函数调用树

本文档按当前代码中的 `--ablation full_mwm` 主路径整理函数调用关系。它只描述完整 Mental World Model pipeline，不包含 `direct_answer_baseline`、`oracle_state`、`oracle_observation` 的旁路执行逻辑。

## 1. CLI 入口

```text
python -m mentis.run
```

默认参数等价于读取 `data/sample_input.jsonl`、使用 `configs/default.yaml`、运行 `--ablation full_mwm`，并写入 `outputs/{world_model}_predictions.jsonl`。

```text
mentis/run.py
└─ main()
   └─ build_arg_parser().parse_args()
   └─ asyncio.run(run_async(args))
      └─ load_config(args.config)
      └─ MentisPipeline(config)
      └─ read_records(args.input)
      └─ sample-id filter
      └─ resolve_output_path(args.output, config.models.world_model)
      └─ StreamingRecordWriter(...)
         └─ for record in records
            └─ pipeline.run_record(
                 record,
                 ablation="full_mwm",
                 input_base_dir=input_base_dir,
               )
            └─ writer.write(output)
```

## 2. Pipeline 初始化

```text
mentis/pipeline.py
└─ MentisPipeline.__init__(config)
   └─ create_llm_client(config)
      └─ OpenAIResponsesClient(config)
   └─ RunLogger(...)
   └─ StateParser(client, config, logger)
   └─ ObservationGenerator(client, config, logger)
   └─ TargetPseudoAgent(config.transition.expected_action_count)
   └─ ActionParser(client, config, logger)
   └─ PhysicalTransitionModel(client, config, logger)
   └─ MentalTransitionModel(client, config, logger)
   └─ NextStateMerger()
   └─ ScoringModule(client, config, logger)
   └─ DecisionModule(config)
   └─ DirectAnswerBaseline(client, config, logger)
```

`DirectAnswerBaseline` 会被初始化，但 `full_mwm` 不会调用它。

## 3. 单条样本主调用树

```text
MentisPipeline.run_record(record, ablation="full_mwm", input_base_dir)
└─ normalize_record(record)
└─ _public_output_record(record)
└─ get_ablation_policy("full_mwm")
   └─ AblationPolicy(name="full_mwm")
└─ SampleInput.model_validate(sample_record)
└─ runtime_sample_from_input(sample, media_base_dir)
   └─ scene_from_story(story)
      └─ normalize_scene_dict(...)
      └─ image_paths_from_story(...)
      └─ modality_from_story(...)
   └─ resolve_scene_media_paths(...), when input_base_dir is set
└─ RunContext.from_sample(runtime_sample, media_base_dir)
└─ _run_sample(runtime_sample, context, policy)
└─ output_record["generated_results"] = generated.as_dict()
└─ return output_record
```

`full_mwm` 的 policy 字段均为默认值：

```text
direct_answer = False
use_oracle_state = False
use_oracle_observation = False
hide_mental_from_evaluator = False
hide_physical_from_evaluator = False
```

因此 `_run_sample()` 会进入完整 MWM 主链路。

## 4. full_mwm 主链路

```text
MentisPipeline._run_sample(sample, context, policy)
└─ GeneratedResults(status="failed")
└─ _get_world_state(sample, context, policy)
   └─ build_state_parser_input(sample)
      └─ _story_modality_item(story)
      └─ _state_media_scene(sample["_media_scene"], story_key)
   └─ StateParser.parse(sample_id, state_input, media_scene)
      └─ build_state_parser_prompt(state_input)
      └─ ModuleBase.call_llm(
           task="state_parser",
           model=config.models.parser_model,
           context={"media_scene": media_scene},
           schema=WorldState,
         )
      └─ WorldState.model_validate(response.parsed)
      └─ response_metadata(response)
   └─ result.current_state_s_t = world_state.as_state_dict()
└─ _get_observation(sample, context, policy, world_state)
   └─ ObservationGenerator.generate(sample_id, world_state, target_agent)
      └─ world_state.as_state_dict()
      └─ build_observation_generation_prompt(state, target_agent)
      └─ ModuleBase.call_llm(
           task="observation_generation",
           model=config.models.world_model,
           context={},
           schema=TargetObservation,
         )
      └─ TargetObservation.model_validate(response.parsed)
      └─ response_metadata(response)
   └─ normalize_target_observation(observation.as_observation_dict())
   └─ result.target_agent_observation_o_t = normalized observation
└─ options = [OptionInput.model_validate(opt) for opt in sample["options"]]
└─ TargetPseudoAgent.sample_actions(options)
   └─ SampledAction(option_id, action_description), one per option
   └─ warning if option count != expected_action_count
└─ ActionParser.parse(sample_id, sampled_actions)
   └─ gather_bounded(sampled_actions, _parse_one, max_concurrency=len(sampled_actions))
      └─ ActionParser._parse_one(sample_id, sampled_action)
         └─ build_action_parser_prompt(sampled_action.action_description)
         └─ ModuleBase.call_llm(
              task="action_parser",
              model=config.models.parser_model,
              context={},
              schema=ActionDecomposition,
            )
         └─ ActionDecomposition.model_validate(response.parsed)
         └─ CandidateAction(...)
         └─ response_metadata(response)
   └─ merge_metadata(per-action metadata)
   └─ result.candidate_actions = [action.as_dict() for action in actions]
└─ _run_branches(context, world_state, observation, actions, policy)
└─ result.next_state_s_t1 = _next_states_from_branches(branch_outputs)
└─ result.score = _score_table_from_branches(branch_outputs)
└─ _failed_branches(branch_outputs)
   └─ if any branch failed: return failed GeneratedResults
└─ _scores_from_branches(branch_outputs)
└─ DecisionModule.decide(scores)
   └─ sorted(scores, key=DecisionModule._sort_key)
      └─ _tie_break_value(score, item)
      └─ _option_order(option_id)
   └─ return final_action, decision_trace
└─ result.status = "success"
└─ result.final_action = final_action
└─ result.decision_trace = decision_trace
└─ result.metadata = _metadata(policy, module_meta, branch_outputs)
└─ return result
```

## 5. 每个候选动作分支

`_run_branches()` 对所有候选动作分支做 bounded concurrency。

```text
MentisPipeline._run_branches(...)
└─ gather_bounded(
     actions,
     worker,
     max(1, min(len(actions), config.transition.max_concurrency)),
   )
   └─ worker(action)
      └─ _run_one_branch(context, world_state, observation, action, policy)
```

单个分支内部，physical transition 和 mental transition 并行执行：

```text
MentisPipeline._run_one_branch(context, world_state, observation, action, policy)
└─ BranchResult(action)
└─ asyncio.gather(
     PhysicalTransitionModel.predict(sample_id, world_state, action),
     MentalTransitionModel.predict(sample_id, world_state, action),
     return_exceptions=True,
   )
```

### 5.1 PhysicalTransitionModel

```text
PhysicalTransitionModel.predict(sample_id, world_state, action)
└─ physical_state = world_state.physical_state.as_dict()
└─ mental_state = world_state.mental_state.as_dict()
└─ physical_action = action.physical_action_description
└─ build_physical_transition_prompt(
     physical_state,
     mental_state,
     physical_action,
   )
└─ ModuleBase.call_llm(
     task="physical_transition",
     model=config.models.world_model,
     context={
       "physical_state": physical_state,
       "mental_state": mental_state,
       "physical_action": physical_action,
     },
     schema=PhysicalState,
   )
└─ PhysicalState.model_validate(response.parsed)
└─ response_metadata(response)
```

### 5.2 MentalTransitionModel

```text
MentalTransitionModel.predict(sample_id, world_state, action)
└─ physical_state = world_state.physical_state.as_dict()
└─ mental_state = world_state.mental_state.as_dict()
└─ action_dict = action.as_dict()
└─ build_mental_transition_prompt(
     physical_state,
     mental_state,
     action_dict,
   )
└─ ModuleBase.call_llm(
     task="mental_transition",
     model=config.models.world_model,
     context={
       "physical_state": physical_state,
       "mental_state": mental_state,
       "action": action_dict,
     },
     schema=MentalState,
   )
└─ MentalState.model_validate(response.parsed)
└─ response_metadata(response)
```

### 5.3 合并下一状态

```text
if physical or mental transition failed:
└─ BranchResult.mark_failed(exc)
└─ return branch

NextStateMerger.merge(physical, mental)
└─ WorldState(physical_state=physical, mental_state=mental)
└─ branch.next_state = next_state
```

### 5.4 ScoringModule

```text
ScoringModule.score(
  sample_id,
  world_state,
  observation,
  question,
  target_agent,
  action,
  next_state,
  ablation=policy,
)
└─ policy = full_mwm AblationPolicy
└─ world_state.as_state_dict()
└─ normalize_target_observation(observation.as_observation_dict())
└─ next_state.as_state_dict()
└─ policy.state_for_evaluator(...)
   └─ full_mwm: no field hidden
└─ policy.observation_for_evaluator(...)
   └─ full_mwm: no field hidden
└─ build_scoring_prompt(
     state_for_eval,
     observation_for_eval,
     question,
     target_agent,
     action.as_dict(),
     next_state.as_state_dict(),
   )
└─ ModuleBase.call_llm(
     task="scoring",
     model=config.models.scoring_model,
     context={
       "state": state_for_eval,
       "observation": observation_for_eval,
       "action": action.as_dict(),
       "next_state": next_state.as_state_dict(),
     },
     schema=ScoreResult,
   )
└─ ScoreResult.model_validate(response.parsed)
└─ raw_value_score(score)
└─ weighted_score(score)
└─ result.raw_value_score = weighted score
└─ result.overall_score = 0.0 if safety_legality_veto else weighted score
└─ response_metadata(response)
```

## 6. LLM 公共调用链

所有使用 LLM 的模块都经过 `ModuleBase.call_llm()`，再进入 OpenAI Responses client。

```text
ModuleBase.call_llm(...)
└─ client.complete_json(...)
   └─ OpenAIResponsesClient.complete_json(...)
      └─ _complete_with_retry(model, prompt, context, schema)
         └─ _call_with_limit(model, prompt, context)
            └─ asyncio.Semaphore(config.api.max_concurrent_requests)
            └─ asyncio.to_thread(_responses_call, model, prompt, context)
               └─ _client_kwargs()
                  └─ timeout
                  └─ base_url if config.api.base_url / OPENAI_BASE_URL is set
               └─ _input_payload(prompt, context)
                  └─ text prompt, or
                  └─ scene_to_openai_content(...) for image/video input
               └─ OpenAI(**client_kwargs)
               └─ client.responses.create(...)
         └─ _extract_non_empty_json(raw)
            └─ repair_json_output(raw)
            └─ extract_json_object(raw), fallback
         └─ _validate_raw_json(parsed, schema)
            └─ validate_raw_template_json(schema, parsed)
            └─ schema.model_validate(parsed)
         └─ _fill_and_validate(parsed, schema)
            └─ normalize_for_schema(schema, parsed)
            └─ schema.model_validate(...).model_dump(mode="json")
      └─ LLMResponse(parsed, raw_text, model, latency_ms, token_usage, request_metadata)
└─ RunLogger.log_llm_call(...), if logging is enabled
└─ return LLMResponse
```

## 7. 输出汇总调用

```text
_next_states_from_branches(branch_outputs)
└─ {option_id: branch.next_state.as_state_dict()}

_score_table_from_branches(branch_outputs)
└─ {option_id: branch.score.as_score_dict()}

_metadata(policy, module_meta, branch_outputs)
└─ _metadata_field_summary(..., "model")
└─ _metadata_field_summary(..., "latency_ms")
└─ _metadata_field_summary(..., "token_usage")
└─ sum_token_usage(...)
└─ _metadata_call_count(...)
└─ _metadata_warnings(...)
└─ generated_results.metadata

GeneratedResults.as_dict()
└─ model_dump(mode="json", by_alias=True)
└─ emits external key next_state_s_{t+1}
```

最终输出结构：

```text
output_record
└─ public input fields
└─ generated_results
   └─ status
   └─ current_state_s_t
   └─ target_agent_observation_o_t
   └─ candidate_actions
   └─ next_state_s_{t+1}
   └─ score
   └─ final_action
   └─ decision_trace
   └─ metadata
```

## 8. full_mwm 不会调用的分支

在 `full_mwm` 中，下面分支不会触发：

```text
policy.direct_answer == False
└─ 不调用 _run_direct_baseline()
└─ 不调用 DirectAnswerBaseline.answer()

policy.use_oracle_state == False
└─ _get_world_state() 不读取 golden_answer.current_state_s_t
└─ 调用 StateParser.parse()

policy.use_oracle_observation == False
└─ _get_observation() 不读取 golden_answer.target_agent_observation_o_t
└─ 调用 ObservationGenerator.generate()

policy.hide_mental_from_evaluator == False
policy.hide_physical_from_evaluator == False
└─ ScoringModule.score() 不隐藏 state / observation 字段
```

因此，`full_mwm` 的核心路径可以压缩为：

```text
run_async
-> MentisPipeline.run_record
-> _run_sample
-> _get_world_state / StateParser.parse
-> _get_observation / ObservationGenerator.generate
-> TargetPseudoAgent.sample_actions
-> ActionParser.parse
-> _run_branches
   -> _run_one_branch
      -> PhysicalTransitionModel.predict
      -> MentalTransitionModel.predict
      -> NextStateMerger.merge
      -> ScoringModule.score
-> DecisionModule.decide
-> GeneratedResults.as_dict
-> StreamingRecordWriter.write
```
