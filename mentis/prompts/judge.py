from __future__ import annotations

from typing import Any

from .common import prompt_header, prompt_json


def build_state_judge_prompt(pred: dict[str, Any], gold: dict[str, Any]) -> str:
    criteria = (
        "Judge whether predicted s_t is semantically equivalent to gold s_t, not whether "
        "field names match exactly. Physical score covers agents, objects, location/time, "
        "events, spatial/contact/occlusion relations, affordances and constraints. Mental "
        "score covers beliefs, intentions, goals, emotions, perceptual access, nested/social "
        "beliefs, and uncertainty. Coupling score covers whether mental claims are supported "
        "by physical evidence instead of hallucinated mind reading."
    )
    return _judge_prompt("current_state_s_t", pred, gold, criteria)


def build_observation_judge_prompt(pred: dict[str, Any], gold: dict[str, Any]) -> str:
    criteria = (
        "Judge target-agent observation o_t under POMDP limits. Physical score covers what "
        "the target can see, hear, touch, locate, and infer about objects/events from their "
        "perspective. Mental score covers self state and inferred other-agent states from "
        "observable cues. Penalize leakage of hidden global state, missing target perspective, "
        "or impossible perceptual access."
    )
    return _judge_prompt("target_agent_observation_o_t", pred, gold, criteria)


def build_next_state_judge_prompt(pred: dict[str, Any], gold: dict[str, Any]) -> str:
    criteria = (
        "Judge one option branch from next_state_s_{t+1}. The package contains option_id, "
        "the candidate action for that option, and the predicted/gold next_state_branch. "
        "The branch must contain physical_state and mental_state. Judge it as an explicit "
        "transition from s_t through that option's candidate action. "
        "Physical score covers whether physical changes obey action "
        "carriers, hand occupancy, reachability, occlusion, containment, support, collision, "
        "and temporal order. Mental score covers belief, intention, emotion, knowledge and "
        "social-inference updates caused by visible/audible consequences. Transition score "
        "covers whether s_t -> action -> s_t+1 is causally reasonable even when the wording "
        "differs from gold."
    )
    return _judge_prompt("next_state_s_{t+1}", pred, gold, criteria)


def build_score_judge_prompt(pred: dict[str, Any], gold: dict[str, Any]) -> str:
    criteria = (
        "Judge score table alignment. Physical score covers physically_plausible values, "
        "mental score covers mentally_consistent values, and semantic_consistency_score "
        "covers whether reasoning and final ranking match gold diagnostics."
    )
    return _judge_prompt("score", pred, gold, criteria)


def _judge_prompt(name: str, pred: dict[str, Any], gold: dict[str, Any], criteria: str) -> str:
    output_schema = {
        "item": name,
        "physical_score": 0.0,
        "mental_score": 0.0,
        "semantic_consistency_score": 0.0,
        "coupling_score": 0.0,
        "transition_score": 0.0,
        "error_types": [],
        "reasoning": "one concise sentence",
        "option_scores": {
            "A": {
                "physical_score": 0.0,
                "mental_score": 0.0,
                "semantic_consistency_score": 0.0,
                "transition_score": 0.0,
                "error_types": [],
                "reasoning": "one concise sentence",
            }
        },
        "confidence": 0.0,
    }
    return (
        prompt_header(f"Judge predicted {name} against gold annotation")
        + "You are an evaluation judge, not the task-solving model. Compare prediction "
        "against gold annotation semantically. Do not re-solve the multiple-choice task "
        "and do not reward lucky final answers when the state reasoning is wrong. Use "
        "scores in [0,1], where 1 means semantically equivalent and causally faithful.\n"
        f"Criteria:\n{criteria}\n"
        f"Output schema:\n{prompt_json(output_schema)}\n"
        f"Prediction package:\n{prompt_json(pred)}\nGold package:\n{prompt_json(gold)}"
    )
