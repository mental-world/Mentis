from __future__ import annotations

from typing import Any

from .common import prompt_header, prompt_json


def build_scoring_prompt(
    target_agent: str,
    world_state: dict[str, Any],
    observation: dict[str, Any],
    question: str,
    action: dict[str, Any],
    next_state: dict[str, Any],
) -> str:
    output_schema = {
        "mentally_consistent": "0.0",
        "physically_plausible": "0.0",
        "socially_appropriate": "0.0",
        "safety_legality_veto": "true/false",
        "reasoning": "concise rationale",
    }
    return (
        prompt_header("Evaluate one imagined action branch")
        + "Use the full tuple (target_agent_description, question, s_t, o_t, a_t, "
        "s_t+1) to score this single action branch. Return exactly the schema below. "
        "Scores must be strings or numbers in [0,1]. mentally_consistent evaluates "
        "whether the action satisfies the current target agent's goal and matches the "
        "target agent's mindset/personality under o_t. physically_plausible evaluates "
        "whether the action and imagined s_t+1 are physically possible from s_t. "
        "socially_appropriate evaluates custom, moral, politeness, role, and social norm "
        "appropriateness.\n"
        + "For each numeric dimension, first assign an internal 1-5 grade, then output only "
        "the scaled [0,1] score using this mapping: grade 1 -> 0.0, grade 2 -> 0.25, "
        "grade 3 -> 0.5, grade 4 -> 0.75, grade 5 -> 1.0. Do not add grade keys.\n"
        + "physically_plausible rubric:\n"
        + "Grade 5: The action and imagined s_t+1 are fully physically possible from s_t. "
        "Reachability, affordances, visibility, contact, support, collision, containment, "
        "and temporal order are all coherent.\n"
        + "Grade 4: The action and imagined s_t+1 are mostly physically plausible. Any "
        "issues are minor, underspecified, or do not affect whether the branch could happen.\n"
        + "Grade 3: The branch is partially physically plausible or uncertain. Some physical "
        "details are missing, weakly supported, or ambiguous, but there is no severe "
        "physical impossibility.\n"
        + "Grade 2: The branch is mostly physically implausible. It contains a major "
        "unsupported physical step, strong conflict with s_t, or weak causal connection "
        "from a_t to s_t+1.\n"
        + "Grade 1: The branch contains a serious physical impossibility or contradiction, "
        "such as impossible reachability, affordance, visibility, contact, support, "
        "collision, containment, temporal order, or an s_t+1 that cannot result from "
        "the action.\n"
        + "mentally_consistent rubric:\n"
        + "Grade 5: The action and imagined s_t+1 fully satisfy the target agent's explicit "
        "goal and match the target's beliefs, attention, intentions, emotions, preferences, "
        "personality, and perspective under o_t.\n"
        + "Grade 4: The branch is mostly mentally consistent. It fits the target's goal and "
        "mindset with only minor caveats or uncertainty.\n"
        + "Grade 3: The branch is mixed or underspecified. It partially matches the target's "
        "goal or mindset, but important beliefs, intentions, emotions, or perspective "
        "evidence are unclear.\n"
        + "Grade 2: The branch is mostly mentally inconsistent. It conflicts with the "
        "target's goal, beliefs, preferences, likely emotional response, or perspective, "
        "but the contradiction is not absolute.\n"
        + "Grade 1: The action or imagined s_t+1 clearly contradicts the question, o_t, or "
        "the target agent's explicit goal, belief state, intention, emotion, preference, "
        "personality, or perspective.\n"
        + "socially_appropriate rubric:\n"
        + "Grade 5: The action and imagined s_t+1 are fully appropriate under the scene's "
        "social norms, roles, morality, politeness, customs, and relationship context.\n"
        + "Grade 4: The branch is mostly socially appropriate. Any awkwardness or norm "
        "concern is minor or context-sensitive.\n"
        + "Grade 3: The branch is socially mixed, ambiguous, or only partly appropriate. "
        "It may satisfy one norm while straining another.\n"
        + "Grade 2: The branch is mostly socially inappropriate. It violates role, "
        "politeness, relationship, moral, or custom expectations, but is not an extreme "
        "safety or legality issue.\n"
        + "Grade 1: The branch contains an obvious social, moral, role, politeness, or "
        "custom violation.\n"
        + "safety_legality_veto must be true only for illegal actions or serious safety risks; "
        "otherwise false.\n"
        + "reasoning must be one concise rationale covering the three dimensions plus "
        "safety/legality, and must cite specific evidence from the imagined s_t+1, such "
        "as changed object positions, physical conditions, character body motions, beliefs, "
        "goals, emotions, relations, or atmosphere. "
        "Do not score directly from the option wording alone; ground every score in the "
        "s_t -> a_t -> s_t+1 branch. Do not choose the final answer and do not add extra keys.\n"
        f"Output schema:\n{prompt_json(output_schema)}\n"
        f"target_agent_description:\n{target_agent}\n"
        f"question:\n{question}\n"
        f"s_t:\n{prompt_json(world_state)}\n"
        f"o_t:\n{prompt_json(observation)}\n"
        f"a_t:\n{prompt_json(action)}\n"
        f"s_t+1:\n{prompt_json(next_state)}"
    )
