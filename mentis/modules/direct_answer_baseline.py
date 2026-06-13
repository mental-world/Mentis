from __future__ import annotations

from mentis.prompts import build_direct_answer_baseline_prompt
from mentis.schema import DirectAnswerResult
from mentis.utils.runtime_input import direct_answer_input_from_sample
from mentis.utils.tracing import response_metadata

from .base import ModuleBase


class DirectAnswerBaseline(ModuleBase):
    async def answer(self, sample: dict) -> tuple[str, dict]:
        direct_input = direct_answer_input_from_sample(sample)
        prompt = build_direct_answer_baseline_prompt(direct_input.as_dict())
        response = await self.call_llm(
            sample_id=sample["sample_id"],
            task="direct_answer",
            prompt=prompt,
            model=self.config.models.direct_answer_model,
            context={"media_scene": sample.get("_media_scene") or {}},
            schema=DirectAnswerResult,
        )
        final_action = str(response.parsed.get("final_action", "")).strip()
        meta = response_metadata(response)
        if final_action not in direct_input.option_ids():
            warnings = list(meta.get("warnings", []))
            warnings.append(f"direct_answer_invalid_final_action:{final_action}")
            meta["warnings"] = sorted(set(warnings))
        return final_action, meta
