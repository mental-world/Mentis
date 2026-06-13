from __future__ import annotations

from mentis.schema import MentalState, PhysicalState, WorldState


class NextStateMerger:
    def merge(
        self,
        physical: PhysicalState,
        mental: MentalState,
    ) -> WorldState:
        return WorldState(
            physical_state=physical,
            mental_state=mental,
        )
