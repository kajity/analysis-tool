from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_STEPS = ("Input", "Preview", "Configure", "Review")


@dataclass
class PulseWorkflow:
    steps: tuple[str, ...] = DEFAULT_STEPS
    step_index: int = 0
    finished: bool = False
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("PulseWorkflow requires at least one step.")
        if not 0 <= self.step_index < len(self.steps):
            raise ValueError("step_index is outside the configured steps.")

    @property
    def current_step(self) -> str:
        return self.steps[self.step_index]

    @property
    def can_go_back(self) -> bool:
        return self.step_index > 0 and not self.finished

    @property
    def can_go_next(self) -> bool:
        return self.step_index < len(self.steps) - 1 and not self.finished

    @property
    def can_finish(self) -> bool:
        return self.step_index == len(self.steps) - 1 and not self.finished

    def reset(self) -> None:
        self.step_index = 0
        self.finished = False
        self.events.append("reset")

    def back(self) -> bool:
        if not self.can_go_back:
            return False
        self.step_index -= 1
        self.events.append(f"back:{self.current_step}")
        return True

    def next(self) -> bool:
        if not self.can_go_next:
            return False
        self.step_index += 1
        self.events.append(f"next:{self.current_step}")
        return True

    def finish(self) -> bool:
        if not self.can_finish:
            return False
        self.finished = True
        self.events.append("finish")
        return True
