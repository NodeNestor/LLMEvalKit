from .base import Runner, RunResult
from .lm_eval_harness import LMEvalHarness
from .lighteval import LightEval
from .evalchemy import Evalchemy
from .bfcl import BFCL
from .mini_swe_agent import MiniSWEAgent
from .ruler import RULER
from .harmbench import HarmBench
from .arena_hard import ArenaHardAuto
from .inspect_ai import InspectAI
from .evalplus import EvalPlus
from .bigcodebench import BigCodeBench
from .livecodebench import LiveCodeBench
from .simple_chat import SimpleChat

RUNNERS = {
    "lm-eval-harness": LMEvalHarness,
    "lighteval": LightEval,
    "evalchemy": Evalchemy,
    "bfcl": BFCL,
    "mini-swe-agent": MiniSWEAgent,
    "ruler": RULER,
    "harmbench": HarmBench,
    "arena-hard": ArenaHardAuto,
    "inspect-ai": InspectAI,
    "evalplus": EvalPlus,
    "bigcodebench": BigCodeBench,
    "livecodebench": LiveCodeBench,
    "simple-chat": SimpleChat,
}


def make_runner(name: str, **kwargs) -> Runner:
    if name not in RUNNERS:
        raise ValueError(f"Unknown runner: {name}. Available: {list(RUNNERS)}")
    return RUNNERS[name](**kwargs)


__all__ = ["Runner", "RunResult", "make_runner", "RUNNERS"]
