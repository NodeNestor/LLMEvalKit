# Comparison — lfm2.5-350m-running vs lfm2-1.2b-running vs qwen3.5-0.8b-running

| metric | lfm2.5-350m-running | lfm2-1.2b-running | qwen3.5-0.8b-running |
|---|---|---|---|
| `simple-chat.prompts_ok_total` | 6/6 | 6/6 | 6/6 |
| `lm-eval.gsm8k.exact_match,flexible-extract` | 0.3000 | 0.5800 | 0.4000 |
| `lm-eval.gsm8k.exact_match,strict-match` | 0.2000 | 0.5200 | 0.2400 |
| `lm-eval.gsm8k.exact_match_stderr,flexible-extract` | 0.0655 | 0.0705 | 0.0700 |
| `lm-eval.gsm8k.exact_match_stderr,strict-match` | 0.0571 | 0.0714 | 0.0610 |
| `lm-eval.gsm8k.sample_len` | 50 | 50 | 50 |
| `inspect.gsm8k.match.accuracy` | 0.0333 | 0.6000 | 0.4667 |
| `inspect.gsm8k.match.stderr` | 0.0333 | 0.0910 | 0.0926 |
| `inspect.humaneval.verify.accuracy` | 0.2000 | 0.5667 | 0.3000 |
| `inspect.humaneval.verify.stderr` | 0.0743 | 0.0920 | 0.0851 |
| `inspect.ifeval.instruction_following.final_acc` | 0.7222 | 0.5750 | 0.6389 |
| `inspect.ifeval.instruction_following.final_stderr` | 0.0907 | 0.0909 | 0.0982 |
| `inspect.ifeval.instruction_following.inst_loose_acc` | 0.7556 | 0.6222 | 0.6667 |
| `inspect.ifeval.instruction_following.inst_loose_stderr` | 0.0681 | 0.0745 | 0.0804 |
| `inspect.ifeval.instruction_following.inst_strict_acc` | 0.7333 | 0.5778 | 0.6222 |
| `inspect.ifeval.instruction_following.inst_strict_stderr` | 0.0730 | 0.0761 | 0.0802 |
| `inspect.ifeval.instruction_following.prompt_loose_acc` | 0.7000 | 0.5667 | 0.6667 |
| `inspect.ifeval.instruction_following.prompt_loose_stderr` | 0.0851 | 0.0920 | 0.0875 |
| `inspect.ifeval.instruction_following.prompt_strict_acc` | 0.7000 | 0.5333 | 0.6000 |
| `inspect.ifeval.instruction_following.prompt_strict_stderr` | 0.0851 | 0.0926 | 0.0910 |
| `inspect.mbpp.verify.accuracy` | 0.0000 | 0.5000 | 0.5333 |
| `inspect.mbpp.verify.stderr` | 0.0000 | 0.0928 | 0.0926 |
| `bigcodebench.generated_samples` | 0 | 0 | 0 |
| `total.wall_time` | 7.8 min | 17.7 min | 27.2 min |
