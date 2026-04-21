# Comparison — lfm2.5-350m-running vs qwen3.5-0.8b-running

| metric | lfm2.5-350m-running | qwen3.5-0.8b-running |
|---|---|---|
| `simple-chat.prompts_ok_total` | 6/6 | 6/6 |
| `lm-eval.gsm8k.exact_match,flexible-extract` | 0.4000 | 0.3000 |
| `lm-eval.gsm8k.exact_match,strict-match` | 0.0000 | 0.0000 |
| `lm-eval.gsm8k.exact_match_stderr,flexible-extract` | 0.1633 | 0.1528 |
| `lm-eval.gsm8k.exact_match_stderr,strict-match` | 0.0000 | 0.0000 |
| `lm-eval.gsm8k.sample_len` | 10 | 10 |
| `inspect.gsm8k.match.accuracy` | 0.0000 | 0.4000 |
| `inspect.gsm8k.match.stderr` | 0.0000 | 0.2449 |
| `inspect.humaneval.verify.accuracy` | 0.0000 | 0.4000 |
| `inspect.humaneval.verify.stderr` | 0.0000 | 0.2449 |
| `inspect.ifeval.instruction_following.final_acc` | 0.6750 | 0.6125 |
| `inspect.ifeval.instruction_following.final_stderr` | 0.2398 | 0.2592 |
| `inspect.ifeval.instruction_following.inst_loose_acc` | 0.7500 | 0.6250 |
| `inspect.ifeval.instruction_following.inst_loose_stderr` | 0.0988 | 0.1976 |
| `inspect.ifeval.instruction_following.inst_strict_acc` | 0.7500 | 0.6250 |
| `inspect.ifeval.instruction_following.inst_strict_stderr` | 0.0988 | 0.1976 |
| `inspect.ifeval.instruction_following.prompt_loose_acc` | 0.6000 | 0.6000 |
| `inspect.ifeval.instruction_following.prompt_loose_stderr` | 0.2449 | 0.2449 |
| `inspect.ifeval.instruction_following.prompt_strict_acc` | 0.6000 | 0.6000 |
| `inspect.ifeval.instruction_following.prompt_strict_stderr` | 0.2449 | 0.2449 |
| `inspect.mbpp.verify.accuracy` | 0.0000 | 0.4000 |
| `inspect.mbpp.verify.stderr` | 0.0000 | 0.2449 |
| `bigcodebench.generated_samples` | 3 | 3 |
| `total.wall_time` | 3.3 min | 18.8 min |
