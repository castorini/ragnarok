# Ragnarok Prompt Modes

## Available Modes

| Mode | Template File | Citation | Description |
|------|---------------|----------|-------------|
| `chatqa` | `chatqa.yaml` | No | Conversational QA style, no citations |
| `ragnarok_v2` | `ragnarok_v2.yaml` | Yes | Earlier citation format |
| `ragnarok_v3` | `ragnarok_v3.yaml` | Yes | Improved citation instructions |
| `ragnarok_v4` | `ragnarok_v4.yaml` | Yes | **Recommended default** — best citation quality |
| `ragnarok_v4_no_cite` | `ragnarok_v4_no_cite.yaml` | No | Same as v4 but without citations |
| `ragnarok_v4_biogen` | `ragnarok_v4_biogen.yaml` | Yes | Biomedical generation with citations |
| `ragnarok_v5_biogen` | `ragnarok_v5_biogen.yaml` | Yes | Updated biomedical generation |
| `ragnarok_v5_biogen_no_cite` | `ragnarok_v5_biogen_no_cite.yaml` | No | Biomedical without citations |

Note: `unspecified` and `cohere` are internal values — do not use directly.

## Mode Selection Guide

- **General RAG**: `ragnarok_v4` (best overall)
- **No citations needed**: `ragnarok_v4_no_cite` or `chatqa`
- **Biomedical domain**: `ragnarok_v4_biogen` or `ragnarok_v5_biogen`
- **Cohere models**: mode is auto-selected; just use `--model command-r-plus`

## Template Structure (YAML)

```yaml
method: "ragnarok_v4"
system_message: "You are a helpful assistant..."
instruction: |
  Given the query and passages, generate a comprehensive answer
  with inline citations [1][2] referencing passage numbers...
prefix_user: "{query}"
```

## Inspecting Modes

```bash
# List all available modes
ragnarok prompt list

# Show a mode's template
ragnarok prompt show --prompt-mode ragnarok_v4

# Render with real input
ragnarok prompt render --prompt-mode ragnarok_v4 --model gpt-4o \
  --input-json '{"query":"test","candidates":["passage 1","passage 2"]}' --part user
```
