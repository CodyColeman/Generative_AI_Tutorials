# 🤖 Tutorial 06 — Agents

Give a language model tools and a loop. An agent is an LLM that observes its environment, decides what action to take, executes that action, and repeats — until it has an answer.

---

## Overview

| File | What it does | Core libraries |
|---|---|---|
| `agents_tutorial.py` | Builds a tool-using agent from scratch, then shows the same agent in smolagents | `transformers`, `smolagents`, `sentence-transformers`, `datasets` |

**Model:** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — small instruction-tuned model, CPU-runnable, reliable at following structured output formats.

**Tools the agent can use:**
- `calculator` — evaluates arithmetic expressions
- `web_search` — stub that returns simulated search results (no API key required)
- `rag_retrieval` — searches a small embedded knowledge base built from AG News

**Goals:**
- Understand the Observe → Think → Act loop mechanically, not just conceptually
- Implement a complete ReAct agent from scratch using only `transformers`
- See how the same agent is expressed more concisely with `smolagents`
- Understand tool definitions, tool parsing, and how the LLM decides which tool to call

---

## Quick Start

```bash
pip install transformers smolagents sentence-transformers datasets torch

python agents_tutorial.py
```

> **First run:** Downloads `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) and `all-MiniLM-L6-v2` (~90 MB). Both are cached after first run. Full script runs on CPU in a few minutes.

---

## What Is an Agent?

A plain LLM call is stateless: you send a prompt, you get a response. An **agent** wraps the LLM in a loop that lets it take actions in the world and observe the results before producing a final answer.

```
              ┌─────────────────────────────────────────┐
              │              AGENT LOOP                  │
              │                                          │
  Question ──►│  Observe ──► Think ──► Act               │──► Final Answer
              │     ▲                   │                │
              │     │     Tool result   │                │
              │     └───── ◄── Tool ◄───┘                │
              │                                          │
              └─────────────────────────────────────────┘
```

Each iteration:
1. **Observe** — the agent receives the current state: original question + history of past thoughts and tool results
2. **Think** — the LLM generates its reasoning and decides what to do next
3. **Act** — the agent either calls a tool (and loops back) or emits a final answer

This is the **ReAct pattern** (Reasoning + Acting), introduced in the [ReAct paper](https://arxiv.org/abs/2210.03629).

---

## The ReAct Pattern in Detail

The LLM is prompted to produce output in a structured format on every turn:

```
Thought: I need to find the current price of AAPL stock.
Action: web_search
Action Input: AAPL stock price today

[Tool executes, returns result]

Observation: AAPL is trading at $227.50 as of market close.

Thought: I now have the price. I can answer the question.
Final Answer: Apple (AAPL) closed at $227.50 today.
```

The agent loop parses each LLM output looking for:
- `Action:` + `Action Input:` → call the named tool, feed back the result as `Observation:`
- `Final Answer:` → stop the loop, return the answer

Everything — question, thoughts, observations — is accumulated in a single growing context window. The LLM sees the full history on every step.

```
Turn 1 context:
  [System prompt + tool descriptions]
  [User question]

Turn 2 context:
  [System prompt + tool descriptions]
  [User question]
  [Thought 1 + Action 1 + Action Input 1]
  [Observation 1]

Turn 3 context:
  [System prompt + tool descriptions]
  [User question]
  [Thought 1 + Action 1 + Action Input 1]
  [Observation 1]
  [Thought 2 + Action 2 + Action Input 2]
  [Observation 2]
  ...
```

---

## Tool Definition

A tool is a function paired with a description the LLM can read. The description is the interface contract — the LLM decides whether to call a tool purely based on its description and the user's question.

```python
@dataclass
class Tool:
    name: str           # Short identifier — used in "Action: <name>"
    description: str    # Natural language description of what it does
                        # and what input it expects. This is what the LLM reads.
    fn: callable        # The actual Python function to call
```

The tool descriptions are injected into the system prompt:

```
You have access to the following tools:

calculator: Evaluates a mathematical expression and returns the result.
            Input: a valid Python arithmetic expression as a string.
            Example: "2 ** 10 + 15 / 3"

web_search: Searches the web for current information.
            Input: a search query string.
            Example: "latest GDP figures United States 2024"

rag_retrieval: Searches a knowledge base of news articles using semantic similarity.
               Input: a natural language query.
               Example: "tech company earnings reports"
```

> ✅ **Write tool descriptions as if explaining to a smart intern who has never seen your code.** The LLM cannot see the function source — only the description. If the description is vague, the agent will misuse the tool.

> ⚠️ Tool names must be exact strings — the parser does a literal match on `Action: <name>`. Use lowercase, no spaces.

---

## 🎛️ Parameter Guide

### `max_iterations` — Agent Loop Limit

Maximum number of Observe→Think→Act cycles before the agent is forced to stop.

| Value | Effect |
|---|---|
| 3 | Fine for simple single-tool questions |
| 5 | ✅ Good default — handles multi-step reasoning |
| 10 | Needed for complex tasks requiring many tool calls |
| Unbounded | Risk: infinite loop if the model gets stuck |

> ✅ **Recommended default:** `5`. Always set a hard limit. An agent that never terminates is a bug, not a feature.

> ⚠️ If the agent hits `max_iterations`, it should still produce a best-effort answer from what it's observed so far — not silently fail.

---

### `max_new_tokens` — Per-Step Generation Length

How many tokens the LLM generates on each think step.

| Value | Effect |
|---|---|
| 128 | Fast, but may cut off reasoning mid-thought |
| 256 | ✅ Good for most tool-calling steps |
| 512 | Needed for complex multi-step reasoning chains |
| 1024+ | Long-form planning; may pad unnecessarily |

> ✅ **Recommended default:** `256`. Most ReAct steps are short: a thought + one action.

> ⚠️ Unlike chat, you don't want the model to write an essay at each step. Short, structured outputs (Thought / Action / Action Input) are more reliable than verbose ones.

---

### `temperature` (agent) — Per-Step Sampling Temperature

Controls randomness in the agent's thinking.

| Value | Effect |
|---|---|
| 0.0 | Greedy / deterministic — most reliable for structured output parsing |
| 0.1–0.3 | ✅ Light stochasticity — some variation without chaos |
| 0.7+ | Creative but fragile — agent may deviate from expected format |

> ✅ **Recommended default:** `0.1` for agents. Lower than you'd use for open-ended generation. Format compliance matters more than creativity in a tool-calling loop.

> ⚠️ High temperature increases the chance the model produces malformed output (`Action:` missing, wrong tool name, etc.) that breaks the parser. If the agent misbehaves, lower temperature first.

---

### `stop_sequences` — Generation Halt Triggers

Strings that cause the LLM to stop generating immediately when produced.

| Value | Why useful |
|---|---|
| `["Observation:"]` | Stops the model from hallucinating its own tool results |
| `["Final Answer:"]` | Stops at the answer without trailing padding |
| `["\n\nHuman:"]` | Prevents the model from role-playing the user's next turn |

> ✅ Always include `"Observation:"` as a stop sequence. Without it, the model will often generate what it imagines the tool result should be rather than waiting for the real one.

---

### `rag_top_k` — Documents Retrieved per Query

How many documents the RAG tool returns for each retrieval call.

| Value | Effect |
|---|---|
| 1 | Minimal context — fast but risks missing the answer |
| 3 | ✅ Good balance — enough context without overloading the prompt |
| 5–10 | Rich context — useful for open-domain QA |
| 20+ | May exceed context window; returns diminishing value |

> ✅ **Recommended default:** `3`. The agent can call `rag_retrieval` again with a different query if the first result isn't sufficient.

---

## Agent Failure Modes

Agents fail in distinctive ways. Knowing these patterns helps you debug faster.

### 1. Hallucinated Tool Results
The model generates its own `Observation:` instead of waiting for the real tool output.

```
Thought: I should search for this.
Action: web_search
Action Input: GDP growth 2024
Observation: GDP grew by 2.8% in 2024.   ← LLM wrote this, not the tool
```

**Fix:** Add `"Observation:"` to `stop_sequences`. The model will stop generating before it can fabricate the observation.

---

### 2. Wrong Tool Name
The model generates `Action: search` when the tool is named `web_search`.

**Fix:** List exact tool names in the system prompt. Make names unambiguous. Validate the parsed name against the tool registry and feed back an error observation if it doesn't match.

---

### 3. Malformed Action Input
The model generates the tool name correctly but the input is in the wrong format.

**Fix:** Include a concrete example in every tool description. The model is excellent at pattern-matching to examples.

---

### 4. Infinite Reasoning Loop
The model keeps thinking without ever committing to a `Final Answer:`.

```
Thought: I need more information.
Action: web_search
...
Thought: I still need more information.
Action: web_search
...
```

**Fix:** Hard cap on `max_iterations`. At the limit, inject a system message: "You must now provide a Final Answer based on what you know."

---

### 5. Context Window Overflow
Long agentic chains accumulate lots of history. Eventually the context fills up and early information is lost or the model crashes.

**Fix:** Implement a sliding window or summarisation step. After N turns, summarise the history into a compact paragraph and replace it in the context.

---

## Manual Loop vs. smolagents

| Aspect | Manual loop | smolagents |
|---|---|---|
| Lines of code | ~150 | ~20 |
| Transparency | Full — every parsing step is visible | Abstracted |
| Flexibility | Total control over prompts, parsing, error handling | Constrained by framework conventions |
| Debugging | Easy to inspect at every step | Requires framework knowledge |
| Multi-agent | Build yourself | Built-in `ManagedAgent` |
| Model compatibility | Any `transformers` model | HuggingFace Hub + API models |
| Learning value | ✅ High — teaches the mechanics | Lower for learning, higher for shipping |

> ✅ **Learn with the manual loop. Ship with smolagents (or LangChain/LlamaIndex for production).** Understanding what's happening under the hood makes you much more effective when the framework does something unexpected.

---

## Tool Architecture — This Tutorial

```
                         Agent Loop
                              │
                    ┌─────────▼──────────┐
                    │   LLM (Qwen 0.5B)  │
                    │   Sees: question   │
                    │   + history        │
                    │   + tool descs     │
                    └─────────┬──────────┘
                              │ generates
                    ┌─────────▼──────────┐
                    │   Output Parser    │
                    │   extracts Action  │
                    │   + Action Input   │
                    └──┬──────┬──────┬───┘
                       │      │      │
               ┌───────▼──┐ ┌─▼────┐ ┌▼────────────┐
               │calculator│ │ web  │ │rag_retrieval│
               │          │ │search│ │             │
               │eval()    │ │stub  │ │MiniLM-L6-v2 │
               │          │ │      │ │+ FAISS-style│
               └───────┬──┘ └─┬────┘ └┬────────────┘
                       │      │       │
                    ┌──▼──────▼───────▼──┐
                    │    Observation      │
                    │  appended to context│
                    └────────────────────┘
```

---

## What's Next

| Tutorial | What it adds |
|---|---|
| ← [03 RAG Models](../03_rag_models/README.md) | The retrieval system powering the `rag_retrieval` tool |
| ← [05 Embeddings](../05_embeddings/README.md) | The embedding model powering the RAG tool's semantic search |
| Production reading | [LangChain agents](https://python.langchain.com/docs/modules/agents/), [LlamaIndex agents](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/) |

---

## Resources

- [ReAct: Synergising Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — the original paper
- [Toolformer](https://arxiv.org/abs/2302.04761) — teaching models to use tools via self-supervised learning
- [smolagents documentation](https://huggingface.co/docs/smolagents)
- [Function Calling — OpenAI docs](https://platform.openai.com/docs/guides/function-calling) — the API-based approach to structured tool use
- [AgentBench](https://github.com/THUDM/AgentBench) — benchmark for evaluating LLM agents
- [Qwen2.5 model family](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) — the model used in this tutorial
