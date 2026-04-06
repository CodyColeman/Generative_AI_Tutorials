# =============================================================================
# Tutorial 06 — Agents
# =============================================================================
#
# Environment:  Python 3.9+, CPU-first
#
# Dependencies:
#   pip install transformers smolagents sentence-transformers datasets torch
#
# Sections:
#   1.  Imports & Configuration
#   2.  Tool Definitions — Calculator, Web Search, RAG Retrieval
#   3.  Build the RAG Knowledge Base
#   4.  System Prompt & ReAct Format
#   5.  Output Parser — Extract Action or Final Answer
#   6.  The Agent Loop (Manual ReAct Implementation)
#   7.  Run the Manual Agent — Single-Tool Questions
#   8.  Run the Manual Agent — Multi-Step Questions
#   9.  Agent Failure Modes — Demonstrated & Explained
#  10.  smolagents — Same Agent in ~20 Lines
#  11.  Summary
#
# CPU note:
#   Qwen2.5-0.5B-Instruct (~1 GB) runs on CPU. Expect ~5–15s per generation
#   step depending on hardware. Each agent question runs 1–5 steps.
# =============================================================================

# ─────────────────────────────────────────────────────
# SECTION 1: IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────

import re
import ast
import json
import time
import textwrap
import operator
import warnings
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

warnings.filterwarnings("ignore")

# ── LLM ───────────────────────────────────────────────
# Qwen2.5-0.5B-Instruct: small instruction-tuned model, CPU-runnable.
# Reliable at following ReAct format with a clear system prompt.
# For stronger reasoning, swap to:
#   "Qwen/Qwen2.5-1.5B-Instruct"   (better, still CPU-feasible)
#   "Qwen/Qwen2.5-7B-Instruct"     (GPU recommended)
#   "mistralai/Mistral-7B-Instruct-v0.3"  (GPU required)
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Embedding model (for RAG tool) ────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Dataset (for RAG knowledge base) ──────────────────
RAG_DATASET      = "ag_news"
RAG_N_DOCS       = 200    # documents to index in the knowledge base
RAG_TOP_K        = 3
# ── rag_top_k ─────────────────────────────────────────────────────────────
# Number of documents returned per retrieval call.
# 3 is a good balance: enough context without flooding the prompt.
# The agent can call rag_retrieval again with a refined query if needed.

# ── Agent loop settings ───────────────────────────────
MAX_ITERATIONS = 5
# ── max_iterations ────────────────────────────────────────────────────────
# Hard cap on Observe → Think → Act cycles.
# 5 handles most multi-step questions. Never leave this unbounded —
# a confused model can loop forever without a limit.

MAX_NEW_TOKENS = 256
# ── max_new_tokens ────────────────────────────────────────────────────────
# Tokens generated per think step.
# 256 covers Thought + Action + Action Input comfortably.
# Increase to 512 if the model is cutting off mid-reasoning.

TEMPERATURE = 0.1
# ── temperature (agent) ───────────────────────────────────────────────────
# Lower than typical generation. Format compliance (Thought / Action /
# Action Input) matters more than creativity in a tool-calling loop.
# 0.1 gives slight variation without breaking the output structure.

# Stop sequences — see README for why "Observation:" is critical
STOP_SEQUENCES = ["Observation:", "\nObservation"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────
# SECTION 2: TOOL DEFINITIONS
# ─────────────────────────────────────────────────────
#
# Each tool is a Python function paired with a name and description.
# The description is injected into the system prompt — it's the only
# information the LLM has about what the tool does and how to call it.
# Write descriptions as contracts: what it accepts, what it returns, example.

print("=" * 65)
print("SECTION 2: Tool Definitions")
print("=" * 65)


@dataclass
class Tool:
    """
    A callable tool that the agent can invoke.

    Attributes
    ----------
    name : str
        Identifier used in "Action: <name>". Must match exactly.
        Use lowercase, no spaces. The parser does a literal string match.
    description : str
        Natural language description injected into the system prompt.
        This is the LLM's entire interface to this tool — make it precise.
        Include: what it does, what input format it expects, an example.
    fn : Callable[[str], str]
        The Python function to call with the action input string.
        Always returns a string (the observation fed back to the model).
    """
    name: str
    description: str
    fn: Callable[[str], str]

    def __call__(self, input_str: str) -> str:
        """Execute the tool and return its result as a string."""
        try:
            result = self.fn(input_str)
            return str(result)
        except Exception as e:
            return f"Tool error: {type(e).__name__}: {e}"


# ── Tool 1: Calculator ────────────────────────────────

def _calculator_fn(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Uses a restricted AST evaluator instead of eval() to prevent
    arbitrary code execution. Only supports arithmetic operations.

    Parameters
    ----------
    expression : str
        A Python arithmetic expression.
        Examples: "2 ** 10", "1234 * 5678", "(100 - 32) * 5 / 9"

    Returns
    -------
    str
        The numerical result as a string, or an error message.
    """
    # Allowed AST node types — whitelist only arithmetic
    SAFE_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    }

    def _eval(node):
        if type(node) not in SAFE_NODES:
            raise ValueError(f"Unsafe operation: {type(node).__name__}")
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):      # Python < 3.8
            return node.n
        if isinstance(node, ast.BinOp):
            ops = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }
            op_fn = ops.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unknown operator: {type(node.op)}")
            return op_fn(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -_eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +_eval(node.operand)
        raise ValueError(f"Unsupported node: {type(node)}")

    expression = expression.strip().strip('"\'')
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        # Format: hide .0 for whole numbers, keep decimals for fractions
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(round(result, 6))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: could not evaluate '{expression}': {e}"


calculator = Tool(
    name="calculator",
    description=(
        "Evaluates a mathematical expression and returns the numerical result. "
        "Input must be a valid arithmetic expression using +, -, *, /, **, %. "
        "Do not include units or text — numbers and operators only. "
        "Example input: \"(1024 * 8) / 1000\" or \"2 ** 32\""
    ),
    fn=_calculator_fn,
)

# ── Tool 2: Web Search (stub) ─────────────────────────
# In production you'd call the Serper API, SerpAPI, Brave Search, or
# DuckDuckGo. Here we return deterministic stub responses keyed on
# keywords so the tutorial runs without any API key.

_WEB_SEARCH_KB = {
    "gdp":         "US GDP grew 2.8% in 2024 according to the Bureau of Economic Analysis. "
                   "Global GDP growth was estimated at 3.2% by the IMF.",
    "inflation":   "US CPI inflation was 3.2% year-over-year as of the latest report. "
                   "Core PCE, the Fed's preferred measure, came in at 2.8%.",
    "population":  "World population reached 8.1 billion in 2024. "
                   "The US population is approximately 335 million.",
    "python":      "Python 3.13 was released in October 2024, featuring a free-threaded mode "
                   "and an experimental JIT compiler for performance improvements.",
    "openai":      "OpenAI released GPT-4o in May 2024. The company was valued at $157 billion "
                   "following a funding round in late 2024.",
    "apple":       "Apple reported quarterly revenue of $94.9 billion in Q4 2024. "
                   "iPhone sales accounted for approximately 52% of total revenue.",
    "climate":     "Global average temperature in 2024 was approximately 1.5°C above pre-industrial "
                   "levels, the highest on record according to the WMO.",
    "default":     "No specific results found. Try rephrasing your query or using the "
                   "rag_retrieval tool to search the local knowledge base instead.",
}

def _web_search_fn(query: str) -> str:
    """
    Stub web search. Returns pre-written results based on keyword matching.

    Parameters
    ----------
    query : str
        Natural language search query. Keywords matched case-insensitively.

    Returns
    -------
    str
        Simulated search result snippet.
    """
    query_lower = query.lower()
    for keyword, result in _WEB_SEARCH_KB.items():
        if keyword in query_lower:
            return f"Search results for '{query}':\n{result}"
    return f"Search results for '{query}':\n{_WEB_SEARCH_KB['default']}"


web_search = Tool(
    name="web_search",
    description=(
        "Searches the web for current information and returns a result snippet. "
        "Use for factual questions about recent events, statistics, or general knowledge. "
        "Input: a natural language search query string. "
        "Example input: \"US inflation rate 2024\" or \"latest Python release\""
    ),
    fn=_web_search_fn,
)

# ── Tool 3: RAG Retrieval (built in Section 3) ────────
# Defined here as a placeholder; populated after the knowledge base is built.
_rag_retrieval_fn_ref = {"fn": None}   # mutable reference for late binding

def _rag_retrieval_dispatch(query: str) -> str:
    if _rag_retrieval_fn_ref["fn"] is None:
        return "RAG knowledge base not yet initialised."
    return _rag_retrieval_fn_ref["fn"](query)

rag_retrieval = Tool(
    name="rag_retrieval",
    description=(
        "Searches a local knowledge base of news articles using semantic similarity. "
        "Use for questions about news topics, world events, sports, business, or technology. "
        "Returns the top matching document excerpts. "
        "Input: a natural language query describing what you're looking for. "
        "Example input: \"tech company quarterly earnings\" or \"sports championship results\""
    ),
    fn=_rag_retrieval_dispatch,
)

# Registry of all available tools — keyed by name for fast lookup
TOOLS: dict[str, Tool] = {
    calculator.name:    calculator,
    web_search.name:    web_search,
    rag_retrieval.name: rag_retrieval,
}

print(f"\n  Registered {len(TOOLS)} tools:")
for name, tool in TOOLS.items():
    first_line = tool.description.split(".")[0]
    print(f"    {name:<16} — {first_line}")
print()


# ─────────────────────────────────────────────────────
# SECTION 3: BUILD THE RAG KNOWLEDGE BASE
# ─────────────────────────────────────────────────────
#
# We embed a small corpus of AG News headlines so the rag_retrieval
# tool has real documents to search. This is the same embedding
# approach as Tutorial 05 — MiniLM-L6-v2 + cosine similarity.

print("=" * 65)
print("SECTION 3: Build RAG Knowledge Base")
print("=" * 65)

print(f"\n  Loading embedding model: {EMBED_MODEL}")
embed_model = SentenceTransformer(EMBED_MODEL)

print(f"  Loading {RAG_N_DOCS} documents from {RAG_DATASET}...")
raw_ds = load_dataset(RAG_DATASET, split="test")
rag_docs = [ex["text"] for ex in raw_ds.select(range(RAG_N_DOCS))]
rag_labels = [
    {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}[ex["label"]]
    for ex in raw_ds.select(range(RAG_N_DOCS))
]

print(f"  Encoding {len(rag_docs)} documents...")
t0 = time.time()
rag_embeddings = embed_model.encode(
    rag_docs,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"  Encoded in {time.time()-t0:.1f}s. Shape: {rag_embeddings.shape}")


def _rag_fn(query: str) -> str:
    """
    Retrieve the top-K most semantically similar documents for a query.

    Parameters
    ----------
    query : str
        Natural language query. The query is embedded with the same model
        used to encode the corpus — they must match.

    Returns
    -------
    str
        Formatted string of top-K document excerpts with their category labels.
        Each excerpt is truncated to 200 characters to keep the observation
        short enough not to overflow the agent's context window.
    """
    q_vec = embed_model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )[0]
    scores = rag_embeddings @ q_vec
    top_indices = np.argsort(scores)[::-1][:RAG_TOP_K]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        excerpt = rag_docs[idx][:200].replace("\n", " ").strip()
        results.append(
            f"[{rank}] ({rag_labels[idx]}, similarity={scores[idx]:.3f}): {excerpt}..."
        )
    return "\n".join(results)


# Wire the RAG function into the tool via the mutable reference
_rag_retrieval_fn_ref["fn"] = _rag_fn

# Smoke test
test_result = _rag_fn("technology startup funding")
print(f"\n  RAG smoke test — query: 'technology startup funding'")
print(f"  Result preview: {test_result[:200]}...")
print(f"\n  ✅ RAG knowledge base ready ({len(rag_docs)} docs, {rag_embeddings.shape[1]}D embeddings)")
print()


# ─────────────────────────────────────────────────────
# SECTION 4: SYSTEM PROMPT & ReAct FORMAT
# ─────────────────────────────────────────────────────
#
# The system prompt does three jobs:
#   1. Tells the model it's an agent with tools
#   2. Shows the exact output format it must follow (ReAct pattern)
#   3. Lists the available tools with their descriptions
#
# The format section is the most important — a small, explicit example
# in the system prompt is more reliable than paragraphs of instruction.

print("=" * 65)
print("SECTION 4: System Prompt & ReAct Format")
print("=" * 65)


def build_system_prompt(tools: dict[str, Tool]) -> str:
    """
    Construct the system prompt that defines the agent's behaviour.

    The system prompt is the core of a ReAct agent. It must:
    - Establish the Thought / Action / Observation / Final Answer cycle
    - List every tool by name and description
    - Show a concrete few-shot example of the expected format

    Parameters
    ----------
    tools : dict[str, Tool]
        The tool registry. All tools are described in the prompt.

    Returns
    -------
    str
        The complete system prompt string.
    """
    tool_block = "\n".join(
        f"- {name}: {tool.description}"
        for name, tool in tools.items()
    )

    return f"""You are a helpful assistant that answers questions step by step using tools.

You have access to the following tools:
{tool_block}

INSTRUCTIONS:
- Think step by step before acting.
- To use a tool, output EXACTLY this format (each on its own line):

Thought: <your reasoning about what to do>
Action: <tool name — must be one of: {", ".join(tools.keys())}>
Action Input: <the input string to pass to the tool>

- After each tool result (labelled Observation:), continue reasoning.
- When you have enough information, output EXACTLY:

Thought: I now have enough information to answer.
Final Answer: <your complete answer to the original question>

- Only use tools when necessary. Simple questions can be answered directly with Final Answer.
- Never invent an Observation — wait for the real tool result.
- Keep each Thought concise (1-2 sentences).

EXAMPLE:
Question: What is 15% of 240?
Thought: I need to calculate 15% of 240.
Action: calculator
Action Input: 240 * 0.15
Observation: 36
Thought: I now have the result.
Final Answer: 15% of 240 is 36.
"""


SYSTEM_PROMPT = build_system_prompt(TOOLS)
print(f"\n  System prompt ({len(SYSTEM_PROMPT)} chars):")
print("  " + "─" * 60)
for line in SYSTEM_PROMPT.split("\n")[:25]:
    print(f"  {line}")
print("  ... (truncated for display)")
print()


# ─────────────────────────────────────────────────────
# SECTION 5: OUTPUT PARSER
# ─────────────────────────────────────────────────────
#
# The parser reads the LLM's raw text output and extracts:
#   - The tool name and input (if the model wants to call a tool)
#   - The final answer (if the model is done)
#   - Nothing (if the output is malformed — we handle this gracefully)

print("=" * 65)
print("SECTION 5: Output Parser")
print("=" * 65)


@dataclass
class ParsedStep:
    """
    The result of parsing one LLM output step.

    Attributes
    ----------
    thought : str
        The model's reasoning text (everything after "Thought:").
    action : str | None
        Tool name if the model wants to call a tool. None for final answers.
    action_input : str | None
        The input string to pass to the tool. None if no tool call.
    final_answer : str | None
        The model's final answer if it decided to stop. None otherwise.
    raw : str
        The unmodified LLM output, preserved for debugging.
    """
    thought:      str
    action:       Optional[str]       = None
    action_input: Optional[str]       = None
    final_answer: Optional[str]       = None
    raw:          str                 = ""


def parse_llm_output(text: str, valid_tool_names: set[str]) -> ParsedStep:
    """
    Parse a single LLM generation step into a structured ParsedStep.

    The parser uses regex to extract the ReAct format fields. It is
    deliberately lenient — the LLM may produce slightly different
    capitalisation or spacing, so we normalise before matching.

    Parameters
    ----------
    text : str
        Raw text output from the LLM for this generation step.
    valid_tool_names : set[str]
        Set of recognised tool names. If the parsed action name is not
        in this set, the parser returns an error observation rather than
        crashing — this is one of the key failure mode handlers.

    Returns
    -------
    ParsedStep
        Structured representation of the step. Check `final_answer` first:
        if set, the agent loop should terminate. If `action` is set,
        call the tool. If both are None, the output was malformed.
    """
    text = text.strip()

    # Extract Thought (everything between "Thought:" and the next keyword)
    thought_match = re.search(
        r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)",
        text, re.DOTALL | re.IGNORECASE,
    )
    thought = thought_match.group(1).strip() if thought_match else ""

    # Check for Final Answer
    final_match = re.search(
        r"Final Answer:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE
    )
    if final_match:
        return ParsedStep(
            thought=thought,
            final_answer=final_match.group(1).strip(),
            raw=text,
        )

    # Check for Action + Action Input
    action_match = re.search(
        r"Action:\s*(\S+)", text, re.IGNORECASE
    )
    input_match = re.search(
        r"Action Input:\s*(.+?)(?=\nObservation:|\nThought:|$)",
        text, re.DOTALL | re.IGNORECASE,
    )

    if action_match:
        action_name  = action_match.group(1).strip().lower()
        action_input = input_match.group(1).strip() if input_match else ""

        if action_name not in valid_tool_names:
            # Unknown tool — return a special marker that the loop can
            # feed back as an error observation
            return ParsedStep(
                thought=thought,
                action="__unknown__",
                action_input=f"Unknown tool '{action_name}'. "
                             f"Available: {', '.join(sorted(valid_tool_names))}",
                raw=text,
            )

        return ParsedStep(
            thought=thought,
            action=action_name,
            action_input=action_input,
            raw=text,
        )

    # Malformed output — neither Final Answer nor Action found
    return ParsedStep(thought=thought, raw=text)


# Parser unit tests
print("\n  Parser unit tests:")
test_cases = [
    (
        "Thought: I need to calculate this.\nAction: calculator\nAction Input: 100 * 1.08",
        "should detect calculator action",
    ),
    (
        "Thought: I know the answer.\nFinal Answer: The capital of France is Paris.",
        "should detect final answer",
    ),
    (
        "Thought: Let me search.\nAction: search\nAction Input: GDP 2024",
        "should catch unknown tool name",
    ),
    (
        "Thought: Hmm.",
        "should handle malformed output gracefully",
    ),
]

for raw, description in test_cases:
    parsed = parse_llm_output(raw, set(TOOLS.keys()))
    if parsed.final_answer:
        result = f"final_answer='{parsed.final_answer[:40]}...'"
    elif parsed.action == "__unknown__":
        result = f"unknown tool caught: '{parsed.action_input[:50]}'"
    elif parsed.action:
        result = f"action='{parsed.action}', input='{parsed.action_input[:30]}'"
    else:
        result = "malformed (no action, no final answer)"
    print(f"  ✅ {description}")
    print(f"     → {result}")
print()


# ─────────────────────────────────────────────────────
# SECTION 6: THE AGENT LOOP
# ─────────────────────────────────────────────────────
#
# The core loop:
#   1. Format the full conversation (system + history) for the model
#   2. Generate the next step
#   3. Parse the output
#   4. If tool call: execute tool, append observation, loop
#   5. If final answer: return it
#   6. If max_iterations hit: force a final answer

print("=" * 65)
print("SECTION 6: The Agent Loop (Manual ReAct)")
print("=" * 65)

print(f"\n  Loading LLM: {LLM_MODEL}")
print("  (This may take a moment to download on first run...)")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
if DEVICE == "cpu":
    llm = llm.to(DEVICE)

print(f"  ✅ LLM loaded in {time.time()-t0:.1f}s")
param_count = sum(p.numel() for p in llm.parameters())
print(f"  Parameters: {param_count/1e6:.0f}M")


@dataclass
class AgentStep:
    """
    A single recorded step in the agent's trajectory.

    Attributes
    ----------
    iteration    : int   Which loop iteration this step belongs to.
    thought      : str   The model's reasoning.
    action       : str   Tool name called, or "final_answer".
    action_input : str   Input passed to the tool.
    observation  : str   Result returned by the tool.
    duration_s   : float Wall-clock time for this step in seconds.
    """
    iteration:    int
    thought:      str
    action:       str
    action_input: str
    observation:  str
    duration_s:   float


@dataclass
class AgentResult:
    """
    The complete result of running the agent on a question.

    Attributes
    ----------
    question     : str          The original question.
    final_answer : str          The agent's final answer (or timeout message).
    steps        : list[AgentStep]  Full trajectory of all steps taken.
    total_time_s : float        Total wall-clock time.
    stopped_by   : str          "final_answer" | "max_iterations" | "error"
    """
    question:     str
    final_answer: str
    steps:        list[AgentStep]
    total_time_s: float
    stopped_by:   str


def run_agent(
    question: str,
    tools: dict[str, Tool],
    llm_model,
    tokenizer,
    system_prompt: str,
    max_iterations: int = MAX_ITERATIONS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float  = TEMPERATURE,
    verbose: bool       = True,
) -> AgentResult:
    """
    Run the ReAct agent loop on a question.

    Parameters
    ----------
    question : str
        The user's question to answer.
    tools : dict[str, Tool]
        Registry of available tools keyed by name.
    llm_model : AutoModelForCausalLM
        The loaded language model.
    tokenizer : AutoTokenizer
        The tokenizer matching the language model.
    system_prompt : str
        The system prompt containing tool descriptions and format instructions.
    max_iterations : int  (default MAX_ITERATIONS)
        Maximum Observe→Think→Act cycles before forcing termination.
        Always set this — an unbounded loop is a bug waiting to happen.
    max_new_tokens : int  (default MAX_NEW_TOKENS)
        Tokens to generate per step. 256 covers Thought + Action + Input.
    temperature : float  (default TEMPERATURE)
        Sampling temperature. Keep low (0.0–0.2) for reliable format compliance.
    verbose : bool  (default True)
        If True, print each step as it executes.

    Returns
    -------
    AgentResult
        Full trajectory and final answer.
    """
    t_total_start = time.time()
    steps: list[AgentStep] = []

    # Build the initial message history in chat format
    # The full history is reformatted and retokenised on every step —
    # this is the "growing context" that makes the agent stateful.
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": question},
    ]

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  Question: {question}")
        print(f"  {'─'*60}")

    for iteration in range(1, max_iterations + 1):
        t_step_start = time.time()

        if verbose:
            print(f"\n  [Step {iteration}/{max_iterations}] Thinking...")

        # ── Format messages for the model ─────────────────
        # Qwen uses a chat template with special tokens.
        # apply_chat_template handles this correctly for instruction models.
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(DEVICE)

        # ── Generate next step ────────────────────────────
        with torch.no_grad():
            output_ids = llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Stop on "Observation:" — prevents the model from
                # hallucinating its own tool results
                stop_strings=STOP_SEQUENCES,
                tokenizer=tokenizer,
            )

        # Decode only the new tokens (skip the input prompt)
        n_input = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][n_input:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if verbose:
            print(f"  LLM output:\n{textwrap.indent(raw_output, '    ')}")

        # ── Parse the output ──────────────────────────────
        parsed = parse_llm_output(raw_output, set(tools.keys()))

        # ── Handle: Final Answer ──────────────────────────
        if parsed.final_answer:
            step = AgentStep(
                iteration=iteration,
                thought=parsed.thought,
                action="final_answer",
                action_input="",
                observation=parsed.final_answer,
                duration_s=time.time() - t_step_start,
            )
            steps.append(step)
            if verbose:
                print(f"\n  ✅ Final Answer: {parsed.final_answer}")
            return AgentResult(
                question=question,
                final_answer=parsed.final_answer,
                steps=steps,
                total_time_s=time.time() - t_total_start,
                stopped_by="final_answer",
            )

        # ── Handle: Tool Call ─────────────────────────────
        if parsed.action and parsed.action != "__unknown__":
            tool     = tools[parsed.action]
            obs_text = tool(parsed.action_input)
        elif parsed.action == "__unknown__":
            # Unknown tool — feed error back as observation
            obs_text = f"Error: {parsed.action_input}"
        else:
            # Malformed output — nudge the model
            obs_text = (
                "Your output was not in the correct format. "
                "Please respond with:\n"
                "Thought: <reasoning>\n"
                "Action: <tool name>\n"
                "Action Input: <input>"
            )
            parsed.action = "format_error"
            parsed.action_input = raw_output

        if verbose:
            action_display = parsed.action or "malformed"
            print(f"\n  Tool: {action_display}({parsed.action_input!r})")
            print(f"  Observation: {obs_text[:200]}")

        step = AgentStep(
            iteration=iteration,
            thought=parsed.thought,
            action=parsed.action or "malformed",
            action_input=parsed.action_input or "",
            observation=obs_text,
            duration_s=time.time() - t_step_start,
        )
        steps.append(step)

        # Append assistant output + observation to the message history.
        # The observation is labelled so the model knows it came from a tool.
        messages.append({"role": "assistant", "content": raw_output})
        messages.append({
            "role": "user",
            "content": f"Observation: {obs_text}\nContinue.",
        })

    # ── Max iterations hit ────────────────────────────────
    # Force the model to produce a best-effort answer from what it knows
    if verbose:
        print(f"\n  ⚠️  Max iterations ({max_iterations}) reached. Forcing final answer...")

    messages.append({
        "role": "user",
        "content": (
            "You have reached the maximum number of steps. "
            "Based on everything you have observed so far, provide your best Final Answer now.\n"
            "Final Answer:"
        ),
    })

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    n_input = inputs["input_ids"].shape[1]
    forced_answer = tokenizer.decode(output_ids[0][n_input:], skip_special_tokens=True).strip()
    forced_answer = re.sub(r"^Final Answer:\s*", "", forced_answer).strip()

    if verbose:
        print(f"  Forced answer: {forced_answer}")

    return AgentResult(
        question=question,
        final_answer=forced_answer or "(no answer produced)",
        steps=steps,
        total_time_s=time.time() - t_total_start,
        stopped_by="max_iterations",
    )


print(f"\n  Agent loop defined. LLM: {LLM_MODEL}")
print(f"  max_iterations={MAX_ITERATIONS}  max_new_tokens={MAX_NEW_TOKENS}  temp={TEMPERATURE}")
print()


# ─────────────────────────────────────────────────────
# SECTION 7: RUN THE MANUAL AGENT — SINGLE-TOOL QUESTIONS
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 7: Single-Tool Questions")
print("=" * 65)

single_tool_questions = [
    ("What is 2 to the power of 16?",                           "calculator"),
    ("What is the current US inflation rate?",                   "web_search"),
    ("Find me news articles about technology companies.",        "rag_retrieval"),
]

single_tool_results = []
for question, expected_tool in single_tool_questions:
    print(f"\n  Expected tool: {expected_tool}")
    result = run_agent(question, TOOLS, llm, tokenizer, SYSTEM_PROMPT, verbose=True)
    single_tool_results.append(result)

    # Print trajectory summary
    print(f"\n  Trajectory summary:")
    for step in result.steps:
        print(f"    Step {step.iteration}: {step.action}({step.action_input[:40]!r}) "
              f"→ {step.observation[:60]!r}  [{step.duration_s:.1f}s]")
    print(f"  Stopped by: {result.stopped_by}  |  Total: {result.total_time_s:.1f}s")

print()


# ─────────────────────────────────────────────────────
# SECTION 8: RUN THE MANUAL AGENT — MULTI-STEP QUESTIONS
# ─────────────────────────────────────────────────────
#
# These questions require the agent to use multiple tools across
# multiple iterations before it can produce a final answer.

print("=" * 65)
print("SECTION 8: Multi-Step Questions")
print("=" * 65)

multi_step_questions = [
    "If the US GDP growth rate last year was as reported, and the economy was worth "
    "$25 trillion, roughly how many trillions did it grow by? Show your calculation.",

    "Find a recent news article about a sports event, then tell me what percentage "
    "of 500 would represent a score of 42.",
]

multi_step_results = []
for question in multi_step_questions:
    print(f"\n  {'─'*60}")
    result = run_agent(question, TOOLS, llm, tokenizer, SYSTEM_PROMPT, verbose=True)
    multi_step_results.append(result)

    print(f"\n  Steps taken: {len(result.steps)}")
    print(f"  Tools used : {[s.action for s in result.steps]}")
    print(f"  Stopped by : {result.stopped_by}")
    print(f"  Total time : {result.total_time_s:.1f}s")

print()


# ─────────────────────────────────────────────────────
# SECTION 9: AGENT FAILURE MODES — DEMONSTRATED
# ─────────────────────────────────────────────────────
#
# We test the known failure cases and show how the loop handles them.
# Understanding these patterns is more valuable than a perfect demo run.

print("=" * 65)
print("SECTION 9: Agent Failure Modes")
print("=" * 65)

print("""
  The five key failure modes for ReAct agents, and how we handle each:

  ┌─────────────────────────────────┬──────────────────────────────────────┐
  │ Failure Mode                    │ How this implementation handles it   │
  ├─────────────────────────────────┼──────────────────────────────────────┤
  │ 1. Hallucinated tool result     │ "Observation:" in stop_strings stops │
  │    Model invents the observation│ generation before it can fabricate   │
  ├─────────────────────────────────┼──────────────────────────────────────┤
  │ 2. Wrong tool name              │ parse_llm_output checks name against │
  │    e.g. "search" not "web_search│ valid_tool_names; returns error obs  │
  ├─────────────────────────────────┼──────────────────────────────────────┤
  │ 3. Malformed output             │ Parser returns ParsedStep with both  │
  │    No Action or Final Answer    │ action=None; loop injects format hint│
  ├─────────────────────────────────┼──────────────────────────────────────┤
  │ 4. Infinite reasoning loop      │ max_iterations hard cap; forced      │
  │    Never commits to an answer   │ Final Answer prompt at the limit     │
  ├─────────────────────────────────┼──────────────────────────────────────┤
  │ 5. Tool execution error         │ Tool.__call__ wraps fn in try/except │
  │    e.g. division by zero        │ returns "Tool error: ..." string     │
  └─────────────────────────────────┴──────────────────────────────────────┘
""")

# Demonstrate calculator error handling live
print("  Live demo — calculator error handling:")
error_cases = [
    ("1 / 0",         "division by zero"),
    ("import os",     "unsafe AST node blocked"),
    ("sqrt(16)",      "unsupported function"),
]
for expr, label in error_cases:
    result = calculator(expr)
    print(f"  calculator({expr!r}) → {result!r}  ({label})")

# Demonstrate unknown tool handling
print("\n  Live demo — unknown tool handling:")
malformed = "Thought: Let me search.\nAction: google\nAction Input: AI news"
parsed = parse_llm_output(malformed, set(TOOLS.keys()))
print(f"  Input action: 'google'  (not in registry)")
print(f"  Parsed action: '{parsed.action}'")
print(f"  Error observation: '{parsed.action_input[:80]}'")
print()


# ─────────────────────────────────────────────────────
# SECTION 10: smolagents — SAME AGENT IN ~20 LINES
# ─────────────────────────────────────────────────────
#
# smolagents is HuggingFace's lightweight agent framework. It handles
# the prompt formatting, tool dispatch, and loop management that we
# wrote manually above. The same three tools, the same Qwen model —
# but expressed in a fraction of the code.

print("=" * 65)
print("SECTION 10: smolagents — Same Agent, Less Code")
print("=" * 65)

try:
    from smolagents import (
        CodeAgent,
        HfApiModel,
        tool as smolagents_tool,
        ToolCallingAgent,
        TransformersModel,
    )

    print("\n  smolagents imported successfully.")

    # ── Define tools with the @tool decorator ──────────
    # smolagents uses type annotations and docstrings as the tool
    # description — the same information we put in Tool.description,
    # but bound to a function signature rather than a separate string.

    @smolagents_tool
    def smol_calculator(expression: str) -> str:
        """
        Evaluates a mathematical expression and returns the numerical result.
        Input must be a valid arithmetic expression using +, -, *, /, **, %.
        Example: "2 ** 10" or "(100 - 32) * 5 / 9"

        Args:
            expression: A Python arithmetic expression as a string.
        """
        return _calculator_fn(expression)

    @smolagents_tool
    def smol_web_search(query: str) -> str:
        """
        Searches the web for current information and returns a result snippet.
        Use for questions about recent events, statistics, or general knowledge.
        Example: "US inflation rate 2024"

        Args:
            query: A natural language search query.
        """
        return _web_search_fn(query)

    @smolagents_tool
    def smol_rag_retrieval(query: str) -> str:
        """
        Searches a local knowledge base of news articles using semantic similarity.
        Use for questions about news events, sports, business, or technology.
        Example: "tech company earnings reports"

        Args:
            query: A natural language query describing what you are looking for.
        """
        return _rag_fn(query)

    # ── Create model wrapper ────────────────────────────
    # TransformersModel wraps our already-loaded Qwen model so
    # smolagents can use it without downloading it again.
    smol_model = TransformersModel(
        model_id=LLM_MODEL,
        model=llm,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # ── Create the agent ────────────────────────────────
    # ToolCallingAgent is smolagents' ReAct-style agent.
    # It handles the loop, prompt formatting, and tool dispatch.
    smol_agent = ToolCallingAgent(
        tools=[smol_calculator, smol_web_search, smol_rag_retrieval],
        model=smol_model,
        max_steps=MAX_ITERATIONS,
        verbosity_level=1,
    )

    print("\n  Running smolagents on the same questions:")
    smol_questions = [
        "What is 17 multiplied by 83?",
        "What is the latest news about Python programming language?",
    ]

    for question in smol_questions:
        print(f"\n  Question: {question}")
        try:
            answer = smol_agent.run(question)
            print(f"  Answer  : {answer}")
        except Exception as e:
            print(f"  (smolagents run error: {e} — this can happen with very small models)")

    print(f"""
  smolagents vs. manual loop comparison:
  ───────────────────────────────────────
  Manual loop (Section 6–9):
    • ~150 lines of code
    • Every parsing decision is explicit and inspectable
    • Full control over stop sequences, error handling, context management
    • Best for: learning, custom behaviour, non-standard formats

  smolagents (Section 10):
    • ~20 lines of code
    • Prompt formatting and parsing handled by the framework
    • Multi-agent (ManagedAgent) built in
    • Best for: prototyping quickly, standard ReAct tasks

  Rule of thumb: understand the manual loop first.
  Then use smolagents (or LangChain) to ship it.
""")

except ImportError:
    print("\n  smolagents not installed. Install with: pip install smolagents")
    print("  Showing the equivalent code structure instead:\n")
    print(textwrap.dedent("""
      from smolagents import ToolCallingAgent, TransformersModel, tool

      @tool
      def calculator(expression: str) -> str:
          \"\"\"Evaluates a math expression. Args: expression: arithmetic string.\"\"\"
          return eval_safe(expression)

      @tool
      def web_search(query: str) -> str:
          \"\"\"Searches the web. Args: query: search query string.\"\"\"
          return search_api(query)

      model = TransformersModel(model_id="Qwen/Qwen2.5-0.5B-Instruct")
      agent = ToolCallingAgent(tools=[calculator, web_search], model=model)
      answer = agent.run("What is 15% of 240?")
    """))

print()


# ─────────────────────────────────────────────────────
# SECTION 11: SUMMARY
# ─────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 11: Summary")
print("=" * 65)

all_results = single_tool_results + multi_step_results
total_steps  = sum(len(r.steps) for r in all_results)
total_time   = sum(r.total_time_s for r in all_results)
tool_usage   = {}
for r in all_results:
    for s in r.steps:
        tool_usage[s.action] = tool_usage.get(s.action, 0) + 1

print(f"""
  What ran:
    ✅ Defined 3 tools: calculator, web_search, rag_retrieval
    ✅ Built RAG knowledge base ({len(rag_docs)} AG News docs, {rag_embeddings.shape[1]}D)
    ✅ Built system prompt with ReAct format and tool descriptions
    ✅ Implemented output parser (Thought / Action / Final Answer)
    ✅ Ran full ReAct agent loop with {len(all_results)} questions
       → {total_steps} total steps across all runs
       → {total_time:.0f}s total agent time
    ✅ Demonstrated 5 failure modes and mitigations
    ✅ Showed smolagents equivalent

  Tool usage across all runs:
""")
for tool_name, count in sorted(tool_usage.items(), key=lambda x: -x[1]):
    bar = "█" * count
    print(f"    {tool_name:<20} {count:>3}x  {bar}")

print(f"""
  Key concepts covered:
    • ReAct loop: Observe → Think → Act
    • Tool dataclass: name, description, fn
    • System prompt design: format instructions + tool descriptions + example
    • Output parser: regex extraction, unknown tool handling, malformed output
    • Stop sequences: preventing hallucinated observations
    • max_iterations: bounding the loop; forced final answer at the limit
    • Context accumulation: full message history on every step
    • smolagents: @tool decorator, ToolCallingAgent, TransformersModel

  To go further:
    • Swap LLM_MODEL for Qwen2.5-1.5B or 7B for stronger tool-calling
    • Replace _web_search_fn with a real Serper/Brave API call
    • Add a memory tool: store/retrieve agent notes across turns
    • Implement context window management (summarisation after N steps)
    • Try multi-agent: one planner agent + specialised sub-agents
    • See Tutorial 03 (RAG) and Tutorial 05 (Embeddings) for deeper dives
      into the components powering the rag_retrieval tool
""")

print("=" * 65)
print("  Tutorial 06 — agents_tutorial.py complete.")
print("  That's all 6 tutorials. Well done! 🎉")
print("=" * 65)
