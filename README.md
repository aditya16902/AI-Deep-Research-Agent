# AI Deep Research Agent

LangChain Deep Agents + Composio powered research workflow that generates focused research questions, searches the web (Tavily + Perplexity), lets a human approve/edit the plan, and compiles a consulting-style HTML report (and Google Doc) with GPT-5 series models. Built to keep costs low while outperforming ChatGPT’s native deep research option on the same prompts when run with `gpt-5-nano`.

## Features
- Multi-agent pipeline: question generator, researcher, report writer.
- Human-in-the-loop gate to approve or edit the research questions before execution.
- Web/search tools via Composio (Tavily, Perplexity) plus Google Docs creation.
- Generates structured HTML reports and one-click Google Doc export.
- Lightweight Streamlit UI with executive-summary preview and local report viewer.
- Stateless setup beyond API keys; LangGraph memory checkpointing per session.

## Installation
With uv (recommended)
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

With pip
```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -e .
```

Requirements: Python 3.11+, OpenAI and Composio API keys (set in the UI or env vars).

## Usage
```bash
streamlit run app.py
```
1) Open the app, paste your Composio and OpenAI keys in the sidebar.  
2) Enter Topic + Domain, click “Research”.  
3) Review the generated questions (approve or edit).  
4) View the executive summary in-app; click “View full report” to open the HTML/Google Doc.

## How it solves deep research (with minimal cost)
- Decomposes the ask into 5 sharp yes/no questions to bound scope and redundancy.
- Uses Tavily for breadth/recency and Perplexity for high-signal synthesis, keeping answers concise, source-backed, and cheaper than naïve brute-force search.
- Runs with LangChain Deep Agents; swap the model to `gpt-5-nano` for strong quality/cost ratio (beats ChatGPT deep research on identical prompts in internal tests).
- Human checkpoint ensures bad question sets are fixed before spend.
- Outputs clean HTML that can be consumed directly or pushed to Google Docs.

## Architecture
- `main.py`: Defines sub-agents (question generator, researcher, report compiler), Composio tools, Deep Agent config, and interrupt/human-in-loop settings.
- `app.py`: Streamlit front-end, API key capture, session state, interrupt handling (approve/edit), report viewer/server, executive-summary extractor.
- Tools: `generate_questions_list` LangChain tool + Composio toolkits for search and Google Docs.
- Model: LangChain `init_chat_model` (OpenAI GPT-5 series by default).

## Supported models
- Defaults to OpenAI GPT-5-series (`openai:gpt-5` in code).  
- Recommended for cost: change the `init_chat_model` call in `main.py` to `model="openai:gpt-5-nano"` (or any LangChain-compatible chat model).
- Bring your own keys; no vendor lock-in in the code path.

## Human-in-the-loop
- The agent pauses after question generation with an interrupt.
- You can approve as-is or edit the numbered list before the research runs.
- Ensures relevance and reduces wasted tool/model calls.

## Technology stack
- LangChain Deep Agents, LangGraph (MemorySaver)
- Composio (Tavily search, Perplexity search, Google Docs)
- Streamlit UI
- Python 3.11+

## Limitations
- API rate limits and tool quotas (OpenAI, Tavily, Perplexity, Google) apply.
- Costs scale with the breadth of the topic and number of tool calls.
- Long/ambiguous topics can still yield shallow coverage; refine the prompt.
- No built-in persistence of past runs beyond the current session state.

## Tips for best results
- Be specific with Topic + Domain (e.g., “LLM eval methods” + “MLOps”).
- Let the tool finish after you approve/edit the questions; avoid repeated restarts.
- Prefer `gpt-5-nano` for low cost; switch to larger GPT-5 variants for maximal depth.
- Use the executive-summary preview to sanity-check before exporting.
- Keep the questions numbered 1–5 when editing to avoid errors.

## Contributing
Pull requests are welcome. Please open an issue for discussion before large changes.

## License
Apache 2.0 (see root `LICENSE`).

## Support
Open an issue in this repository or start a discussion with reproducible details and logs.
