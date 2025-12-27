import os
import sys
import uuid
import streamlit as st
from langgraph.types import Command

import http.server
import socketserver
import webbrowser
import threading
import time
from html.parser import HTMLParser


REPORT_PORT = 8000
REPORT_FILE = "temp.html"

from html.parser import HTMLParser

class ExecSummaryParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_exec_section = False
        self.section_depth = 0
        self.buffer = []
        self.exec_html = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "section" and attrs_dict.get("id") == "executive-summary":
            self.in_exec_section = True
            self.section_depth = 1
            self.buffer.append("<section id=\"executive-summary\">")
            return

        if self.in_exec_section:
            if tag == "section":
                self.section_depth += 1
            attrs_str = " ".join(f'{k}="{v}"' for k, v in attrs)
            self.buffer.append(f"<{tag}{(' ' + attrs_str) if attrs_str else ''}>")

    def handle_endtag(self, tag):
        if self.in_exec_section:
            self.buffer.append(f"</{tag}>")
            if tag == "section":
                self.section_depth -= 1
                if self.section_depth == 0:
                    self.in_exec_section = False
                    self.exec_html = "".join(self.buffer)

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self.in_exec_section:
            self.buffer.append(text)


def extract_executive_summary(html: str) -> str:
    parser = ExecSummaryParser()
    parser.feed(html)
    return parser.exec_html or ""




# ---------- HTTP server to serve temp.html ----------
def start_report_server():
    """Start a simple HTTP server in a background thread and open temp.html."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=".", **kwargs)

    # allow port reuse
    class ReuseTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    try:
        httpd = ReuseTCPServer(("", REPORT_PORT), Handler)
    except OSError:
        # Port already in use; just open the URL and return
        webbrowser.open(f"http://localhost:{REPORT_PORT}/{REPORT_FILE}")
        return None

    def open_browser():
        time.sleep(1)
        webbrowser.open(f"http://localhost:{REPORT_PORT}/{REPORT_FILE}")

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    return httpd


st.set_page_config(page_title="AI Domain Deep Research Agent", layout="wide")


# -----------------------------
# Sidebar: API keys
# -----------------------------
def setup_api_keys():
    with st.sidebar:
        st.header("🔑 API Keys")

        composio_key = st.text_input(
            "Composio API Key",
            type="password",
            help="Used for Tavily / Perplexity / Google Docs tools",
            key="composio_key",
        )
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Used for GPT-5 model",
            key="openai_key",
        )

        if composio_key:
            os.environ["COMPOSIO_API_KEY"] = composio_key
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        ready = bool(
            os.environ.get("COMPOSIO_API_KEY")
            and os.environ.get("OPENAI_API_KEY")
        )

        if not ready:
            st.warning("Enter both API keys above to enable the agent.")
        else:
            st.success("API keys set. You can use the app now.")

        return ready


api_ready = setup_api_keys()
if not api_ready:
    st.title("AI Domain Deep Research Agent")
    st.write("Enter your **Composio** and **OpenAI** API keys in the sidebar to start.")
    st.stop()


# -----------------------------
# Import agent after keys set
# -----------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import agent  # noqa: E402


# -----------------------------
# Session state initialization
# -----------------------------
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

if "state" not in st.session_state:
    st.session_state.state = "input"  # input | interrupt | edit | completed

if "interrupt_q" not in st.session_state:
    st.session_state.interrupt_q = ""

if "tool_name" not in st.session_state:
    st.session_state.tool_name = None

if "html_content" not in st.session_state:
    st.session_state.html_content = ""

if "report_server" not in st.session_state:
    st.session_state.report_server = None  # TCPServer instance if running

config = st.session_state.config


# -----------------------------
# Helper: run initial agent call
# -----------------------------
def run_agent(topic: str, domain: str):
    messages = [
        {
            "role": "user",
            "content": f"Topic: {topic}, Domain: {domain}",
        }
    ]
    result = agent.invoke({"messages": messages}, config=config)

    if result.get("__interrupt__"):
        interrupts = result["__interrupt__"][0].value
        action_requests = interrupts["action_requests"]
        action = action_requests[0]
        st.session_state.interrupt_q = action["args"]["q"]
        st.session_state.tool_name = action["name"]
        st.session_state.state = "interrupt"
    else:
        st.session_state.html_content = result["messages"][-1].content
        st.session_state.state = "completed"


# -----------------------------
# Main state machine
# -----------------------------
if st.session_state.state == "input":
    st.title("AI Domain Deep Research Agent")

    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Topic", value="History of Reinforcement Learning")
    with col2:
        domain = st.text_input("Domain", value="Technology")

    if st.button("🚀 Research", type="primary"):
        if not topic.strip() or not domain.strip():
            st.error("Please enter both topic and domain.")
        else:
            with st.spinner("Running deep research agent..."):
                run_agent(topic.strip(), domain.strip())
            st.rerun()

elif st.session_state.state == "interrupt":
    st.title("👀 Review Research Questions (Human-in-the-loop)")

    q_raw = st.session_state.interrupt_q or ""
    q_display = q_raw.replace("\\n", "\n")

    st.text_area(
        "Generated research questions (read-only):",
        value=q_display,
        height=250,
        disabled=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", type="primary", use_container_width=True):
            decisions = [{"type": "approve"}]
            with st.spinner("Continuing research with approved questions..."):
                result = agent.invoke(
                    Command(resume={"decisions": decisions}),
                    config=config,
                )
            st.session_state.html_content = result["messages"][-1].content
            st.session_state.state = "completed"
            st.rerun()

    with col2:
        if st.button("✏️ Edit questions", use_container_width=True):
            st.session_state.state = "edit"
            st.rerun()

elif st.session_state.state == "edit":
    st.title("✏️ Edit Research Questions")

    q_raw = st.session_state.interrupt_q or ""
    q_display = q_raw.replace("\\n", "\n")

    edited = st.text_area(
        "Edit questions (keep them numbered 1–5):",
        value=q_display,
        height=300,
        key="edited_q",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Submit edited", type="primary", use_container_width=True):
            edited_for_agent = edited.replace("\r\n", "\n").replace("\n", "\\n")

            decisions = [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": st.session_state.tool_name or "generate_questions_list",
                        "args": {"q": edited_for_agent},
                    },
                }
            ]

            with st.spinner("Continuing research with edited questions..."):
                result = agent.invoke(
                    Command(resume={"decisions": decisions}),
                    config=config,
                )

            st.session_state.html_content = result["messages"][-1].content
            st.session_state.state = "completed"
            st.rerun()

    with col2:
        if st.button("⬅️ Back to review", use_container_width=True):
            st.session_state.state = "interrupt"
            st.rerun()

elif st.session_state.state == "completed":
    html_content = st.session_state.html_content or ""
    exec_html = extract_executive_summary(html_content) or html_content

    st.title("✅ Research Completed")

    # Show Executive Summary first
    st.markdown("### Executive Summary")
    st.markdown(exec_html, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📄 View full report", type="primary", use_container_width=True):
            # Write HTML to temp.html
            temp_path = os.path.join(os.path.dirname(__file__), REPORT_FILE)
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            # If a previous server exists, shut it down first
            if st.session_state.report_server is not None:
                try:
                    st.session_state.report_server.shutdown()
                    st.session_state.report_server.server_close()
                except Exception:
                    pass
                st.session_state.report_server = None

            # Start new HTTP server + open browser
            server = start_report_server()
            if server is not None:
                st.session_state.report_server = server
                st.success(
                    f"Report server running at http://localhost:{REPORT_PORT}/{REPORT_FILE}"
                )

        if st.button("🔄 Restart", use_container_width=True):
            # Reset everything except API keys
            keep_keys = {
                "composio_key": st.session_state.get("composio_key"),
                "openai_key": st.session_state.get("openai_key"),
            }
            st.session_state.clear()
            for k, v in keep_keys.items():
                st.session_state[k] = v
            st.session_state.state = "input"
            st.rerun()


# -----------------------------
# Sidebar help text
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.header("📋 How to use")
    st.markdown(
        """
1. Enter **Composio** and **OpenAI** API keys.
2. Provide **Topic** and **Domain**.
3. Click **Research**.
4. When interrupted, **Approve** or **Edit** the questions.
5. After completion, view the **Executive Summary**.
6. Click **View full report** to open the complete HTML in a new tab.
7. Use **Restart** to begin a new session.
"""
    )

