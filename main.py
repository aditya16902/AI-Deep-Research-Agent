import os
import getpass
from composio import Composio
from composio_langchain import LangchainProvider
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

import os
from composio import Composio



##Tools
# Text to list converter (Converts RQs into list)
@tool
def generate_questions_list(q: str) -> list:
    """Takes string q, with questions about the given topic in the specified domain. 
    Its formatted as a numbered list, and NOTHING ELSE. Then this function/ tool helps to Extract questions into a list."""

    questions_list = [question.strip() for question in q.split('\n') if question.strip()]
    return questions_list

#Tools from Composio for agent centric web-search (Tavily) and research (Perplexity) and document creation (Google docs)
composio = Composio(provider=LangchainProvider())
composio_tools = composio.tools.get(user_id="default", toolkits=["COMPOSIO_SEARCH_TAVILY_SEARCH", "PERPLEXITYAI_PERPLEXITY_AI_SEARCH", "GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN"])


# Child Agent 1 [Generates Research Questions]
research_subagent_1 = {
    "name": "research-question-agent",
    "description": "Used to generate research questions about the topic '{topic}' in the domain '{domain}'",
    "system_prompt": """
        You are an expert at breaking down research topics into specific questions.
        """,
    "tools": [*composio_tools, generate_questions_list]

}

# Child Agent 2 [research a specific question]
research_subagent_2 = {
    "name": "research-question-agent",
    "description": "Used to answer a research question",
    "system_prompt": """
        You are a sophisticated research assistant, skilled in researching about a particular research question related to a given domain
        """,
    "tools": [*composio_tools]
}

# Child Agent 3 [Compiling final report]
research_subagent_3 = {
    "name": "report-compiling-agent",
    "description": "Used to create professional, McKinsey-style report.",
    "system_prompt": f"""
            You are a sophisticated research assistant. skilled in creating research reports from existing research findings.
            """,
    "tools": [*composio_tools]
    
}


# mother agent
research_subagents = [research_subagent_1, research_subagent_2, research_subagent_3]
model = init_chat_model(model="openai:gpt-5-mini")
checkpointer = MemorySaver()
research_instructions1 = f"""You are a sophisticated head researcher. 

        1) Based on the users input ( - Topic: topic, Domain: domain), Generate exactly 5 specific yes/no research questions 
        about the given topic (use COMPOSIO_SEARCH_TAVILY_SEARCH, PERPLEXITYAI_PERPLEXITY_AI_SEARCH if needed) in the specified domain.
        Respond ONLY with the text of the 5 questions formatted as a numbered list, and NOTHING ELSE. 
        Then use generate_questions_list tool to Extract questions into a list.
        
        2)Now, Answer the each of the research questions about the topic 'topic' in the domain 'domain':\n\nquestion\n\nUse 
        the PERPLEXITYAI_PERPLEXITY_AI_SEARCH and COMPOSIO_SEARCH_TAVILY_SEARCH tools to provide a concise, well-sourced answer.

        3)Now, Compile the following research findings into a professional, McKinsey-style report. The report should be structured as follows:

            1. Executive Summary: Provide a concise, high-level overview of the topic, highlighting the most important insights and outcomes without repeating the introduction.
            2. Introduction: Briefly introduce the topic and domain, and summarize the key findings.
            3. Research Analysis: For each research question, create a section with a clear heading and provide a detailed, analytical answer. Do NOT use a Q&A format; instead, weave the answer into a narrative and analytical style.
            4. Conclusion/Implications: Summarize the overall insights and implications of the research.

            Use clear, structured HTML for the report.

            Topic: topic
            Domain: domain

            Research Questions and Findings (for your reference):
            qa_sections

            Use the GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN tool to create a Google Doc with the report. The text should be in HTML format. You have to create the google document with all the compiled info.

        !return output format ONLY as a dictionary, and NOTHING ELSE. with key being the research Questions, final report being keys and values being your Research Finding for respective RQs and Research report content (Use HTML content inside accordingly, so i can directly parse later accordingly on website).
 

        """
research_instructions2 = f"""
You are a senior, highly analytical head researcher.

Your task proceeds in **three strict phases**. Follow each phase exactly, and adhere to all formatting requirements.

---

### **PHASE 1 — Generate Research Questions**

1. Based on the user's input (**Topic: topic**, **Domain: domain**), generate **exactly 5 highly specific YES/NO research questions**.
2. These questions must be evaluative, research-ready, and directly answerable using available search tools  
   (**COMPOSIO_SEARCH_TAVILY_SEARCH**, **PERPLEXITYAI_PERPLEXITY_AI_SEARCH**).
3. Respond **ONLY** with the text of the 5 questions, formatted as a **numbered list (1-5)** and **nothing else**.
4. After producing the list, call the **generate_questions_list** tool to extract the questions into a list.

---

### **PHASE 2 — make a deep research about the Research Questions**

For each generated question:

1. Use **PERPLEXITYAI_PERPLEXITY_AI_SEARCH** and **COMPOSIO_SEARCH_TAVILY_SEARCH** to gather evidence.
2. Produce a **detiled, well-sourced analytical answer**.

---

### **PHASE 3 — Compile a McKinsey-Style Research Report**

Using the findings from Phase 2, compile a polished, consulting-grade report with the following structure. The report should be comprehensive and detailed—matching the depth and length typically produced by advanced LLM “deep research” workflows.


1. **Executive Summary**
    Provide a concise, high-level overview of the topic, highlighting the most important insights and outcomes without repeating the introduction.

2. **Introduction**  
   Briefly introduce the topic and domain, and summarize the key findings.

3. **Research Analysis**  
   For each research question, create a section with a clear subheading.  
   **Do NOT use a Q&A format.**  
   Instead, integrate the findings into a coherent analytical narrative for each section.

4. **Conclusion & Implications**  
   Summarize cross-cutting insights and outline implications for decision-makers.

Requirements:

- The entire report must be formatted in **clean, structured HTML**.

---

### **FINAL OUTPUT CONSTRAINTS**

Return the final output (report from PHASE 3) **ONLY as a HTML**, with no surrounding text.

Use this structure exactly so it can be directly parsed by the website.
"""



agent = create_deep_agent(model=model,subagents = research_subagents,
                            system_prompt = research_instructions2, tools=[generate_questions_list, *composio_tools],     
                            interrupt_on = {"generate_questions_list": {"allowed_decisions": ["approve", "edit"]}},
                            checkpointer=checkpointer)
