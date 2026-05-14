# ⚡ InsightFlow AI: Enterprise Retail Analytics Copilot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-AI_Orchestration-orange?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-MCP_Server-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?style=for-the-badge&logo=postgresql)

**InsightFlow AI** is a production-grade Conversational Business Intelligence (BI) Copilot designed to translate natural language questions into governed SQL queries, deterministic statistical analysis, and executive-ready visual dashboards.

This project demonstrates advanced AI engineering patterns, moving beyond simple "Text-to-SQL" to implement a robust, highly secure, and mathematically deterministic enterprise analytics architecture.

---

## 🚀 Key Architectural Highlights

### 1. Advanced Agentic Workflow (LangGraph)
- **Autonomous Multi-Stage Pipeline**: Utilizes LangGraph to orchestrate a complex pipeline: Prompt Validation $\rightarrow$ Intent Classification $\rightarrow$ Schema Inspection $\rightarrow$ SQL Generation $\rightarrow$ Validation $\rightarrow$ Execution $\rightarrow$ Insight Generation.
- **Self-Correction & Reflection**: Incorporates an automated reflection loop. If the generated SQL fails execution, the database error is fed back to the LLM to autonomously debug and rewrite the query (with strict retry limits to prevent infinite loops).
- **Query Intent Classification**: Intelligently distinguishes between clear queries ("Top payment methods") and ambiguous ones ("Tell me about sales"), proactively asking the user for clarification when required temporal or regional granularity is missing.

### 2. Deterministic Statistical Intelligence Layer
- **Zero-Hallucination Math**: Unlike standard AI pipelines that ask the LLM to calculate percentages or averages (often leading to hallucinations), this engine intercepts the SQL results and computes a **Deterministic Statistical Context** in pure Python using Pandas.
- **Automated KPI Extraction**: Calculates concentration risks, standard deviations, ranking gaps, and distribution skewness mathematically.
- **Narrative Grounding**: The LLM acts strictly as a storyteller, injected with a pre-computed statistical context to guarantee 100% mathematical accuracy in the final executive summaries.

### 3. Secure Execution via MCP (Model Context Protocol)
- **Decoupled Architecture**: SQL execution and schema retrieval are handled by a dedicated FastAPI MCP server.
- **Security & Governance**: Employs Prompt Guarding to block prompt injection and restricts the LLM to read-only (`SELECT`) business intelligence queries. No raw UUIDs or PII are exposed to the LLM.

### 4. Enterprise-Grade Visual Analytics (Streamlit + Plotly)
- **Intelligent Chart Routing**: Automatically detects the shape of the data (time-series, ranking, categorical distribution, multi-metric) and dynamically selects the optimal enterprise visualization (Combo charts, Splines, Horizontal Bars, Donuts).
- **Premium UI/UX**: Features a Tableau Pulse / Power BI Copilot aesthetic with glassmorphism KPI cards, custom CSS styling, dynamic annotations pointing out outliers/top-performers, and rich dual-axis tooltips.
- **Interactive Controls**: Users can dynamically override chart types, adjust Top N limits, and multi-select metrics directly from the generated dashboard without re-triggering the LLM.

---

## 🏗️ System Architecture

```text
User Question 
 └──> Streamlit UI 
       └──> Prompt Security Guard
             └──> LangGraph Orchestrator
                   ├──> Intent Classification (LLM)
                   ├──> Schema Retrieval (MCP Server -> PostgreSQL)
                   ├──> SQL Generation (LLM + RAG Context)
                   ├──> SQL Execution (MCP Server -> PostgreSQL)
                   ├──> Deterministic Intelligence Layer (Pandas)
                   └──> Executive Insight Generation (LLM)
                         └──> Visual Analytics Engine (Plotly)
```

---

## 🛠️ Technology Stack

- **AI/LLM**: LangChain, LangGraph, OpenAI (`gpt-4o-mini`)
- **Backend**: FastAPI (MCP Server), Uvicorn
- **Data Engineering**: Python, Pandas, Numpy
- **Database**: PostgreSQL
- **Frontend**: Streamlit, Plotly Express & Graph Objects

---

## ⚙️ Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/retail-ai-agent.git
   cd retail-ai-agent
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_api_key
   OPENAI_MODEL=gpt-4o-mini
   DATABASE_URL=postgresql://user:password@localhost:5432/retail_db
   MCP_SERVER_URL=http://localhost:8000
   ```

4. **Run the MCP Backend Server:**
   ```bash
   uvicorn mcp_server.server:app --reload --port 8000
   ```

5. **Run the Streamlit Frontend:**
   ```bash
   streamlit run frontend/app.py
   ```

---

## 👤 About the Author
I am a passionate Software & AI Engineer actively seeking full-time opportunities. I specialize in bridging the gap between cutting-edge LLM capabilities and reliable, secure, enterprise-grade software architectures. 

Feel free to connect with me on LinkedIn or reach out via email!

*Note: The datasets used in this repository are ignored via `.gitignore` for security and compliance purposes.*
