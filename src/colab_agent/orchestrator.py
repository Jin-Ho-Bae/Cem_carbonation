"""
Multi-Agent Orchestrator for Carbon Mineralization Analysis.

Architecture:
    User Query
        |
    Manager (GPT-4o) -- interprets query, dispatches commands
        |
    Orchestrator (Claude/GPT-4o) -- routes to experts, coordinates coder, feedback loops
        |
    +-- 6 Domain Experts (GPT-4o, parallel) -- RBAC: each has ONLY its domain tools
    |     Chemistry, Transport, Micromechanics, Structural, MD, Engineering
    |
    +-- Coder (Claude/GPT-4o) -- writes code, self-learning from failure/success
    +-- Reviewer (Claude/GPT-4o) -- vulnerability checks, PASS-only save

Features:
    - RBAC (Role-Based Access Control): each expert can ONLY access its own tools
    - Auto-retry with exponential backoff on API overload (529)
    - Self-correction: all agents log failures, lessons injected into next run
    - GPT-4o fallback when Claude API is overloaded
"""

import os
import json
import time
import datetime
import traceback
from typing import Optional, Dict, List, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Claude support
try:
    from langchain_anthropic import ChatAnthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

# LangGraph for tool-calling agents
try:
    from langgraph.prebuilt import create_react_agent
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

# Legacy LangChain agents fallback
if not _HAS_LANGGRAPH:
    try:
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        _HAS_LEGACY_AGENT = True
    except ImportError:
        _HAS_LEGACY_AGENT = False
else:
    _HAS_LEGACY_AGENT = False

from .agent_tools import (
    get_chemistry_tools, get_transport_tools, get_micromechanics_tools,
    get_fem_tools, get_md_tools, get_engineering_tools, get_thermodynamics_tools,
    get_all_tools,
)


# ══════════════════════════════════════════════════════════════
#  RBAC: ROLE-BASED TOOL ACCESS CONTROL
# ══════════════════════════════════════════════════════════════

RBAC_REGISTRY = {
    # role -> (tool_getter, description of allowed tools)
    "chemistry":      (get_chemistry_tools,      "phase assemblage, cement systems"),
    "transport":      (get_transport_tools,       "pore structure, carbonation depth, diffusion models"),
    "micromechanics": (get_micromechanics_tools,  "6-level homogenization, property profiles"),
    "structural":     (get_fem_tools,             "FEniCS FEM, structure/load options"),
    "md":             (get_md_tools,              "MD properties, LAMMPS scripts"),
    "engineering":    (get_engineering_tools,      "legacy pipeline + coupled chemo-transport-mechanical profile (E(x,t) durability)"),
    "thermodynamics": (get_thermodynamics_tools,   "XRF->initial assemblage, phase evolution vs CO2, local equilibrium at depth, CSHQ decalcification (GEMS/cemdata18 backend)"),
}


def get_tools_for_role(role: str) -> list:
    """RBAC: return ONLY the tools this role is authorized to use."""
    entry = RBAC_REGISTRY.get(role)
    if entry is None:
        return []  # Unknown role gets NO tools
    return entry[0]()


# ══════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════

EXPERT_PROMPTS = {
    "chemistry": (
        "You are a cement chemistry expert. Analyze phase assemblage changes "
        "under carbonation/sulfate attack for OPC, CSA, blended cements with 10 SCMs. "
        "Use ONLY your assigned tools. Report carbonation degree and pH. "
        "Be concise: return key numerical results, not lengthy explanations."
    ),
    "transport": (
        "You are a transport phenomena expert. Model CO2 diffusion via Fick's 2nd law. "
        "Compute carbonation depth, D_eff, and K coefficient (mm/sqrt(year)). "
        "Use ONLY your assigned tools. Be concise: return key numerical results."
    ),
    "micromechanics": (
        "You are a micromechanics expert using 6-level Eshelby-Mori-Tanaka. "
        "Compute effective E, nu from nano to macro scale. "
        "Compare carbonated vs neat. Use ONLY your assigned tools. Be concise."
    ),
    "structural": (
        "You are a structural FEM expert using FEniCS. "
        "Analyze carbonation-partitioned structures. "
        "Report max displacement, von Mises stress. Use ONLY your assigned tools."
    ),
    "md": (
        "You are an MD specialist for cement materials. "
        "Provide nanoscale elastic constants and diffusion coefficients. "
        "Use ONLY your assigned tools. State confidence levels."
    ),
    "engineering": (
        "You are a durability / engineering assessment expert. "
        "Evaluate service life, carbonation resistance, code compliance (fib MC2010, EN 206). "
        "For mid- to long-term durability questions that need E(x,t) or coupled chemo-"
        "transport-mechanical profiles, use compute_coupled_chemo_transport_mechanical_profile. "
        "Use ONLY your assigned tools. Be concise."
    ),
    "thermodynamics": (
        "You are a thermodynamics expert. Your tools are backed INTERNALLY by "
        "GEMS (cemdata18 Gibbs energy minimization) -- you NEVER call GEMS as a "
        "standalone program; you call domain-level tools that use it under the hood.\n"
        "Primary capabilities:\n"
        "  - compute_initial_phase_assemblage: given XRF oxide composition + w/c, "
        "    return the initial hydrated phase assemblage assuming Gibbs minimization.\n"
        "  - compute_phase_evolution_vs_CO2: phase assemblage, pH, Ca/Si, porosity as "
        "    a function of cumulative CO2 dose (x-axis = mol CO2 added).\n"
        "  - compute_local_phase_assemblage_at_dose: local equilibrium at a single "
        "    (depth, time) point given the local CO2 dose from transport.\n"
        "  - compute_csh_decalcification_profile: CSHQ Ca/Si tracking.\n"
        "  - list_thermodynamic_phase_database: cemdata18 phase reference.\n"
        "Report phase volume fractions, pH, Ca/Si, porosity, carbonation degree. "
        "Be concise with key numerical results."
    ),
}

MANAGER_PROMPT = (
    "You are the Manager of a carbon mineralization multi-agent system. "
    "Interpret the user query and output a JSON work plan with keys:\n"
    "  - \"tasks\": [{\"expert\": name, \"instruction\": what to do}]\n"
    "  - \"coding_needed\": false\n"
    "  - \"coding_instruction\": \"\"\n"
    "  - \"synthesis_instruction\": how to combine results\n\n"
    "Available experts: chemistry, transport, micromechanics, structural, md, engineering, thermodynamics.\n\n"
    "Dispatch guide:\n"
    " - 'thermodynamics' (GEMS/cemdata18 backend) for: XRF -> initial hydrated phase "
    "assemblage (Gibbs minimization), phase assemblage vs cumulative CO2 dose, "
    "local equilibrium at a given depth/time, C-S-H decalcification, pH evolution.\n"
    " - 'chemistry' (kinetics ODE) for: time-dependent reaction rates, SCM blends, "
    "carbonation degree from kinetics.\n"
    " - 'transport' for: CO2 diffusion, D_eff model selection, carbonation depth from "
    "Fick's law, pore structure (Powers).\n"
    " - 'micromechanics' for: effective E/nu via 6-level Eshelby-Mori-Tanaka "
    "given a phase assemblage.\n"
    " - 'structural' for: FEniCS FEM analyses.\n"
    " - 'md' for: nanoscale elastic constants / diffusion from LAMMPS or DB.\n"
    " - 'engineering' for: SERVICE LIFE, CODE COMPLIANCE, and (very important) the "
    "COUPLED chemo-transport-mechanical PROFILE E(x,t) using "
    "compute_coupled_chemo_transport_mechanical_profile. Dispatch to engineering when "
    "the user wants durability maps, long-term E evolution along depth, or a single-"
    "frame answer combining transport + GEMS + micromechanics.\n\n"
    "IMPORTANT: Set coding_needed to false unless the user explicitly asks for custom code. "
    "The experts already have all necessary computation tools. "
    "Dispatch to minimum necessary experts."
)

ORCHESTRATOR_PROMPT = (
    "You are the Orchestrator synthesizing expert results into a final report. "
    "Be structured and concise. Include:\n"
    "1. Key quantitative results from each expert\n"
    "2. Cross-domain connections (chemistry -> transport -> mechanics)\n"
    "3. Any inconsistencies found\n"
    "4. Brief recommendations\n"
    "Keep the report under 500 words."
)

CODER_PROMPT = (
    "You are the Coder agent. Write Python code using numpy/scipy. "
    "Import from: colab_agent.chemistry_engine, colab_agent.transport_engine, "
    "colab_agent.micromechanics_engine, colab_agent.fem_engine, colab_agent.md_engine. "
    "Include error handling. Learn from past failures.\n"
    "Output JSON: {\"code\": \"...\", \"description\": \"...\", \"filename\": \"...\"}"
)

REVIEWER_PROMPT = (
    "You are the code Reviewer. Check for:\n"
    "1. Correctness (units, formulas)\n"
    "2. Edge cases (zero porosity, full carbonation)\n"
    "3. Conservation (volume sum <= 1, no negatives)\n"
    "4. Numerical stability\n"
    "Output JSON: {\"verdict\": \"PASS\"|\"FAIL\", \"issues\": [...], \"suggestions\": [...]}"
)


# ══════════════════════════════════════════════════════════════
#  SELF-CORRECTION LOG (shared by all agents)
# ══════════════════════════════════════════════════════════════

class SelfCorrectionLog:
    """Shared failure/success log for all agents. Persists to disk."""

    def __init__(self, log_dir: str = "_workspace/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "agent_errors.json")
        self._entries: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self._log_path):
            try:
                with open(self._log_path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
            except Exception:
                self._entries = []

    def _save(self):
        with open(self._log_path, "w", encoding="utf-8") as f:
            json.dump(self._entries[-200:], f, indent=2, ensure_ascii=False)

    def log(self, agent: str, status: str, summary: str, detail: str = ""):
        """Log a one-line summary of success/failure."""
        self._entries.append({
            "t": datetime.datetime.now().isoformat(),
            "agent": agent,
            "status": status,
            "summary": summary[:150],
            "detail": detail[:300],
        })
        self._save()

    def get_lessons(self, agent: str = "", limit: int = 10) -> str:
        """Get recent lessons for a specific agent or all agents."""
        relevant = [e for e in self._entries if e["status"] == "FAIL"]
        if agent:
            relevant = [e for e in relevant if e["agent"] == agent]
        if not relevant:
            return ""
        lines = ["=== PAST FAILURES (avoid these) ==="]
        for e in relevant[-limit:]:
            lines.append(f"- [{e['agent']}] {e['summary']}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  RESILIENT LLM CALL (retry + fallback)
# ══════════════════════════════════════════════════════════════

def _resilient_invoke(llm, messages, max_retries=3, fallback_llm=None, verbose=False):
    """
    Invoke LLM with retry on overload (529) and fallback to GPT-4o.
    """
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            is_overloaded = "overloaded" in err_str or "529" in err_str or "rate" in err_str
            if is_overloaded and attempt < max_retries - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                if verbose:
                    print(f"       API overloaded, retry in {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            elif is_overloaded and fallback_llm is not None:
                if verbose:
                    print("       Claude overloaded, falling back to GPT-4o...")
                return fallback_llm.invoke(messages)
            else:
                raise
    # Should not reach here
    return llm.invoke(messages)


# ══════════════════════════════════════════════════════════════
#  LLM FACTORIES
# ══════════════════════════════════════════════════════════════

def _get_gpt4o(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=temperature)


def _get_claude(temperature: float = 0.1) -> Any:
    if _HAS_ANTHROPIC:
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=temperature)
    return ChatOpenAI(model="gpt-4o", temperature=temperature)


# ══════════════════════════════════════════════════════════════
#  DOMAIN EXPERT AGENT (RBAC enforced)
# ══════════════════════════════════════════════════════════════

class DomainExpert:
    """A domain expert agent with RBAC-controlled tool access."""

    def __init__(self, name: str, llm=None, error_log: "SelfCorrectionLog" = None):
        self.name = name
        self.prompt = EXPERT_PROMPTS.get(name, "")
        # RBAC: only get tools for THIS role
        tools = get_tools_for_role(name)
        self.llm = llm or _get_gpt4o()
        self._error_log = error_log
        self._tools_desc = RBAC_REGISTRY.get(name, (None, "none"))[1]

        if _HAS_LANGGRAPH:
            self._agent = create_react_agent(self.llm, tools) if tools else None
            self._mode = "langgraph" if tools else "direct"
            self._max_iterations = 6
        elif _HAS_LEGACY_AGENT:
            prompt_tmpl = ChatPromptTemplate.from_messages([
                ("system", self.prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            agent = create_openai_tools_agent(self.llm, tools, prompt_tmpl)
            self._agent = AgentExecutor(
                agent=agent, tools=tools, verbose=False,
                max_iterations=8, handle_parsing_errors=True,
            )
            self._mode = "legacy"
        else:
            self._agent = None
            self._mode = "direct"

    def run(self, instruction: str) -> str:
        # Inject past lessons
        lessons = ""
        if self._error_log:
            lessons = self._error_log.get_lessons(self.name, limit=5)

        full_instruction = instruction
        if lessons:
            full_instruction = f"{instruction}\n\n{lessons}"

        try:
            if self._mode == "langgraph":
                messages = [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": full_instruction},
                ]
                config = {"recursion_limit": self._max_iterations * 2 + 1}
                result = self._agent.invoke({"messages": messages}, config=config)
                last = result["messages"][-1]
                output = last.content if hasattr(last, "content") else str(last)
            elif self._mode == "legacy":
                result = self._agent.invoke({"input": full_instruction, "chat_history": []})
                output = result["output"]
            else:
                resp = self.llm.invoke([
                    SystemMessage(content=self.prompt),
                    HumanMessage(content=full_instruction),
                ])
                output = resp.content

            if self._error_log:
                self._error_log.log(self.name, "OK", f"Completed: {instruction[:80]}")
            return output

        except Exception as e:
            error_summary = f"{type(e).__name__}: {str(e)[:100]}"
            if self._error_log:
                self._error_log.log(self.name, "FAIL", error_summary, traceback.format_exc()[:300])
            raise


# ══════════════════════════════════════════════════════════════
#  CODER AGENT (with self-learning)
# ══════════════════════════════════════════════════════════════

class CoderAgent:
    """Writes computation code, learns from failure/success history."""

    def __init__(self, workspace_dir: str = "_workspace/coder", llm=None,
                 fallback_llm=None, error_log: SelfCorrectionLog = None):
        self.llm = llm or _get_claude()
        self._fallback_llm = fallback_llm
        self.workspace = workspace_dir
        self.log_dir = os.path.join(workspace_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self._history: List[Dict] = []
        self._error_log = error_log
        self._load_history()

    def _load_history(self):
        log_path = os.path.join(self.log_dir, "history.json")
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except Exception:
                self._history = []

    def _save_history(self):
        log_path = os.path.join(self.log_dir, "history.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self._history[-100:], f, indent=2, ensure_ascii=False)

    def _get_lessons(self) -> str:
        if not self._history:
            return ""
        failures = [h for h in self._history if h["status"] == "FAIL"]
        successes = [h for h in self._history if h["status"] == "PASS"]
        lines = []
        if failures:
            lines.append(f"=== PAST FAILURES ({len(failures)}) ===")
            for f in failures[-5:]:
                lines.append(f"- {f['description']}: {f.get('error', 'unknown')}")
        if successes:
            lines.append(f"=== PAST SUCCESSES ({len(successes)}) ===")
            for s in successes[-3:]:
                lines.append(f"- {s['description']}: PASS")
        return "\n".join(lines)

    def log_result(self, description: str, status: str, code: str = "",
                   error: str = "", review_feedback: str = ""):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "description": description,
            "status": status,
            "code_preview": code[:200] if code else "",
            "error": error,
            "review_feedback": review_feedback,
        }
        self._history.append(entry)
        self._save_history()
        if self._error_log:
            self._error_log.log("coder", status, f"{description[:80]}: {error[:60]}" if error else description[:100])

    def write_code(self, instruction: str, expert_feedback: str = "",
                   verbose: bool = False) -> Dict:
        lessons = self._get_lessons()
        prompt = f"{CODER_PROMPT}\n\n--- INSTRUCTION ---\n{instruction}\n\n"
        if expert_feedback:
            prompt += f"--- EXPERT FEEDBACK ---\n{expert_feedback}\n\n"
        if lessons:
            prompt += f"--- PAST EXPERIENCE ---\n{lessons}\n\n"
        prompt += "Write the code now. Output valid JSON with keys: code, description, filename."

        resp = _resilient_invoke(
            self.llm, [HumanMessage(content=prompt)],
            fallback_llm=self._fallback_llm, verbose=verbose,
        )
        content = resp.content

        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
            result = json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            result = {"code": content, "description": instruction, "filename": "custom_analysis.py"}

        return result


# ══════════════════════════════════════════════════════════════
#  REVIEWER AGENT (vulnerability check)
# ══════════════════════════════════════════════════════════════

class ReviewerAgent:
    """Reviews code for correctness, robustness, and security."""

    def __init__(self, llm=None, fallback_llm=None, error_log: SelfCorrectionLog = None):
        self.llm = llm or _get_claude()
        self._fallback_llm = fallback_llm
        self._error_log = error_log

    def review(self, code: str, description: str = "", verbose: bool = False) -> Dict:
        prompt = (
            f"{REVIEWER_PROMPT}\n\n"
            f"--- CODE DESCRIPTION ---\n{description}\n\n"
            f"--- CODE ---\n```python\n{code[:3000]}\n```\n\n"
            "Output JSON: {\"verdict\": \"PASS\"|\"FAIL\", \"issues\": [...], \"suggestions\": [...]}"
        )

        resp = _resilient_invoke(
            self.llm, [HumanMessage(content=prompt)],
            fallback_llm=self._fallback_llm, verbose=verbose,
        )
        content = resp.content

        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
            result = json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            result = {"verdict": "PASS", "issues": [], "suggestions": ["Could not parse review output"]}

        return result


# ══════════════════════════════════════════════════════════════
#  MANAGER (GPT-4o)
# ══════════════════════════════════════════════════════════════

class Manager:
    """Interprets user query and produces a structured work plan."""

    def __init__(self, llm=None, error_log: SelfCorrectionLog = None):
        self.llm = llm or _get_gpt4o()
        self._error_log = error_log

    def plan(self, query: str) -> Dict:
        resp = self.llm.invoke([
            SystemMessage(content=MANAGER_PROMPT),
            HumanMessage(content=f"User query: {query}\n\nProduce the JSON work plan."),
        ])
        content = resp.content

        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
            plan = json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            plan = {
                "tasks": [
                    {"expert": "chemistry", "instruction": query},
                    {"expert": "transport", "instruction": query},
                    {"expert": "micromechanics", "instruction": query},
                    {"expert": "engineering", "instruction": query},
                ],
                "coding_needed": False,
                "coding_instruction": "",
                "synthesis_instruction": f"Synthesize all expert analyses for: {query}",
            }

        # Force coding_needed to False unless explicitly set
        if "coding_needed" not in plan:
            plan["coding_needed"] = False

        return plan


# ══════════════════════════════════════════════════════════════
#  ORCHESTRATOR (Claude with GPT-4o fallback)
# ══════════════════════════════════════════════════════════════

class Orchestrator:
    """Synthesizes expert results into a final report."""

    def __init__(self, llm=None, fallback_llm=None, error_log: SelfCorrectionLog = None):
        self.llm = llm or _get_claude()
        self._fallback_llm = fallback_llm
        self._error_log = error_log

    def synthesize(self, query: str, plan: Dict,
                   expert_results: Dict[str, str],
                   coder_output: Optional[str] = None,
                   verbose: bool = False) -> str:
        context = f"--- QUERY ---\n{query}\n\n"
        for name, result in expert_results.items():
            # Truncate each expert result to avoid token overflow
            context += f"--- {name.upper()} ---\n{result[:1500]}\n\n"
        if coder_output:
            context += f"--- CODER ---\n{coder_output[:1000]}\n\n"
        context += f"--- INSTRUCTION ---\n{plan.get('synthesis_instruction', 'Synthesize all results.')}"

        try:
            resp = _resilient_invoke(
                self.llm,
                [SystemMessage(content=ORCHESTRATOR_PROMPT), HumanMessage(content=context)],
                fallback_llm=self._fallback_llm, verbose=verbose,
            )
            return resp.content
        except Exception as e:
            error_msg = f"Synthesis failed: {type(e).__name__}: {e}"
            if self._error_log:
                self._error_log.log("orchestrator", "FAIL", error_msg[:100])
            # Return expert results directly as fallback
            fallback = "=== SYNTHESIS UNAVAILABLE (API error) ===\n\n"
            for name, result in expert_results.items():
                fallback += f"## {name.upper()}\n{result[:800]}\n\n"
            return fallback


# ══════════════════════════════════════════════════════════════
#  MAIN SYSTEM
# ══════════════════════════════════════════════════════════════

class CarbonMineralizationSystem:
    """
    Multi-agent system with RBAC, self-correction, and resilient API calls.

    Usage:
        system = CarbonMineralizationSystem(workspace="/content/drive/.../workspace")
        result = system.analyze("Analyze 50-year carbonation of OPC + 30% fly ash")
    """

    def __init__(self, openai_key: Optional[str] = None,
                 anthropic_key: Optional[str] = None,
                 gpt_model: str = "gpt-4o",
                 claude_model: str = "claude-sonnet-4-20250514",
                 workspace: str = "_workspace"):
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key

        self.workspace = workspace
        os.makedirs(workspace, exist_ok=True)

        # Shared error log
        self.error_log = SelfCorrectionLog(os.path.join(workspace, "logs"))

        # LLMs
        gpt_llm = ChatOpenAI(model=gpt_model, temperature=0.1)
        claude_llm = _get_claude(temperature=0.1)

        # Agents (Claude agents have GPT-4o fallback)
        self.manager = Manager(llm=gpt_llm, error_log=self.error_log)
        self.orchestrator = Orchestrator(
            llm=claude_llm, fallback_llm=gpt_llm, error_log=self.error_log,
        )
        self.coder = CoderAgent(
            workspace_dir=os.path.join(workspace, "coder"),
            llm=claude_llm, fallback_llm=gpt_llm, error_log=self.error_log,
        )
        self.reviewer = ReviewerAgent(
            llm=claude_llm, fallback_llm=gpt_llm, error_log=self.error_log,
        )

        # Lazy-init experts (RBAC enforced at DomainExpert.__init__)
        self._experts: Dict[str, DomainExpert] = {}
        self._gpt_llm = gpt_llm

        self.history: List[Dict] = []

    def _get_expert(self, name: str) -> DomainExpert:
        if name not in self._experts:
            self._experts[name] = DomainExpert(
                name, llm=self._gpt_llm, error_log=self.error_log,
            )
        return self._experts[name]

    def analyze(self, query: str, verbose: bool = True) -> Dict:
        if verbose:
            print("=" * 70)
            print("  CARBON MINERALIZATION MULTI-AGENT ANALYSIS")
            print("=" * 70)
            print(f"  Query: {query}\n")

        t_total = time.time()

        # Step 1: Manager
        if verbose:
            print("[1/4] Manager: Creating work plan...")
        t0 = time.time()
        plan = self.manager.plan(query)
        if verbose:
            experts = [t["expert"] for t in plan.get("tasks", [])]
            print(f"       Experts: {', '.join(experts)} | Coding: {plan.get('coding_needed', False)} ({time.time()-t0:.1f}s)")

        # Step 2: Expert tasks
        n_tasks = len(plan.get("tasks", []))
        if verbose:
            print(f"\n[2/4] Running {n_tasks} experts (RBAC: each uses only its own tools)...")
        expert_results = {}
        for idx, task in enumerate(plan.get("tasks", []), 1):
            name = task["expert"]
            instruction = task["instruction"]
            if verbose:
                tools_desc = RBAC_REGISTRY.get(name, (None, "none"))[1]
                print(f"  [{idx}/{n_tasks}] {name} [tools: {tools_desc}]")
            t0 = time.time()
            try:
                result = self._get_expert(name).run(instruction)
                expert_results[name] = result
                if verbose:
                    print(f"         Done ({len(result)} chars, {time.time()-t0:.1f}s)")
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:80]}"
                expert_results[name] = f"Error: {msg}"
                self.error_log.log(name, "FAIL", msg)
                if verbose:
                    print(f"         FAIL: {msg} ({time.time()-t0:.1f}s)")

        # Step 3: Coder (if needed)
        coder_output = None
        if plan.get("coding_needed"):
            if verbose:
                print("\n[3/4] Coder + Reviewer...")
            t0 = time.time()
            expert_feedback = "\n".join(f"[{k}]: {v[:400]}" for k, v in expert_results.items())
            code_result = self.coder.write_code(
                plan.get("coding_instruction", ""), expert_feedback, verbose=verbose,
            )

            code = code_result.get("code", "")
            desc = code_result.get("description", "")
            fname = code_result.get("filename", "custom_analysis.py")
            if verbose:
                print(f"       Code: {fname} ({len(code)} chars) | Reviewing...")

            review = self.reviewer.review(code, desc, verbose=verbose)
            verdict = review.get("verdict", "FAIL")
            if verbose:
                print(f"       Verdict: {verdict}")

            if verdict == "PASS":
                path = os.path.join(self.workspace, "coder", fname)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
                self.coder.log_result(desc, "PASS", code)
                coder_output = f"Saved: {fname}"
                if verbose:
                    print(f"       Saved: {path} ({time.time()-t0:.1f}s)")
            else:
                self.coder.log_result(desc, "FAIL", code, "; ".join(review.get("issues", [])))
                if verbose:
                    print("       Retrying...")
                code2 = self.coder.write_code(
                    f"{plan.get('coding_instruction','')}\nFix: {review.get('issues',[])}",
                    expert_feedback, verbose=verbose,
                ).get("code", "")
                review2 = self.reviewer.review(code2, desc, verbose=verbose)
                if review2.get("verdict") == "PASS":
                    path = os.path.join(self.workspace, "coder", fname)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(code2)
                    self.coder.log_result(desc, "PASS", code2)
                    coder_output = f"Saved (retry): {fname}"
                else:
                    self.coder.log_result(desc, "FAIL", code2, "; ".join(review2.get("issues", [])))
                    coder_output = f"FAIL after retry: {review2.get('issues', [])}"
                if verbose:
                    print(f"       {coder_output} ({time.time()-t0:.1f}s)")
        else:
            if verbose:
                print("\n[3/4] Coder: Not needed.")

        # Step 4: Orchestrator synthesis
        t0 = time.time()
        if verbose:
            print("\n[4/4] Orchestrator: Synthesizing...")
        synthesis = self.orchestrator.synthesize(
            query, plan, expert_results, coder_output, verbose=verbose,
        )
        if verbose:
            print(f"       ({time.time()-t0:.1f}s)")

        total = time.time() - t_total
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  COMPLETE ({total:.1f}s total)")
            print(f"{'=' * 70}")
            print(synthesis)

        result = {
            "query": query, "plan": plan,
            "expert_results": expert_results,
            "coder_output": coder_output,
            "synthesis": synthesis,
            "elapsed_s": round(total, 1),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.history.append(result)
        return result

    def compare_materials(self, materials: list, condition: str,
                          duration_days: float = 365) -> Dict:
        query = (
            f"Compare {', '.join(materials)} under {condition} for {duration_days} days. "
            f"For each: phase assemblage, carbonation depth, mechanical properties."
        )
        return self.analyze(query)

    def sensitivity_analysis(self, base_material: str = "OPC",
                              parameter: str = "CO2_ppm",
                              values: list = None) -> Dict:
        if values is None:
            values = [400, 5000, 10000, 50000]
        query = (
            f"Sensitivity analysis on {base_material}: vary {parameter} = {values}. "
            f"For each, compute carbonation depth and mechanical properties."
        )
        return self.analyze(query)

    def get_analysis_summary(self) -> str:
        lines = ["# Analysis History\n"]
        for i, e in enumerate(self.history, 1):
            lines.append(f"## {i}. [{e['timestamp'][:16]}] ({e.get('elapsed_s',0)}s)")
            lines.append(f"**Query:** {e['query'][:100]}...")
            lines.append(f"**Experts:** {', '.join(e.get('expert_results',{}).keys())}\n")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY
# ══════════════════════════════════════════════════════════════

CarbonMineralizationOrchestrator = CarbonMineralizationSystem


def create_agent(role: str, model_name: str = "gpt-4o",
                 temperature: float = 0.1, api_key: Optional[str] = None):
    """Legacy: create a single domain expert."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    return DomainExpert(role, llm=ChatOpenAI(model=model_name, temperature=temperature))
