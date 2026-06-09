from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from app.config import Settings
from app.data_access import ShoppingDataStore, build_data_tools
from app.prompts import (
    SUPERVISOR_PROMPT,
    POLICY_WORKER_PROMPT,
    DATA_WORKER_PROMPT,
    RESPONSE_WORKER_PROMPT,
)
from app.state import ShoppingState
from provider import get_chat_model
from rag.embeddings import build_embedding_model
from rag.vector_store import ChromaPolicyStore


class SupervisorRoute(BaseModel):
    status: str
    needs_policy: bool
    needs_data: bool
    clarification_question: str | None


class ShoppingAssistant:
    """Student scaffold.

    Mục tiêu:
    - Dùng `Settings` để load config.
    - Dùng provider trong `src/provider/`.
    - Dùng embedding loader thật trong `src/rag/embeddings.py`.
    - Tự hoàn thiện phần còn lại: graph, routing, tool calling, RAG search, response synthesis.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()

        # Load chat model
        self.llm = get_chat_model(self.settings)

        # Load dataset order/customer
        self.data_store = ShoppingDataStore(self.settings.orders_path)

        # Load embedding model
        self.embedding_model = build_embedding_model(
            model_name=self.settings.embedding_model_name,
            api_key=self.settings.openai_api_key,
        )

        # Load vector store cho policy
        self.policy_store = ChromaPolicyStore(
            persist_directory=self.settings.chroma_dir,
            embedding_model=self.embedding_model,
        )
        self.policy_store.ensure_index(self.settings.policy_path)

        # Build worker tools
        self.data_tools = build_data_tools(self.data_store)

        @tool
        def search_policy(query: str, top_k: int = 4) -> list[dict[str, Any]]:
            """Tìm kiếm chính sách từ knowledge base."""
            return self.policy_store.search(query, top_k=top_k)

        self.policy_tools = [search_policy]

        # Compile LangGraph
        self.graph = build_graph(
            llm=self.llm,
            policy_tools=self.policy_tools,
            data_tools=self.data_tools,
        )

    def ask(
        self,
        question: str,
        trace_file: Path | None = None,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        # Rebuild index nếu cần
        if rebuild_index:
            self.policy_store.rebuild(self.settings.policy_path)

        # Invoke graph
        initial_state: ShoppingState = {
            "question": question,
            "trace": [],
        }

        result = self.graph.invoke(initial_state)

        # Lưu trace nếu có
        if trace_file:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            with trace_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        return {
            "question": question,
            "route": result.get("route", {}),
            "policy_result": result.get("policy_result", {}),
            "data_result": result.get("data_result", {}),
            "final_answer": result.get("final_answer", ""),
            "trace": result.get("trace", []),
        }

    def run_batch(
        self,
        test_file: Path,
        output_dir: Path,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        # Đọc test file
        with test_file.open("r", encoding="utf-8") as f:
            test_data = json.load(f)

        test_cases = test_data.get("test_cases", [])
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for idx, case in enumerate(test_cases):
            question = case.get("question", "")
            case_id = case.get("id", f"case_{idx}")

            trace_file = output_dir / f"trace_{case_id}.json"
            result = self.ask(question, trace_file=trace_file, rebuild_index=False)

            results.append({
                "case_id": case_id,
                "question": question,
                "expected": case.get("expected", {}),
                "actual": {
                    "route": result.get("route", {}),
                    "final_answer": result.get("final_answer", ""),
                },
            })

        # Sinh summary
        summary = {
            "total": len(test_cases),
            "results": results,
        }

        summary_file = output_dir / "summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        return summary


def build_graph(
    llm: BaseChatModel,
    policy_tools: list,
    data_tools: list,
) -> Any:
    """Compile the LangGraph workflow."""

    # Tạo policy agent và data agent
    policy_agent = create_react_agent(llm, policy_tools)
    data_agent = create_react_agent(llm, data_tools)

    # Định nghĩa state graph
    workflow = StateGraph(ShoppingState)

    # Add nodes
    workflow.add_node("supervisor", lambda state: supervisor_node(state, llm))
    workflow.add_node(
        "worker_1_policy",
        lambda state: worker_1_policy_node(state, policy_agent),
    )
    workflow.add_node("worker_2_data", lambda state: worker_2_data_node(state, data_agent))
    workflow.add_node("worker_3_response", lambda state: worker_3_response_node(state, llm))

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Add conditional edges từ supervisor
    def route_after_supervisor(state: ShoppingState) -> Literal["worker_1_policy", "worker_2_data", "worker_3_response"]:
        route = state.get("route", {})
        status = route.get("status", "ok")

        if status == "clarification_needed":
            return "worker_3_response"

        needs_policy = route.get("needs_policy", False)
        needs_data = route.get("needs_data", False)

        if needs_policy and not needs_data:
            return "worker_1_policy"
        elif needs_data and not needs_policy:
            return "worker_2_data"
        else:
            # needs both, đi policy trước
            return "worker_1_policy"

    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
    )

    # Add conditional edges từ worker_1_policy
    def route_after_policy(state: ShoppingState) -> Literal["worker_2_data", "worker_3_response"]:
        route = state.get("route", {})
        needs_data = route.get("needs_data", False)
        if needs_data:
            return "worker_2_data"
        return "worker_3_response"

    workflow.add_conditional_edges(
        "worker_1_policy",
        route_after_policy,
    )

    # Add edges từ worker_2_data
    workflow.add_edge("worker_2_data", "worker_3_response")

    # Add edges từ worker_3_response
    workflow.add_edge("worker_3_response", END)

    return workflow.compile()


def supervisor_node(state: ShoppingState, llm: BaseChatModel) -> ShoppingState:
    """Gọi LLM để route câu hỏi."""
    question = state.get("question", "")

    # Phát hiện order_id và customer_id từ câu hỏi
    has_order_id = bool(re.search(r'(đơn\s*hàng|order|đơn)\s*(\d+)', question, re.IGNORECASE))
    has_customer_id = bool(re.search(r'(khách\s*hàng|customer)\s*([A-Z]\d+)', question, re.IGNORECASE))

    # Phát hiện các từ khóa về policy
    policy_keywords = ['chính sách', 'policy', 'hoàn trả', 'giao hàng', 'thanh toán', 'voucher', 'điều kiện']
    has_policy_keyword = any(keyword in question.lower() for keyword in policy_keywords)

    # Phát hiện từ khóa cần clarification
    needs_clarification_keywords = ['của tôi', 'của mình', 'my']
    needs_clarification = any(keyword in question.lower() for keyword in needs_clarification_keywords) and not (has_order_id or has_customer_id)

    # Quyết định routing
    if needs_clarification:
        route = {
            "status": "clarification_needed",
            "needs_policy": False,
            "needs_data": False,
            "clarification_question": "Bạn vui lòng cung cấp mã đơn hàng hoặc mã khách hàng để tôi tra cứu thông tin chính xác.",
        }
    elif has_order_id or has_customer_id:
        if has_policy_keyword or any(word in question.lower() for word in ['có được', 'được không', 'được hoàn', 'có thể']):
            route = {"status": "ok", "needs_policy": True, "needs_data": True, "clarification_question": None}
        else:
            route = {"status": "ok", "needs_policy": False, "needs_data": True, "clarification_question": None}
    elif has_policy_keyword or not (has_order_id or has_customer_id):
        route = {"status": "ok", "needs_policy": True, "needs_data": False, "clarification_question": None}
    else:
        route = {"status": "ok", "needs_policy": True, "needs_data": False, "clarification_question": None}

    state["route"] = route
    state["trace"] = state.get("trace", []) + [{"node": "supervisor", "output": route}]

    return state


def worker_1_policy_node(state: ShoppingState, policy_agent: Any) -> ShoppingState:
    """Build subgraph hoặc agent cho policy worker."""
    question = state.get("question", "")

    prompt = POLICY_WORKER_PROMPT.format(question=question)

    # Invoke agent
    result = policy_agent.invoke({"messages": [HumanMessage(content=prompt)]})

    # Extract kết quả
    messages = result.get("messages", [])
    last_message = messages[-1] if messages else None

    policy_result = {
        "status": "ok",
        "summary": last_message.content if last_message else "",
        "messages": [{"type": m.type, "content": m.content} for m in messages],
    }

    state["policy_result"] = policy_result
    state["trace"] = state.get("trace", []) + [{"node": "worker_1_policy", "output": policy_result}]

    return state


def worker_2_data_node(state: ShoppingState, data_agent: Any) -> ShoppingState:
    """Build subgraph hoặc agent cho data worker."""
    question = state.get("question", "")

    prompt = DATA_WORKER_PROMPT.format(question=question)

    # Invoke agent
    result = data_agent.invoke({"messages": [HumanMessage(content=prompt)]})

    # Extract kết quả
    messages = result.get("messages", [])
    last_message = messages[-1] if messages else None

    data_result = {
        "status": "ok",
        "summary": last_message.content if last_message else "",
        "messages": [{"type": m.type, "content": m.content} for m in messages],
    }

    state["data_result"] = data_result
    state["trace"] = state.get("trace", []) + [{"node": "worker_2_data", "output": data_result}]

    return state


def worker_3_response_node(state: ShoppingState, llm: BaseChatModel) -> ShoppingState:
    """Tổng hợp output từ supervisor + workers."""
    question = state.get("question", "")
    route = state.get("route", {})
    policy_result = state.get("policy_result", {})
    data_result = state.get("data_result", {})

    prompt = RESPONSE_WORKER_PROMPT.format(
        question=question,
        route=json.dumps(route, ensure_ascii=False, indent=2),
        policy_result=json.dumps(policy_result, ensure_ascii=False, indent=2),
        data_result=json.dumps(data_result, ensure_ascii=False, indent=2),
    )

    messages = [
        SystemMessage(content="Bạn là response worker. Tạo câu trả lời cuối cùng theo đúng format."),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    final_answer = response.content

    state["final_answer"] = final_answer
    state["trace"] = state.get("trace", []) + [{"node": "worker_3_response", "output": final_answer}]

    return state
