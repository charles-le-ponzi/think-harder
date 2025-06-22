"""
title: Think Harder (Agent Swarm)
author: Ponzi (Refactored by Assistant)
author_url: https://github.com/dory11111
funding_url: https://github.com/open-webui
version: 8.9.1
description: >-
  A dynamic, multi-stage agent execution framework that transforms a query into a detailed execution plan. All agent reports are streamed as collapsible citations. All streaming operations use "end-first" preview to show you the conclusion immediately.
**Requirements & Settings**:
1.  **Activate Filter**: Enable the "Think Harder" toggle switch in `Workspace -> Models -> (Edit Your Model) -> Functions`.
2.  **Configure System**: Customize prompts, history, and retry behavior in the model's valve settings.
3.  **⚠️ CRITICAL WARNING ⚠️**: To prevent infinite loops and unpredictable behavior, apply this filter to a **single, dedicated model** not used for general chat.
**Updates**:
V8.9.1: Fixed a bug where agent stream previews did not show the end of the content. All streaming statuses now correctly display the last N characters of the response.
V8.9.0: Implemented end-first streaming for the synthesizer and ensured all agent reports are visible as citations.
V8.8.1: Corrected critical `AttributeError` and invalid syntax issues. Refactored into modular components.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp
from fastapi import Request
from pydantic import BaseModel, Field, model_validator

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Custom Exception ---
class LLMAPIError(Exception):
    """Custom exception for failures in LLM API calls."""


# --- Standalone Utility Components ---


class FilterUtils:
    """Provides static utility methods for sending data back to the UI."""

    @staticmethod
    async def send_citation(
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        url: str,
        title: str,
        content: str,
    ) -> None:
        if emitter:
            await emitter(
                {
                    "type": "citation",
                    "data": {
                        "document": [content],
                        "metadata": [{"source": url, "html": False}],
                        "source": {"name": title},
                    },
                }
            )

    @staticmethod
    async def send_status(
        emitter: Optional[Callable[[dict], Awaitable[None]]],
        status_message: str,
        done: bool,
    ) -> None:
        if emitter:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": status_message, "done": done},
                }
            )


class ThoughtParser:
    """Parses and strips thought/instruction tags from LLM outputs."""

    def __init__(self, tags_json: str):
        self.patterns = self._compile_patterns(tags_json)

    def _compile_patterns(self, tags_json: str) -> List[re.Pattern]:
        try:
            tags = json.loads(tags_json)
            if not isinstance(tags, list):
                return []
            return [
                re.compile(
                    f"{re.escape(tp['start'])}.*?{re.escape(tp['end'])}",
                    re.DOTALL | re.IGNORECASE,
                )
                for tp in tags
                if isinstance(tp, dict) and "start" in tp and "end" in tp
            ]
        except json.JSONDecodeError:
            return []

    def strip(self, text: str) -> str:
        for pattern in self.patterns:
            text = pattern.sub("", text)
        return text.strip()


class LLMClient:
    """An asynchronous client for making cancellable, retrying LLM API calls."""

    def __init__(
        self,
        request: Request,
        thought_parser: ThoughtParser,
        stream_throttle_seconds: float,
        timeout_seconds: int = 180,
    ):
        self.base_url = str(request.base_url)
        self.headers = {
            "Authorization": request.headers.get("Authorization", ""),
            "Content-Type": "application/json",
        }
        self.thought_parser = thought_parser
        self.stream_throttle_seconds = stream_throttle_seconds
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def call(
        self,
        model: str,
        messages: List[Dict],
        stream: bool,
        update_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        url, payload, full_response = (
            f"{self.base_url}api/chat/completions",
            {"model": model, "messages": messages, "stream": stream},
            "",
        )
        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout
            ) as session, session.post(
                url, json=payload, headers=self.headers
            ) as response:
                response.raise_for_status()
                if stream:
                    last_update = time.time()
                    async for chunk_bytes in response.content:
                        if not chunk_bytes:
                            continue
                        for line in chunk_bytes.decode("utf-8").strip().split("\n"):
                            if not line.startswith("data:"):
                                continue
                            data_part = line[len("data: ") :]
                            if data_part.strip() == "[DONE]":
                                break
                            try:
                                content = (
                                    json.loads(data_part)
                                    .get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content")
                                )
                                if content:
                                    full_response += content
                                    now = time.time()
                                    if update_callback and (
                                        now - last_update > self.stream_throttle_seconds
                                    ):
                                        await update_callback(full_response)
                                        last_update = now
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
                    if update_callback:
                        await update_callback(full_response)
                else:
                    full_response = (await response.json())["choices"][0]["message"][
                        "content"
                    ]
        except aiohttp.ClientResponseError as e:
            raise LLMAPIError(
                f"API call to {model} failed with status {e.status}: {e.message}"
            ) from e
        except asyncio.TimeoutError as e:
            raise LLMAPIError(
                f"API call to {model} timed out after {self.timeout.total}s."
            ) from e
        except Exception as e:
            logger.exception(f"Unexpected error in LLM call to model {model}")
            raise LLMAPIError(f"An unexpected error occurred: {e}") from e
        return self.thought_parser.strip(full_response)


# --- Main Filter Class ---


class Filter:
    version = "8.9.1"

    class Valves(BaseModel):
        orchestrator_model_id: str = Field(
            default="qwen2:7b", title="Orchestrator Model ID"
        )
        agent_model_id: str = Field(
            default="qwen2:7b", title="Specialist Agent Model ID"
        )
        synthesizer_model_id: str = Field(
            default="qwen2:7b", title="Synthesizer Model ID"
        )
        max_concurrent_agents: int = Field(
            default=5, ge=1, le=10, title="Max Concurrent Agents"
        )
        agent_timeout_seconds: int = Field(
            default=180, ge=30, le=600, title="Agent Task Timeout (s)"
        )
        num_history_messages: int = Field(
            default=3, ge=0, le=10, title="Number of History Messages"
        )
        agent_retries: int = Field(default=1, ge=0, le=5, title="Agent Retries")
        retry_delay_seconds: float = Field(
            default=2.0, ge=0.5, le=10.0, title="Retry Delay (s)"
        )
        end_first_window_chars: int = Field(
            default=200,
            ge=50,
            le=1000,
            title="End-First Streaming Preview (Chars)",
            description="The number of characters from the end of a streaming response to show in the status preview.",
        )

        orchestrator_prompt_template: str = Field(
            default="""You are a "Chief of Staff" AI Orchestrator. Your mission is to transform a user's high-level goal into a detailed, multi-stage, multi-agent execution plan.
USER QUERY: "{user_query}"
**PLANNING DIRECTIVES:**
1.  **Deconstruct:** Analyze the query to identify distinct phases (e.g., Research, Analysis, Creation). Define these as `[STAGE]`.
2.  **Delegate:** For each stage, define parallel specialist agents with detailed, single-line prompts.
3.  **Define Personas:** Each prompt must contain: **Role:** (Expert Title), **Context:** (Inputs), **Core Task:** (Instructions), **Output Format:** (Structure).
**OUTPUT FORMAT RULES:**
*   Respond with ONLY the text-based plan.
*   Each stage MUST start with `[STAGE X (Description)]`.
*   Each agent MUST start with `- Agent Name: [Single-line prompt]`.
**EXAMPLE:**
[STAGE 1 (Market Intelligence)]
- Chief Market Analyst: **Role:** Seasoned analyst. **Context:** User query on a new VR headset. **Core Task:** Analyze market size, competitors, and gaps. **Output Format:** Markdown report with Market Size, Competitive Landscape, and Top 3 Opportunities.""",
            title="Orchestrator Prompt",
            description="Instructs the orchestrator agent on how to create a plan.",
            json_schema_extra={"widget": "textarea"},
        )

        synthesizer_prompt_template: str = Field(
            default="""You are a final synthesizer AI. Your role is to take the collection of reports and analyses from a team of specialist agents and consolidate them into a single, cohesive, and comprehensive final answer that directly addresses the user's original query.
USER QUERY: "{user_query}"
---
AGENT ELABORATIONS:
{agent_elaborations}
---
Based on the provided context, synthesize the final, complete response for the user.""",
            title="Synthesizer Prompt",
            description="Instructs the final agent to interpret agent reports into a final plan.",
            json_schema_extra={"widget": "textarea"},
        )

        stream_throttle_seconds: float = Field(
            default=0.5, ge=0.1, le=2.0, title="Stream Update Throttle (s)"
        )
        custom_thought_tags_json: str = Field(
            default='[{"start": "<|im_start|>thought", "end": "<|im_end|>"}]',
            title="Custom Thought Tags (JSON)",
        )
        orchestrator_use_no_think: bool = Field(
            default=False, title="Orchestrator: Use /no_think (Qwen3)"
        )
        agent_use_no_think: bool = Field(
            default=False, title="Agents: Use /no_think (Qwen3)"
        )
        synthesizer_use_no_think: bool = Field(
            default=False, title="Synthesizer: Use /no_think (Qwen3)"
        )

        @model_validator(mode="after")
        def validate_json(self) -> "Filter.Valves":
            try:
                json.loads(self.custom_thought_tags_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in 'Custom Thought Tags': {e}") from e
            return self

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBjbGFzcz0ic2l6ZS02Ij4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAxOHYtNS4yNW0wIDBhNi4wMSA2LjAxIDAgMCAwIDEuNS0uMTg5bS0xLjUuMTg5YTYuMDEgNi4wMSAwIDAgMS0xLjUtLjE4OW0zLjc1IDcuNDc4YTEyLjA2IDEyLjA2IDAgMCAxLTQuNSAwbTMuNzUgMi4zODNhMTQuNDA2IDE0LjQwNiAwIDAgMS0zIDBNMTQuMjUgMTh2LS4xOTJjMC0uOTgzLjY1ganoLjgyMyAxLjUwOC0yLjMxNmE3LjUgNy41IDAgMSAwLTcuNTE3IDBjLjg1LjQ5MyAxLjUwOSAxLjMzMyAxLjUwOSAyLjMxNlYxOCIgLz4KPC9zdmc+Cg=="

    def _prepare_prompt(self, base_prompt: str, use_no_think: bool) -> str:
        return f"{base_prompt.strip()}\n\n/no_think" if use_no_think else base_prompt

    async def _create_execution_plan(
        self, llm_client: LLMClient, user_query: str
    ) -> List[List[Dict[str, str]]]:
        base_prompt = self.valves.orchestrator_prompt_template.format(
            user_query=user_query
        )
        final_prompt = self._prepare_prompt(
            base_prompt, self.valves.orchestrator_use_no_think
        )
        raw_plan_text = await llm_client.call(
            self.valves.orchestrator_model_id,
            [{"role": "user", "content": final_prompt}],
            stream=False,
        )
        plan, agent_pattern = [], re.compile(r"-\s*(.*?):\s*(.*)")
        stage_chunks = re.split(r"(\[STAGE .*?\])", raw_plan_text)
        for chunk in stage_chunks:
            if not chunk.strip() or chunk.startswith("[STAGE"):
                continue
            stage_agents = [
                {"name": match.group(1).strip(), "prompt": match.group(2).strip()}
                for line in chunk.strip().split("\n")
                if (match := agent_pattern.match(line.strip()))
            ]
            if stage_agents:
                plan.append(stage_agents)
        if not plan:
            logger.error(
                f"Orchestrator failed to generate a valid plan. Raw Response: {raw_plan_text}"
            )
            return [
                [
                    {
                        "name": "Fallback Analyst",
                        "prompt": "Analyze the original user query and explain why a plan could not be generated.",
                    }
                ]
            ]
        return plan

    async def _get_elaborations(
        self,
        llm_client: LLMClient,
        agents: List[Dict],
        messages_history: List[Dict],
        stage_context: str,
        emitter: Optional[Callable[[dict], Awaitable[None]]],
    ) -> Dict[str, str]:
        semaphore = asyncio.Semaphore(self.valves.max_concurrent_agents)

        async def run_agent_task(agent: Dict[str, Any], index: int) -> tuple[str, str]:
            name, base_prompt = agent.get("name", f"Agent_{index+1}"), agent.get(
                "prompt", "Error: No prompt provided."
            )
            for attempt in range(self.valves.agent_retries + 1):
                try:
                    async with semaphore:
                        await FilterUtils.send_status(
                            emitter, f"Agent '{name}' is analyzing...", False
                        )
                        full_prompt = (
                            f"{stage_context}\n\nYour Task: {base_prompt}"
                            if stage_context
                            else f"Your Task: {base_prompt}"
                        )
                        final_prompt = self._prepare_prompt(
                            full_prompt, self.valves.agent_use_no_think
                        )
                        history_count = self.valves.num_history_messages
                        limited_history = (
                            messages_history[-history_count - 1 : -1]
                            if history_count > 0
                            else []
                        )
                        agent_messages = copy.deepcopy(
                            limited_history + [messages_history[-1]]
                        )
                        agent_messages.insert(
                            -1, {"role": "system", "content": final_prompt}
                        )

                        async def stream_callback(text: str):
                            # CORRECTED: Apply end-first streaming to agent tasks
                            preview_chars = min(
                                self.valves.end_first_window_chars, 150
                            )  # Cap agent previews
                            preview = text[-preview_chars:]
                            sanitized_preview = re.sub(
                                r"\s+", " ", preview
                            ).strip()  # Sanitize for single-line status
                            await FilterUtils.send_status(
                                emitter,
                                f"Agent '{name}': ...{sanitized_preview} ({len(text)} chars)",
                                False,
                            )

                        report = await llm_client.call(
                            self.valves.agent_model_id,
                            agent_messages,
                            stream=True,
                            update_callback=stream_callback,
                        )

                        await FilterUtils.send_citation(
                            emitter,
                            url=f"Agent Report: {name}",
                            title=f"Agent '{name}' Full Output",
                            content=report,
                        )
                        await FilterUtils.send_status(
                            emitter, f"✅ Agent '{name}' finished.", False
                        )
                        return name, report
                except Exception as e:
                    logger.warning(
                        f"Agent '{name}' failed on attempt {attempt + 1}: {e}"
                    )
                    if attempt >= self.valves.agent_retries:
                        raise
                    await asyncio.sleep(self.valves.retry_delay_seconds)
            raise RuntimeError("Agent task failed unexpectedly after all retries.")

        tasks = [run_agent_task(agent, i) for i, agent in enumerate(agents)]
        results = await asyncio.gather(
            *(
                asyncio.wait_for(task, timeout=self.valves.agent_timeout_seconds)
                for task in tasks
            ),
            return_exceptions=True,
        )
        elaborations = {
            agents[i].get("name", f"Agent {i+1}"): (
                f"Error: {'Agent task timed out.' if isinstance(res, asyncio.TimeoutError) else f'Agent task failed: {res}'}"
                if isinstance(res, Exception)
                else res[1]
            )
            for i, res in enumerate(results)
        }
        return elaborations

    def _format_context_for_next_stage(self, elaborations: Dict[str, str]) -> str:
        context = "CONTEXT FROM PREVIOUS STAGE:\n---\n"
        for name, report in elaborations.items():
            context += f"Report from '{name}':\n{report}\n\n"
        return context.strip()

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> dict:
        if not self.toggle or not all((__user__, __request__, __event_emitter__)):
            return body
        messages = body.get("messages", [])
        if not messages or messages[-1].get("role") != "user":
            return body

        try:
            thought_parser = ThoughtParser(self.valves.custom_thought_tags_json)
            llm_client = LLMClient(
                request=__request__,
                thought_parser=thought_parser,
                stream_throttle_seconds=self.valves.stream_throttle_seconds,
                timeout_seconds=self.valves.agent_timeout_seconds,
            )
            user_query = messages[-1]["content"]
            await FilterUtils.send_status(
                __event_emitter__, "Think Harder: Designing execution plan...", False
            )
            execution_plan = await self._create_execution_plan(llm_client, user_query)

            cumulative_context = ""
            for i, stage_agents in enumerate(execution_plan):
                await FilterUtils.send_status(
                    __event_emitter__,
                    f"Stage {i+1}/{len(execution_plan)}: Running agents...",
                    False,
                )
                stage_elaborations = await self._get_elaborations(
                    llm_client,
                    stage_agents,
                    messages,
                    cumulative_context,
                    __event_emitter__,
                )
                cumulative_context = self._format_context_for_next_stage(
                    stage_elaborations
                )

            await FilterUtils.send_status(
                __event_emitter__, "Synthesizing final response...", False
            )
            base_prompt = self.valves.synthesizer_prompt_template.format(
                user_query=user_query, agent_elaborations=cumulative_context
            )
            final_synth_prompt = self._prepare_prompt(
                base_prompt, self.valves.synthesizer_use_no_think
            )

            async def synth_callback(cumulative_text: str):
                preview_chars = self.valves.end_first_window_chars
                preview = cumulative_text[-preview_chars:]
                sanitized_preview = re.sub(
                    r"\s+", " ", preview
                ).strip()  # Sanitize for single-line status
                await FilterUtils.send_status(
                    __event_emitter__,
                    f"🧠 Synthesis: ...{sanitized_preview} ({len(cumulative_text)} chars)",
                    False,
                )

            final_plan = await llm_client.call(
                self.valves.synthesizer_model_id,
                [{"role": "user", "content": final_synth_prompt}],
                stream=True,
                update_callback=synth_callback,
            )

            await FilterUtils.send_citation(
                __event_emitter__,
                url="Synthesizer Final Report",
                title="Synthesizer Final Report",
                content=final_plan,
            )
            body["messages"].insert(
                -1,
                {
                    "role": "system",
                    "content": f"A multi-stage agent team analyzed this query. Their final plan is below. Use it to formulate your response.\n\n{final_plan}",
                },
            )
            await FilterUtils.send_status(
                __event_emitter__, "Think Harder: Analysis Complete!", True
            )

        except LLMAPIError as e:
            logger.error(f"Think Harder failed due to an API error: {e}")
            await FilterUtils.send_status(
                __event_emitter__, f"Think Harder Error: {e}", True
            )
        except Exception:
            logger.exception("An unexpected error occurred in the Think Harder filter.")
            await FilterUtils.send_status(
                __event_emitter__,
                "An unexpected internal error occurred. Please check the logs.",
                True,
            )
        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        thought_parser = ThoughtParser(self.valves.custom_thought_tags_json)
        for message in body.get("messages", []):
            if message.get("role") == "assistant" and isinstance(
                message.get("content"), str
            ):
                message["content"] = thought_parser.strip(message["content"])
        return body