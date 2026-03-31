from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool

from .llm_client import LLMClient
from .models import ActionItem, AgentPlan, CommandPlan, EmailInput, IntentItem, ProcessEmailResponse, ToolResult
from .tools import create_ticket, send_auto_reply, start_approval_flow


class MailAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def execute(self, email: EmailInput) -> ProcessEmailResponse:
        if self.llm.enabled:
            try:
                return self._execute_by_react(email)
            except Exception:
                # Keep service available even if LLM agent parsing/tool loop fails.
                return self._execute_by_rules(email)
        return self._execute_by_rules(email)

    def plan(self, email: EmailInput) -> AgentPlan:
        if self.llm.enabled:
            try:
                return self._plan_by_llm(email)
            except Exception:
                # Fail-open to deterministic rules for availability.
                return self._plan_by_rules(email)
        return self._plan_by_rules(email)

    def plan_command(self, command: str) -> CommandPlan:
        text = command.strip()
        if not text:
            return CommandPlan(
                intent="unknown",
                confidence=1.0,
                params={},
                recommended_action="请输入自然语言命令，例如：帮我分类最近10封未读邮件。",
            )
        if self.llm.enabled:
            try:
                return self._plan_command_by_llm(text)
            except Exception:
                return self._plan_command_by_rules(text)
        return self._plan_command_by_rules(text)

    def _plan_by_llm(self, email: EmailInput) -> AgentPlan:
        schema_hint = {
            "classification": "verification_code|advertisement|notification|inquiry|incident|approval|complaint|other",
            "intents": [{"intent": "create_ticket|start_approval|send_reply|request_more_info|none", "confidence": 0.0, "reason": ""}],
            "actions": [{"tool": "create_ticket|start_approval_flow|send_auto_reply", "params": {}}],
            "reply_draft": "string",
            "recommended_action": "string",
        }
        prompt = f"""
You are an email operations agent.
Analyze the email and output STRICT JSON only.

JSON schema example:
{json.dumps(schema_hint, ensure_ascii=False)}

Rules:
1) If it is fault/exception/outage, include create_ticket and send_auto_reply.
2) If it is approval/budget/reimbursement/purchase request, include start_approval_flow and send_auto_reply.
3) If it is verification code email, classify as verification_code and usually no workflow action.
4) If it is advertisement/promotional content, classify as advertisement and recommend archive/unsubscribe.
5) If it is system/bank/order notification, classify as notification.
3) Always keep confidence in [0,1].
4) reply_draft must be polite and concise.

Email:
subject: {email.subject}
sender: {email.sender}
body: {email.body}
"""
        content = self.llm.chat(
            messages=[
                {"role": "system", "content": "You are a strict JSON planner."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        data = self._load_json_from_text(content)
        return AgentPlan(
            classification=data["classification"],
            intents=[IntentItem(**it) for it in data.get("intents", [])],
            actions=[ActionItem(**a) for a in data.get("actions", [])],
            reply_draft=data.get("reply_draft", ""),
            recommended_action=data.get("recommended_action", "人工确认后处理"),
        )

    def _execute_by_react(self, email: EmailInput) -> ProcessEmailResponse:
        executor = self._build_react_executor()
        result = executor.invoke(
            {
                "input": (
                    "Please process this email.\n"
                    f"subject: {email.subject}\n"
                    f"sender: {email.sender}\n"
                    f"body: {email.body}"
                )
            }
        )

        output_text = str(result.get("output", "")).strip()
        data = self._load_json_from_text(output_text)
        plan = AgentPlan(
            classification=data["classification"],
            intents=[IntentItem(**it) for it in data.get("intents", [])],
            actions=[ActionItem(**a) for a in data.get("actions", [])],
            reply_draft=data.get("reply_draft", ""),
            recommended_action=data.get("recommended_action", "人工确认后处理"),
        )
        tool_results = self._tool_results_from_steps(result.get("intermediate_steps", []))
        return ProcessEmailResponse(
            classification=plan.classification,
            intents=plan.intents,
            actions=plan.actions,
            reply_draft=plan.reply_draft,
            recommended_action=plan.recommended_action,
            tool_results=tool_results,
        )

    def _plan_command_by_llm(self, command: str) -> CommandPlan:
        schema_hint = {
            "intent": "classify_inbox|process_inbox|read_unread|help|unknown",
            "confidence": 0.0,
            "params": {"limit": 10, "need_summary": False, "strict_mode": False, "unread_only": True},
            "recommended_action": "string",
        }
        prompt = f"""
You are a mailbox command planner.
Given a user command, output STRICT JSON only.

JSON schema example:
{json.dumps(schema_hint, ensure_ascii=False)}

Rules:
1) classify_inbox: classify unread emails only.
2) process_inbox: classify and execute actions on unread emails.
3) read_unread: fetch recent unread emails for quick preview.
4) help: when user asks what commands are supported.
5) unknown: when command is ambiguous.
6) confidence must be in [0,1].
7) params.limit should be an integer in [1, 20], default 10.
8) If user asks for summarize/summary/总结/概括, set params.need_summary=true.
9) If user asks strict unread mode (严格/strict/不要回退/直接报错), set params.strict_mode=true.
10) If user does not explicitly mention unread email, set params.unread_only=false.

User command:
{command}
""".strip()
        content = self.llm.chat(
            messages=[
                {"role": "system", "content": "You are a strict JSON command planner."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        data = self._load_json_from_text(content)
        params = data.get("params", {})
        limit = self._safe_limit(params.get("limit"), default=self._extract_limit(command, default=10))
        need_summary = self._extract_need_summary(command, default=bool(params.get("need_summary", False)))
        strict_mode = self._extract_strict_mode(command, default=bool(params.get("strict_mode", False)))
        unread_only = self._extract_unread_only(command, default=bool(params.get("unread_only", True)))
        intent = data.get("intent", "unknown")
        recommended = data.get("recommended_action", "按识别到的意图自动执行。")
        if intent in {"classify_inbox", "process_inbox", "read_unread"}:
            action_text = {
                "classify_inbox": "分类未读邮件",
                "process_inbox": "自动处理未读邮件",
                "read_unread": "读取未读邮件摘要" if unread_only else "读取最新邮件摘要",
            }[intent]
            extra = "并生成总结。" if need_summary else "。"
            strict_text = "（严格模式，IMAP失败即报错）" if strict_mode else ""
            recommended = f"建议执行：{action_text}（最近 {limit} 封）{strict_text}{extra}"
        return CommandPlan(
            intent=intent,
            confidence=float(data.get("confidence", 0.6)),
            params={
                "limit": limit,
                "need_summary": need_summary,
                "strict_mode": strict_mode,
                "unread_only": unread_only,
            },
            recommended_action=recommended,
        )

    def _plan_command_by_rules(self, command: str) -> CommandPlan:
        text = command.lower()
        limit = self._extract_limit(command, default=10)
        need_summary = self._extract_need_summary(command, default=False)
        strict_mode = self._extract_strict_mode(command, default=False)
        unread_only = self._extract_unread_only(command, default=True)
        if any(k in text for k in ["帮助", "help", "怎么用", "支持什么"]):
            return CommandPlan(
                intent="help",
                confidence=0.98,
                params={
                    "limit": limit,
                    "need_summary": need_summary,
                    "strict_mode": strict_mode,
                    "unread_only": unread_only,
                },
                recommended_action="展示可用命令示例，用户可直接发送自然语言进行执行。",
            )
        if any(k in text for k in ["分类", "归类", "classify"]):
            extra = "并附上总结。" if need_summary else ""
            strict_text = "（严格模式）" if strict_mode else ""
            return CommandPlan(
                intent="classify_inbox",
                confidence=0.92,
                params={
                    "limit": limit,
                    "need_summary": need_summary,
                    "strict_mode": strict_mode,
                    "unread_only": True,
                },
                recommended_action=f"对最近 {limit} 封未读邮件进行分类，不执行动作{strict_text}{extra}",
            )
        if any(k in text for k in ["处理", "执行", "自动处理", "process"]):
            extra = "并附上执行总结。" if need_summary else ""
            strict_text = "（严格模式）" if strict_mode else ""
            return CommandPlan(
                intent="process_inbox",
                confidence=0.9,
                params={
                    "limit": limit,
                    "need_summary": need_summary,
                    "strict_mode": strict_mode,
                    "unread_only": True,
                },
                recommended_action=f"对最近 {limit} 封未读邮件进行分类并自动执行动作{strict_text}{extra}",
            )
        if any(k in text for k in ["读取", "查看", "列出", "未读", "inbox", "收件箱"]):
            extra = "并生成总结。" if need_summary else ""
            strict_text = "（严格模式，IMAP失败即报错）" if strict_mode else ""
            target = "未读邮件摘要" if unread_only else "最新邮件摘要"
            return CommandPlan(
                intent="read_unread",
                confidence=0.82,
                params={
                    "limit": limit,
                    "need_summary": need_summary,
                    "strict_mode": strict_mode,
                    "unread_only": unread_only,
                },
                recommended_action=f"读取最近 {limit} 封{target}{strict_text}{extra}",
            )
        return CommandPlan(
            intent="unknown",
            confidence=0.45,
            params={
                "limit": limit,
                "need_summary": need_summary,
                "strict_mode": strict_mode,
                "unread_only": unread_only,
            },
            recommended_action="命令不明确，建议改成：分类最近10封未读邮件 / 处理最近5封未读邮件。",
        )

    def _execute_by_rules(self, email: EmailInput) -> ProcessEmailResponse:
        plan = self._plan_by_rules(email)
        tool_map = {
            "create_ticket": create_ticket,
            "start_approval_flow": start_approval_flow,
            "send_auto_reply": send_auto_reply,
        }
        tool_results: List[ToolResult] = []
        for action in plan.actions:
            fn = tool_map.get(action.tool)
            if fn is None:
                tool_results.append(
                    ToolResult(
                        tool=action.tool,
                        success=False,
                        result={"error": f"tool not found: {action.tool}"},
                    )
                )
                continue
            try:
                result = fn(**action.params)
                tool_results.append(ToolResult(tool=action.tool, success=True, result=result))
            except Exception as e:
                tool_results.append(
                    ToolResult(
                        tool=action.tool,
                        success=False,
                        result={"error": str(e)},
                    )
                )

        return ProcessEmailResponse(
            classification=plan.classification,
            intents=plan.intents,
            actions=plan.actions,
            reply_draft=plan.reply_draft,
            recommended_action=plan.recommended_action,
            tool_results=tool_results,
        )

    def _build_react_executor(self) -> AgentExecutor:
        llm = self.llm.build_chat_model(temperature=0.1)
        tools = self._build_langchain_tools()
        prompt = PromptTemplate.from_template(
            """
You are an email operations ReAct agent.
Decide classification, intent and actions, then call tools when needed.

Available tools:
{tools}

Tool names:
{tool_names}

Rules:
1) If incident/fault/outage, call create_ticket and usually send_auto_reply.
2) If approval/budget/reimbursement/purchase, call start_approval_flow and usually send_auto_reply.
3) For verification_code/advertisement/notification, usually no tool call is required.
4) confidence must be in [0,1].
5) Final answer MUST be strict JSON only (no markdown/code fences), schema:
{{
  "classification": "verification_code|advertisement|notification|inquiry|incident|approval|complaint|other",
  "intents": [{{"intent":"create_ticket|start_approval|send_reply|request_more_info|none","confidence":0.0,"reason":"..."}}],
  "actions": [{{"tool":"create_ticket|start_approval_flow|send_auto_reply","params":{{}}}}],
  "reply_draft": "string",
  "recommended_action": "string"
}}
6) actions should match the tools you actually called.

Use this format:
Question: the input question you must answer
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: JSON object
Observation: tool output
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: <STRICT JSON>

Question: {input}
Thought:{agent_scratchpad}
""".strip()
        )
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=6,
        )

    def _build_langchain_tools(self) -> List[StructuredTool]:
        def _create_ticket(subject: str, sender: str, description: str, priority: str = "medium") -> Dict[str, Any]:
            return create_ticket(subject=subject, sender=sender, description=description, priority=priority)

        def _start_approval_flow(
            sender: str, title: str, details: str, amount: float | None = None
        ) -> Dict[str, Any]:
            return start_approval_flow(sender=sender, title=title, details=details, amount=amount)

        def _send_auto_reply(recipient: str, subject: str, body: str) -> Dict[str, Any]:
            return send_auto_reply(recipient=recipient, subject=subject, body=body)

        return [
            StructuredTool.from_function(
                func=_create_ticket,
                name="create_ticket",
                description="Create a ticket for incidents or technical failures.",
            ),
            StructuredTool.from_function(
                func=_start_approval_flow,
                name="start_approval_flow",
                description="Start approval workflow for reimbursement/purchase/budget requests.",
            ),
            StructuredTool.from_function(
                func=_send_auto_reply,
                name="send_auto_reply",
                description="Queue an auto reply email to sender/recipient.",
            ),
        ]

    def _plan_by_rules(self, email: EmailInput) -> AgentPlan:
        text = f"{email.subject}\n{email.body}".lower()

        intents: List[IntentItem] = []
        actions: List[ActionItem] = []

        if any(k in text for k in ["验证码", "verification code", "otp", "动态码", "校验码"]):
            classification = "verification_code"
            intents.append(IntentItem(intent="none", confidence=0.95, reason="Detected verification-code keywords"))
            reply = ""
            return AgentPlan(
                classification=classification,
                intents=intents,
                actions=actions,
                reply_draft=reply,
                recommended_action="提取验证码并标记为工具邮件；通常不需要自动回复，建议归档到“验证码”文件夹。",
            )

        if any(k in text for k in ["优惠", "促销", "折扣", "退订", "marketing", "sale", "广告"]):
            classification = "advertisement"
            intents.append(IntentItem(intent="none", confidence=0.9, reason="Detected marketing keywords"))
            reply = ""
            return AgentPlan(
                classification=classification,
                intents=intents,
                actions=actions,
                reply_draft=reply,
                recommended_action="建议标记为广告邮件并归档；可触发退订流程（人工确认后）。",
            )

        if any(k in text for k in ["通知", "提醒", "账单", "发货", "到账", "通知您", "notification"]):
            classification = "notification"
            intents.append(IntentItem(intent="none", confidence=0.82, reason="Detected notification keywords"))
            reply = ""
            return AgentPlan(
                classification=classification,
                intents=intents,
                actions=actions,
                reply_draft=reply,
                recommended_action="建议归档为通知类邮件；若涉及异常账单/交易再升级人工处理。",
            )

        if any(k in text for k in ["故障", "报错", "异常", "无法", "error", "500", "login failed"]):
            classification = "incident"
            intents.append(IntentItem(intent="create_ticket", confidence=0.92, reason="Detected fault keywords"))
            intents.append(IntentItem(intent="send_reply", confidence=0.88, reason="Need immediate response"))
            actions.append(
                ActionItem(
                    tool="create_ticket",
                    params={
                        "subject": email.subject,
                        "sender": email.sender,
                        "description": email.body,
                        "priority": "high",
                    },
                )
            )
            reply = (
                "您好，问题已收到并已创建工单，技术团队会尽快排查并同步处理进展。"
                "如有截图或报错时间点，请补充以加快定位。"
            )
            actions.append(
                ActionItem(
                    tool="send_auto_reply",
                    params={"recipient": email.sender, "subject": f"Re: {email.subject}", "body": reply},
                )
            )
            return AgentPlan(
                classification=classification,
                intents=intents,
                actions=actions,
                reply_draft=reply,
                recommended_action="创建高优先级工单并自动回复告知受理，随后进入故障跟踪。",
            )

        if any(k in text for k in ["审批", "报销", "采购", "预算", "approval", "reimbursement"]):
            classification = "approval"
            intents.append(IntentItem(intent="start_approval", confidence=0.9, reason="Detected approval keywords"))
            intents.append(IntentItem(intent="send_reply", confidence=0.8, reason="Need acknowledgment"))
            amount = self._extract_amount(email.body)
            actions.append(
                ActionItem(
                    tool="start_approval_flow",
                    params={
                        "sender": email.sender,
                        "title": email.subject,
                        "details": email.body,
                        "amount": amount,
                    },
                )
            )
            reply = "您好，您的审批请求已进入流程，后续进展会自动邮件通知您。"
            actions.append(
                ActionItem(
                    tool="send_auto_reply",
                    params={"recipient": email.sender, "subject": f"Re: {email.subject}", "body": reply},
                )
            )
            return AgentPlan(
                classification=classification,
                intents=intents,
                actions=actions,
                reply_draft=reply,
                recommended_action="发起审批流并发送确认邮件，等待审批节点处理。",
            )

        classification = "inquiry"
        intents.append(IntentItem(intent="send_reply", confidence=0.65, reason="Default informative response"))
        reply = "您好，邮件已收到，我们会尽快处理并回复您。"
        actions.append(
            ActionItem(
                tool="send_auto_reply",
                params={"recipient": email.sender, "subject": f"Re: {email.subject}", "body": reply},
            )
        )
        return AgentPlan(
            classification=classification,
            intents=intents,
            actions=actions,
            reply_draft=reply,
            recommended_action="自动发送收悉回复，并将邮件分配给对应业务同学跟进。",
        )

    @staticmethod
    def _extract_amount(text: str) -> float | None:
        m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
        return float(m.group(1)) if m else None

    @staticmethod
    def _extract_limit(text: str, default: int = 10) -> int:
        lowered = text.lower()
        if ("最新" in text or "latest" in lowered) and any(k in text for k in ["一封", "1封", "一条", "1条", "一份", "1份"]):
            return 1

        m = re.search(r"(\d{1,3})", text)
        if not m:
            zh_map = {
                "一": 1,
                "二": 2,
                "两": 2,
                "三": 3,
                "四": 4,
                "五": 5,
                "六": 6,
                "七": 7,
                "八": 8,
                "九": 9,
                "十": 10,
            }
            zh_match = re.search(r"(一|二|两|三|四|五|六|七|八|九|十)\s*(封|条|份)", text)
            if zh_match:
                return MailAgent._safe_limit(zh_map.get(zh_match.group(1), default), default=default)
            if "一封" in text or "一条" in text or "一份" in text:
                return 1
            return default
        return MailAgent._safe_limit(m.group(1), default=default)

    @staticmethod
    def _safe_limit(value: Any, default: int = 10) -> int:
        try:
            n = int(value)
        except Exception:
            n = default
        return max(1, min(20, n))

    @staticmethod
    def _extract_need_summary(text: str, default: bool = False) -> bool:
        lowered = text.lower()
        return default or any(k in lowered for k in ["总结", "概括", "摘要", "summarize", "summary"])

    @staticmethod
    def _extract_strict_mode(text: str, default: bool = False) -> bool:
        lowered = text.lower()
        keys = [
            "严格",
            "strict",
            "不回退",
            "不要回退",
            "禁止回退",
            "直接报错",
            "失败就报错",
            "imap失败就报错",
        ]
        return default or any(k in lowered for k in keys)

    @staticmethod
    def _extract_unread_only(text: str, default: bool = True) -> bool:
        lowered = text.lower()
        has_unread = any(k in lowered for k in ["未读", "unread"])
        if has_unread:
            return True
        if any(k in lowered for k in ["最新", "recent", "最近", "last"]):
            return False
        return default

    @staticmethod
    def _load_json_from_text(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return json.loads(text)

    @staticmethod
    def _tool_results_from_steps(steps: List[Any]) -> List[ToolResult]:
        results: List[ToolResult] = []
        for step in steps:
            if not isinstance(step, tuple) or len(step) != 2:
                continue
            action, observation = step
            tool_name = str(getattr(action, "tool", "unknown"))
            normalized = MailAgent._normalize_result(observation)
            success = "error" not in normalized
            results.append(ToolResult(tool=tool_name, success=success, result=normalized))
        return results

    @staticmethod
    def _normalize_result(observation: Any) -> Dict[str, Any]:
        if isinstance(observation, dict):
            return observation
        if isinstance(observation, str):
            text = observation.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {"message": text}
        return {"message": str(observation)}
