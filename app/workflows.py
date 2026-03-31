from __future__ import annotations

from typing import Any, Dict, List

from .agent import MailAgent
from .mail_client import IncomingMail, MailClient
from .models import (
    EmailInput,
    ExecuteEmailActionResponse,
    InboxClassifyItem,
    InboxClassifyResponse,
    InboxProcessItem,
    InboxProcessResponse,
    ProcessEmailResponse,
    UserCommandResponse,
)


class MailWorkflow:
    def __init__(self, agent: MailAgent) -> None:
        self.agent = agent

    def process_email(self, email: EmailInput) -> ProcessEmailResponse:
        return self.agent.execute(email)

    def process_inbox(self, mail_client: MailClient, limit: int = 5) -> InboxProcessResponse:
        incoming_mails: List[IncomingMail] = mail_client.fetch_unread(limit=limit)
        items: List[InboxProcessItem] = []

        for mail in incoming_mails:
            result = self.process_email(
                EmailInput(subject=mail.subject, sender=mail.sender, body=mail.body)
            )
            reply_sent = False
            if result.reply_draft.strip():
                try:
                    mail_client.send_reply(
                        to_email=mail.sender,
                        subject=f"Re: {mail.subject}",
                        body=result.reply_draft,
                    )
                    reply_sent = True
                except Exception:
                    reply_sent = False

            items.append(
                InboxProcessItem(
                    uid=mail.uid,
                    subject=mail.subject,
                    sender=mail.sender,
                    classification=result.classification,
                    reply_sent=reply_sent,
                    workflow_result=result,
                )
            )

        return InboxProcessResponse(processed_count=len(items), items=items)

    def classify_inbox(self, mail_client: MailClient, limit: int = 5) -> InboxClassifyResponse:
        incoming_mails: List[IncomingMail] = mail_client.fetch_unread(limit=limit)
        items: List[InboxClassifyItem] = []
        for mail in incoming_mails:
            plan = self.agent.plan(EmailInput(subject=mail.subject, sender=mail.sender, body=mail.body))
            items.append(
                InboxClassifyItem(
                    uid=mail.uid,
                    subject=mail.subject,
                    sender=mail.sender,
                    body=mail.body,
                    classification=plan.classification,
                    intents=plan.intents,
                    actions=plan.actions,
                    reply_draft=plan.reply_draft,
                    recommended_action=plan.recommended_action,
                )
            )
        return InboxClassifyResponse(classified_count=len(items), items=items)

    def execute_email_action(
        self,
        email_input: EmailInput,
        send_reply: bool,
        mail_client: MailClient | None = None,
    ) -> ExecuteEmailActionResponse:
        result = self.process_email(email_input)
        reply_sent = False
        if send_reply and mail_client is not None and result.reply_draft.strip():
            try:
                mail_client.send_reply(
                    to_email=email_input.sender,
                    subject=f"Re: {email_input.subject}",
                    body=result.reply_draft,
                )
                reply_sent = True
            except Exception:
                reply_sent = False
        return ExecuteEmailActionResponse(workflow_result=result, reply_sent=reply_sent)

    def execute_user_command(self, command: str, mail_client: MailClient) -> UserCommandResponse:
        plan = self.agent.plan_command(command)
        limit = self._safe_limit(plan.params.get("limit"), default=10)
        command_lower = command.lower()
        need_summary = bool(plan.params.get("need_summary", False)) or any(
            k in command_lower for k in ["总结", "概括", "摘要", "summarize", "summary"]
        )
        unread_only = bool(plan.params.get("unread_only", True))
        strict_mode = bool(plan.params.get("strict_mode", False)) or any(
            k in command_lower for k in ["严格", "strict", "不回退", "不要回退", "直接报错", "失败就报错"]
        )

        if plan.intent == "help":
            return UserCommandResponse(
                command=command,
                intent=plan.intent,
                confidence=plan.confidence,
                recommended_action=plan.recommended_action,
                executed=True,
                execution_result={
                    "examples": [
                        "帮我分类最近10封未读邮件",
                        "直接处理最近5封未读邮件",
                        "读取最近8封未读邮件主题",
                    ]
                },
            )

        if plan.intent == "classify_inbox":
            result = self.classify_inbox(mail_client=mail_client, limit=limit)
            return UserCommandResponse(
                command=command,
                intent=plan.intent,
                confidence=plan.confidence,
                recommended_action=plan.recommended_action,
                executed=True,
                execution_result=result.model_dump(),
            )

        if plan.intent == "process_inbox":
            result = self.process_inbox(mail_client=mail_client, limit=limit)
            return UserCommandResponse(
                command=command,
                intent=plan.intent,
                confidence=plan.confidence,
                recommended_action=plan.recommended_action,
                executed=True,
                execution_result=result.model_dump(),
            )

        if plan.intent == "read_unread":
            source = "imap_unseen"
            strict_unread = unread_only
            warning = ""
            if not unread_only:
                mails = mail_client.fetch_recent(limit=limit)
                source = "recent_mail"
            elif strict_mode:
                try:
                    mails = mail_client.fetch_unread_strict(limit=limit)
                except Exception as imap_error:
                    return UserCommandResponse(
                        command=command,
                        intent=plan.intent,
                        confidence=plan.confidence,
                        recommended_action=plan.recommended_action,
                        executed=False,
                        execution_result={
                            "strict_mode": True,
                            "unread_only": True,
                            "strict_unread": False,
                            "mailbox_source": "imap_unseen",
                            "error": "IMAP 严格未读读取失败，已按严格模式终止执行（未回退 POP3）。",
                            "imap_error": str(imap_error),
                        },
                    )
            else:
                try:
                    mails = mail_client.fetch_unread_strict(limit=limit)
                except Exception as imap_error:
                    mails = mail_client.fetch_unread(limit=limit)
                    source = "pop3_recent_fallback_or_non_strict"
                    strict_unread = False
                    warning = str(imap_error)
            preview = [
                {"uid": m.uid, "subject": m.subject, "sender": m.sender, "body_preview": m.body[:160]}
                for m in mails
            ]
            execution_result: Dict[str, Any] = {
                "count": len(preview),
                "items": preview,
                "mailbox_source": source,
                "strict_unread": strict_unread,
                "strict_mode": strict_mode,
                "unread_only": unread_only,
            }
            if warning:
                execution_result["warning"] = (
                    "IMAP 严格未读读取失败，已回退到 POP3 最近邮件，结果可能不等于严格未读。"
                )
                execution_result["imap_error"] = warning
            if need_summary:
                summary_text, llm_used = self._summarize_items(preview, unread_only=unread_only)
                execution_result["summary"] = summary_text
                execution_result["summary_generated"] = True
                execution_result["summary_llm_used"] = llm_used
                execution_result["llm_enabled"] = self.agent.llm.enabled
            return UserCommandResponse(
                command=command,
                intent=plan.intent,
                confidence=plan.confidence,
                recommended_action=plan.recommended_action,
                executed=True,
                execution_result=execution_result,
            )

        return UserCommandResponse(
            command=command,
            intent=plan.intent,
            confidence=plan.confidence,
            recommended_action=plan.recommended_action,
            executed=False,
            execution_result={
                "message": "无法识别命令，请尝试：分类最近10封未读邮件 / 处理最近5封未读邮件 / 读取最近8封未读邮件。"
            },
        )

    @staticmethod
    def _safe_limit(value: Any, default: int = 10) -> int:
        try:
            n = int(value)
        except Exception:
            n = default
        return max(1, min(20, n))

    def _summarize_items(self, items: List[Dict[str, str]], unread_only: bool) -> tuple[str, bool]:
        if not items:
            return ("暂无未读邮件。" if unread_only else "暂无可读取的最新邮件。"), False
        first = items[0]
        label = "最新未读邮件" if unread_only else "最新邮件"
        fallback = (
            f"{label}来自 {first.get('sender', '未知发件人')}，"
            f"主题为《{first.get('subject', '无主题')}》。"
        )
        if not self.agent.llm.enabled:
            return fallback, False

        lines: List[str] = []
        for idx, item in enumerate(items[:3], start=1):
            lines.append(
                f"{idx}. sender={item.get('sender','')}; subject={item.get('subject','')}; "
                f"body_preview={item.get('body_preview','')}"
            )
        scope_text = "未读邮件" if unread_only else "最新邮件"
        prompt = (
            f"请基于以下{scope_text}内容给出简短中文总结（2-3句），"
            "包含：邮件性质、风险或建议动作。不要输出JSON。\n\n"
            + "\n".join(lines)
        )
        try:
            text = self.agent.llm.chat(
                messages=[
                    {"role": "system", "content": "你是邮件总结助手，输出简洁、准确、可执行的中文总结。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            final_text = text.strip() or fallback
            return final_text, True
        except Exception:
            return fallback, False
