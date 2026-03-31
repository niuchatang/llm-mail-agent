from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


EmailCategory = Literal[
    "verification_code",
    "advertisement",
    "notification",
    "inquiry",
    "incident",
    "approval",
    "complaint",
    "other",
]
IntentType = Literal[
    "create_ticket",
    "start_approval",
    "send_reply",
    "request_more_info",
    "none",
]
CommandIntent = Literal[
    "classify_inbox",
    "process_inbox",
    "read_unread",
    "help",
    "unknown",
]


class EmailInput(BaseModel):
    subject: str = Field(..., description="邮件主题")
    sender: str = Field(..., description="发件人")
    body: str = Field(..., description="邮件正文")


class IntentItem(BaseModel):
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ActionItem(BaseModel):
    tool: str
    params: Dict[str, Any]


class ToolResult(BaseModel):
    tool: str
    success: bool
    result: Dict[str, Any]


class AgentPlan(BaseModel):
    classification: EmailCategory
    intents: List[IntentItem]
    actions: List[ActionItem]
    reply_draft: str
    recommended_action: str


class ProcessEmailResponse(BaseModel):
    classification: EmailCategory
    intents: List[IntentItem]
    actions: List[ActionItem]
    reply_draft: str
    recommended_action: str
    tool_results: List[ToolResult]


class MailboxConfigInput(BaseModel):
    email_address: str
    app_password: str
    imap_host: str = "imap.163.com"
    imap_port: int = 993
    smtp_host: str = "smtp.163.com"
    smtp_port: int = 465
    pop3_host: str = "pop.163.com"
    pop3_port: int = 995
    use_pop3_fallback: bool = True


class InboxProcessItem(BaseModel):
    uid: str
    subject: str
    sender: str
    classification: EmailCategory
    reply_sent: bool
    workflow_result: ProcessEmailResponse


class InboxProcessResponse(BaseModel):
    processed_count: int
    items: List[InboxProcessItem]


class InboxClassifyItem(BaseModel):
    uid: str
    subject: str
    sender: str
    body: str
    classification: EmailCategory
    intents: List[IntentItem]
    actions: List[ActionItem]
    reply_draft: str
    recommended_action: str


class InboxClassifyResponse(BaseModel):
    classified_count: int
    items: List[InboxClassifyItem]


class ExecuteEmailActionInput(BaseModel):
    email: EmailInput
    send_reply: bool = True


class ExecuteEmailActionResponse(BaseModel):
    workflow_result: ProcessEmailResponse
    reply_sent: bool


class CommandPlan(BaseModel):
    intent: CommandIntent
    confidence: float = Field(ge=0.0, le=1.0)
    params: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: str


class UserCommandInput(BaseModel):
    command: str = Field(..., description="用户自然语言命令")


class UserCommandResponse(BaseModel):
    command: str
    intent: CommandIntent
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_action: str
    executed: bool
    execution_result: Dict[str, Any]
