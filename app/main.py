from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .agent import MailAgent
from .llm_client import LLMClient
from .mail_client import MailClient, MailboxConfig
from .models import (
    EmailInput,
    ExecuteEmailActionInput,
    ExecuteEmailActionResponse,
    InboxClassifyResponse,
    InboxProcessResponse,
    MailboxConfigInput,
    ProcessEmailResponse,
    UserCommandInput,
    UserCommandResponse,
)
from .workflows import MailWorkflow

load_dotenv()

app = FastAPI(title="LLM Mail Agent", version="1.0.0")

llm_client = LLMClient()
agent = MailAgent(llm=llm_client)
workflow = MailWorkflow(agent=agent)
STATIC_DIR = Path(__file__).parent / "static"
mailbox_config: MailboxConfig | None = None


def _build_mailbox_config(payload: MailboxConfigInput) -> MailboxConfig:
    return MailboxConfig(
        email_address=payload.email_address,
        app_password=payload.app_password,
        imap_host=payload.imap_host,
        imap_port=payload.imap_port,
        smtp_host=payload.smtp_host,
        smtp_port=payload.smtp_port,
        pop3_host=payload.pop3_host,
        pop3_port=payload.pop3_port,
        use_pop3_fallback=payload.use_pop3_fallback,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/workspace")
def workspace_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "workspace.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_enabled": llm_client.enabled, "model": llm_client.model}


@app.post("/process-email", response_model=ProcessEmailResponse)
def process_email(payload: EmailInput) -> ProcessEmailResponse:
    try:
        return workflow.process_email(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"process_email failed: {e}")


@app.post("/configure-mailbox")
def configure_mailbox(payload: MailboxConfigInput) -> dict:
    global mailbox_config
    # Save config only. Validation is handled by /validate-mailbox
    # to avoid repeated login attempts and intermittent auth throttling.
    mailbox_config = _build_mailbox_config(payload)
    return {"status": "ok", "email_address": payload.email_address}


@app.post("/login-mailbox")
def login_mailbox(payload: MailboxConfigInput) -> dict:
    """
    Login flow for frontend:
    1) validate credentials
    2) save mailbox config only when validation succeeds
    """
    global mailbox_config
    config = _build_mailbox_config(payload)
    validation = MailClient(config).validate_credentials()
    if not validation.get("success"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "mailbox login failed",
                **validation,
            },
        )
    mailbox_config = config
    return {"status": "ok", "email_address": payload.email_address}


@app.post("/logout-mailbox")
def logout_mailbox() -> dict:
    global mailbox_config
    mailbox_config = None
    return {"status": "ok"}


@app.get("/mailbox-status")
def mailbox_status() -> dict:
    if mailbox_config is None:
        return {"logged_in": False}
    return {"logged_in": True, "email_address": mailbox_config.email_address}


@app.post("/validate-mailbox")
def validate_mailbox(payload: MailboxConfigInput) -> dict:
    config = _build_mailbox_config(payload)
    return MailClient(config).validate_credentials()


@app.post("/process-inbox", response_model=InboxProcessResponse)
def process_inbox(limit: int = 5) -> InboxProcessResponse:
    if mailbox_config is None:
        raise HTTPException(status_code=400, detail="Mailbox is not configured")
    try:
        client = MailClient(mailbox_config)
        return workflow.process_inbox(mail_client=client, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"process_inbox failed: {e}")


@app.post("/classify-inbox", response_model=InboxClassifyResponse)
def classify_inbox(limit: int = 10) -> InboxClassifyResponse:
    if mailbox_config is None:
        raise HTTPException(status_code=400, detail="Mailbox is not configured")
    try:
        client = MailClient(mailbox_config)
        return workflow.classify_inbox(mail_client=client, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"classify_inbox failed: {e}")


@app.post("/execute-email-action", response_model=ExecuteEmailActionResponse)
def execute_email_action(payload: ExecuteEmailActionInput) -> ExecuteEmailActionResponse:
    try:
        client = MailClient(mailbox_config) if mailbox_config is not None else None
        return workflow.execute_email_action(
            email_input=payload.email,
            send_reply=payload.send_reply,
            mail_client=client,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"execute_email_action failed: {e}")


@app.post("/command", response_model=UserCommandResponse)
def command(payload: UserCommandInput) -> UserCommandResponse:
    if mailbox_config is None:
        raise HTTPException(status_code=400, detail="Mailbox is not configured")
    try:
        client = MailClient(mailbox_config)
        return workflow.execute_user_command(command=payload.command, mail_client=client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"command failed: {e}")
