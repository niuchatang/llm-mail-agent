from __future__ import annotations

import email
import imaplib
import poplib
import smtplib
from dataclasses import dataclass
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from email.utils import parseaddr
from typing import List


@dataclass
class MailboxConfig:
    email_address: str
    app_password: str
    imap_host: str
    imap_port: int
    smtp_host: str
    smtp_port: int
    pop3_host: str
    pop3_port: int
    use_pop3_fallback: bool = True


@dataclass
class IncomingMail:
    uid: str
    subject: str
    sender: str
    body: str


class MailClient:
    def __init__(self, config: MailboxConfig) -> None:
        self.config = config

    def fetch_unread(self, limit: int = 5) -> List[IncomingMail]:
        try:
            return self._fetch_unread_by_imap(limit=limit)
        except Exception as e:
            msg = str(e).lower()
            # For 163/NetEase unsafe-login scenarios, fallback to POP3 if enabled.
            if self.config.use_pop3_fallback and ("unsafe login" in msg or "failed to select mailbox" in msg):
                return self._fetch_recent_by_pop3(limit=limit)
            raise

    def fetch_unread_strict(self, limit: int = 5) -> List[IncomingMail]:
        """
        Strict unread mode: IMAP UNSEEN only, no POP3 fallback.
        """
        return self._fetch_unread_by_imap(limit=limit)

    def fetch_recent(self, limit: int = 5) -> List[IncomingMail]:
        """
        Fetch recent emails (not unread-only). Prefer IMAP ALL, fallback to POP3.
        """
        try:
            return self._fetch_recent_by_imap(limit=limit)
        except Exception:
            return self._fetch_recent_by_pop3(limit=limit)

    def _fetch_unread_by_imap(self, limit: int = 5) -> List[IncomingMail]:
        mails: List[IncomingMail] = []
        with imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port) as imap:
            imap.login(self.config.email_address, self.config.app_password)
            self._select_mailbox(imap)
            status, data = imap.search(None, "UNSEEN")
            if status != "OK" or not data:
                return mails

            ids = data[0].split()
            for uid in ids[-limit:]:
                status, msg_data = imap.fetch(uid, "(RFC822)")
                if status != "OK" or not msg_data:
                    continue
                raw = None
                for item in msg_data:
                    if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
                        raw = bytes(item[1])
                        break
                if raw is None:
                    continue
                parsed = email.message_from_bytes(raw)
                subject = self._decode_header(parsed.get("Subject", ""))
                sender = parseaddr(parsed.get("From", ""))[1] or parsed.get("From", "")
                body = self._extract_text_body(parsed)
                mails.append(IncomingMail(uid=uid.decode(), subject=subject, sender=sender, body=body))
                # Mark as seen once fetched to avoid duplicate processing loops.
                imap.store(uid, "+FLAGS", "\\Seen")
        return mails

    def _fetch_recent_by_pop3(self, limit: int = 5) -> List[IncomingMail]:
        mails: List[IncomingMail] = []
        pop = poplib.POP3_SSL(self.config.pop3_host, self.config.pop3_port, timeout=20)
        try:
            pop.user(self.config.email_address)
            pop.pass_(self.config.app_password)
            _, listing, _ = pop.list()
            # POP3 has no standard unread flag; fetch recent N messages only.
            ids = [int(line.decode().split(" ")[0]) for line in listing][-limit:]
            for mid in ids:
                _, lines, _ = pop.retr(mid)
                raw = b"\r\n".join(lines)
                parsed = email.message_from_bytes(raw)
                subject = self._decode_header(parsed.get("Subject", ""))
                sender = parseaddr(parsed.get("From", ""))[1] or parsed.get("From", "")
                body = self._extract_text_body(parsed)
                mails.append(IncomingMail(uid=f"POP3-{mid}", subject=subject, sender=sender, body=body))
        finally:
            try:
                pop.quit()
            except Exception:
                pass
        return mails

    def _fetch_recent_by_imap(self, limit: int = 5) -> List[IncomingMail]:
        mails: List[IncomingMail] = []
        with imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port) as imap:
            imap.login(self.config.email_address, self.config.app_password)
            self._select_mailbox(imap)
            status, data = imap.search(None, "ALL")
            if status != "OK" or not data:
                return mails

            ids = data[0].split()
            for uid in ids[-limit:]:
                status, msg_data = imap.fetch(uid, "(RFC822)")
                if status != "OK" or not msg_data:
                    continue
                raw = None
                for item in msg_data:
                    if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
                        raw = bytes(item[1])
                        break
                if raw is None:
                    continue
                parsed = email.message_from_bytes(raw)
                subject = self._decode_header(parsed.get("Subject", ""))
                sender = parseaddr(parsed.get("From", ""))[1] or parsed.get("From", "")
                body = self._extract_text_body(parsed)
                mails.append(IncomingMail(uid=uid.decode(), subject=subject, sender=sender, body=body))
        return mails

    def validate_credentials(self) -> dict:
        imap_ok = False
        smtp_ok = False
        pop3_ok = False
        imap_error = ""
        smtp_error = ""
        pop3_error = ""

        try:
            with imaplib.IMAP4_SSL(self.config.imap_host, self.config.imap_port) as imap:
                imap.login(self.config.email_address, self.config.app_password)
                imap.select("INBOX")
            imap_ok = True
        except Exception as e:
            imap_error = str(e)
            if self.config.use_pop3_fallback:
                try:
                    pop = poplib.POP3_SSL(self.config.pop3_host, self.config.pop3_port, timeout=20)
                    try:
                        pop.user(self.config.email_address)
                        pop.pass_(self.config.app_password)
                    finally:
                        try:
                            pop.quit()
                        except Exception:
                            pass
                    pop3_ok = True
                except Exception as pe:
                    pop3_error = str(pe)

        try:
            with smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port) as smtp:
                smtp.login(self.config.email_address, self.config.app_password)
            smtp_ok = True
        except Exception as e:
            smtp_error = str(e)

        return {
            "imap_ok": imap_ok,
            "smtp_ok": smtp_ok,
            "pop3_ok": pop3_ok,
            "success": (imap_ok or pop3_ok) and smtp_ok,
            "imap_error": imap_error,
            "smtp_error": smtp_error,
            "pop3_error": pop3_error,
            "used_pop3_fallback": self.config.use_pop3_fallback,
        }

    @staticmethod
    def _select_mailbox(imap: imaplib.IMAP4_SSL) -> None:
        """
        Some providers/locales may not accept a single mailbox token.
        Try common choices and provide explicit error when none succeeds.
        """
        candidates = ["INBOX", "Inbox", "INBOX.", ""]
        last_error = ""
        for mailbox in candidates:
            try:
                if mailbox:
                    status, data = imap.select(mailbox)
                else:
                    status, data = imap.select()
                if status == "OK":
                    return
                last_error = f"status={status}, data={data}"
            except Exception as e:
                last_error = str(e)
        raise RuntimeError(f"failed to select mailbox: {last_error}")

    def send_reply(self, to_email: str, subject: str, body: str) -> None:
        to_addr = parseaddr(to_email)[1] or to_email
        msg = EmailMessage()
        msg["From"] = self.config.email_address
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port) as smtp:
            smtp.login(self.config.email_address, self.config.app_password)
            smtp.send_message(msg)

    @staticmethod
    def _decode_header(value: str) -> str:
        try:
            return str(make_header(decode_header(value)))
        except Exception:
            return value

    @staticmethod
    def _extract_text_body(msg: Message) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition", ""))
                if ctype == "text/plain" and "attachment" not in disp:
                    raw_payload = part.get_payload(decode=True)
                    payload = bytes(raw_payload) if isinstance(raw_payload, (bytes, bytearray)) else b""
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        return payload.decode(charset, errors="replace")
                    except Exception:
                        return payload.decode("utf-8", errors="replace")
            return ""
        raw_payload = msg.get_payload(decode=True)
        payload = bytes(raw_payload) if isinstance(raw_payload, (bytes, bytearray)) else b""
        charset = msg.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except Exception:
            return payload.decode("utf-8", errors="replace")
