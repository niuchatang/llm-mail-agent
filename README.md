# LLM Mail Agent

基于大模型（LLM）构建的智能邮件处理 Agent 示例系统，采用 **LangChain + Agent + Tool Calling** 架构，实现：

- 邮件分类（咨询/故障/审批/投诉/其他）
- 扩展分类：验证码邮件、广告邮件、通知邮件
- 需求识别与任务拆解（intent extraction）
- 推荐处理建议（recommended_action）
- 自动回复草稿生成
- 业务流程触发（工单创建、审批流调用）

## 架构概览

1. **Agent 层（LangChain 驱动）**：理解邮件内容，生成结构化执行计划。
2. **Tools 层**：提供可调用工具（create_ticket、start_approval_flow、send_auto_reply）。
3. **Workflow 层**：串联 “理解 -> 计划 -> 执行 -> 汇总结果”。

> 支持两种模式：
> - 有 `OPENAI_API_KEY`：通过 LangChain 的 `ChatOpenAI` 调用 OpenAI 兼容接口进行 LLM 推理
> - 无 Key：自动降级到规则引擎（便于本地演示）

## 目录

```text
llm-mail-agent/
├── app/
│   ├── agent.py
│   ├── llm_client.py
│   ├── main.py
│   ├── models.py
│   ├── tools.py
│   └── workflows.py
├── .env.example
└── requirements.txt
```

## 快速启动

```bash
cd "/Users/apple/Desktop/算法/llm-mail-agent"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

启动后可直接打开前端页面：

- [http://127.0.0.1:8001](http://127.0.0.1:8001)
- 在页面填写邮件内容并点击“处理邮件”
- 或在同一页面配置邮箱（IMAP/SMTP）后点击“处理未读邮件”

## 调用示例

```bash
curl -X POST "http://127.0.0.1:8001/process-email" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "生产系统登录失败，麻烦尽快处理",
    "sender": "alice@example.com",
    "body": "从今天上午开始无法登录后台，报错500，影响客户操作。请帮忙排查。"
  }'
```

返回示例包含：
- `classification`
- `intents`
- `actions`
- `reply_draft`
- `recommended_action`
- `tool_results`

## 真实邮箱处理流程

1. 打开网页控制台 `http://127.0.0.1:8001`
2. 填写邮箱配置：
   - `email_address`
   - `app_password`（建议邮箱应用专用密码，不要用主密码）
   - IMAP/SMTP host 和端口
3. 点击“保存邮箱配置”
4. 点击“处理未读邮件”
5. 系统会拉取未读邮件，逐封执行 Agent，并自动发送回复草稿

## 环境变量（可选）

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（默认 `https://api.openai.com/v1`）
- `OPENAI_MODEL`（默认 `gpt-4o-mini`）

未配置时，系统会自动使用规则引擎完成演示。
