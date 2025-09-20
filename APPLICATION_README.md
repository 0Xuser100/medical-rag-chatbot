# Flask Application Entry Point (`app/application.py`)

This document details how the main Flask script orchestrates configuration, session handling, LangChain integration, and HTTP endpoints for the Medical RAG chatbot.

## 1. Imports & Global Setup

- **LangChain + RAG utilities**: `components.memory.create_session_qa_chain` supplies a fully configured conversational retriever chain on demand.
- **Configuration values**: `config.config` exposes `OPEN_AI_API_KEY`, Langfuse keys (`public_key`, `secret_key`, `host`), and other environment-driven settings.
- **Langfuse observability**: When keys are present, the module instantiates `Langfuse` and its `CallbackHandler`, wiring telemetry into every LLM invocation.
- **Flask app**: The app uses a randomly generated `secret_key` (`os.urandom(24)`) to encrypt session cookies, so conversation logs stay private per user.

## 2. Helper: `get_qa_chain_with_memory(session_messages=None)`

This wrapper keeps route handlers clean. Given the current Flask session transcript, it calls `create_session_qa_chain()` to:

1. Choose the correct memory policy (window vs. summary) based on dialogue length.
2. Load the FAISS vector store and OpenAI chat model.
3. Return a ready-to-run LangChain `ConversationalRetrievalChain` configured with the custom safety prompt and retriever.

The function is wrapped in a try/except so upstream handlers can surface a single error banner while the console logs the full trace.

## 3. Template Filter: `nl2br`

Jinja renders assistant responses in chat bubbles. The `nl2br` filter converts newline characters into `<br>` elements so multi-sentence answers preserve spacing without switching to `<pre>` blocks. It is registered by assigning `app.jinja_env.filters['nl2br'] = nl2br` immediately after definition.

## 4. `index` Route (`/`, GET & POST)

The main route supports both rendering and form submission:

1. **Session bootstrap**: On first visit, it seeds `session['messages']` with an empty list for alternating `{role, content}` dicts.
2. **Handling POST**: When a user submits a prompt (`request.form['prompt']`):
   - The message is appended to `session['messages']` as a user turn.
   - If no OpenAI key is configured, the view renders `index.html` with a descriptive error message.
   - Otherwise it retrieves a conversational QA chain via `get_qa_chain_with_memory(session.get('messages', []))`.
   - The chain is invoked with the current question while streaming callbacks to Langfuse when enabled: `qa_chain.invoke({"question": user_input}, config={"callbacks": [langfuse_handler]})`.
   - The assistant's answer (from the `answer` field) is appended to the session log before redirecting back to GET. Redirect-after-POST keeps the browser refresh-safe.
3. **Rendering GET**: Simply passes the session conversation into `templates/index.html`, where messages are looped over and styled.

## 5. Error Management

- The POST branch is wrapped in a try/except; any failure prints the traceback for developers.
- User-facing errors are categorized: missing API credentials, FAISS/vector problems, or generic exceptions. Each renders the template with a friendly banner while preserving conversation history.
- Because every exception flows through `CustomException`, log messages preserve the origin file and line, simplifying debugging in production logs.

## 6. Support Routes

- **`/clear`**: Removes `messages` and `conversation_id` from the session, then redirects to the home page. The next request regenerates a new QA chain with fresh memory.
- **`/export`**: Serializes the chat history into a text/plain response, labeling user and assistant turns and forcing a download via `Content-Disposition: attachment`.

## 7. Development Server Configuration

The `if __name__ == "__main__"` guard runs `app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)`. This exposes Flask on all interfaces for Docker/App Runner, enables automatic reloads on code changes, and surfaces rich error stack traces during development.

## 8. Extension Points

- Add new API endpoints by importing `get_qa_chain_with_memory()` and reusing the session/message pipeline.
- Plug in alternative observability callbacks by modifying the Langfuse handler list in `qa_chain.invoke`.
- Wrap the Flask app in a production WSGI server (Gunicorn, uvicorn) by importing `app` from this module.

Understanding `app/application.py` clarifies how UI requests translate into LangChain chain invocations, how memory is preserved between turns, and where to hook in new functionality as the Medical RAG assistant evolves.
