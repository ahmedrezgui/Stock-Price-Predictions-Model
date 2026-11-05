"""Lazy, explicit model selection shared by the agent and its graders."""

import os
from langchain_core.runnables import RunnableLambda


def create_chat_model(*, simple=False):
    provider = os.getenv("ASSISTANT_PROVIDER", "gemini").lower()
    model = os.getenv("ASSISTANT_SIMPLE_MODEL") if simple else None
    model = model or os.getenv("ASSISTANT_MODEL")
    if not model and provider == "gemini":
        model = (os.getenv("SIMPLE_TASK_MODEL") if simple else None) or os.getenv("GENERATIVE_MODEL")
    if not model:
        raise ValueError("Set ASSISTANT_MODEL (and optionally ASSISTANT_SIMPLE_MODEL).")
    common = {"temperature": 0}
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini.")
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, timeout=60, max_retries=2, **common)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            client_kwargs={"timeout": 60},
            **common,
        )
    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"]
        if any(not os.getenv(key) for key in required):
            raise ValueError("Azure requires " + ", ".join(required))
        return AzureChatOpenAI(
            azure_deployment=model,
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            timeout=60,
            max_retries=2,
            **common,
        )
    raise ValueError("ASSISTANT_PROVIDER must be gemini, ollama or azure.")


def lazy_chain(prompt, schema=None, parser=None):
    def invoke(value, config):
        model = create_chat_model(simple=True)
        chain = prompt | (model.with_structured_output(schema) if schema else model)
        if parser is not None:
            chain = chain | parser
        return chain.invoke(value, config=config)

    return RunnableLambda(invoke)
