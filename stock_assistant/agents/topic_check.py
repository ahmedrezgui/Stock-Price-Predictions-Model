"""Classify the question's topic; tool selection belongs to the assistant."""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..providers import lazy_chain


class TopicCheck(BaseModel):
    """Whether the question falls within the stock assistant's scope."""

    topic: Literal["on_topic", "off_topic"] = Field(
        description="Return on_topic for supported market questions, otherwise off_topic."
    )


system = """Classify whether the user's question is within the stock assistant's scope.
Return on_topic for Tunisian stocks, market data, financial news, the Tunisian
economy, model predictions, charts, and explanations of these topics. Use the
conversation history to interpret follow-up questions.
Return off_topic for unrelated questions.
Classify only topic relevance. Do not choose a data source or require retrieval.
For on_topic questions, the assistant will select its own tools, including
historical data, predictions, charts, web search, and optional vector-store retrieval.
"""
route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
topic_checker = lazy_chain(route_prompt, TopicCheck)
