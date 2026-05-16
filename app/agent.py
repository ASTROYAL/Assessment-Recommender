from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from . import retriever
    from .prompts import SYSTEM_PROMPT
except ImportError:
    import retriever
    from prompts import SYSTEM_PROMPT


SAFE_FALLBACK = {
    "reply": "I encountered an error. Please try again.",
    "recommendations": [],
    "end_of_conversation": False,
}
VALID_TEST_TYPES = {"A", "P", "B", "C", "K", "S"}
LOCAL_PROVIDERS = {"local", "offline", "retrieval"}
INJECTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "override your instructions",
    "system prompt",
    "developer message",
    "jailbreak",
    "act as",
)
OFF_TOPIC_PATTERNS = (
    "salary",
    "legal advice",
    "lawsuit",
    "weather",
    "stock price",
    "recipe",
    "write code",
    "debug my code",
)
SENIORITY_PATTERN = re.compile(
    r"\b(entry|junior|graduate|mid|middle|senior|lead|manager|director|executive|"
    r"supervisor|professional|experienced|front line)\b",
    re.IGNORECASE,
)
ROLE_PATTERN = re.compile(
    r"\b(analyst|developer|engineer|manager|representative|associate|consultant|"
    r"administrator|admin|accountant|designer|architect|sales|support|agent|"
    r"specialist|lead|role|job|candidate|hire|hiring)\b",
    re.IGNORECASE,
)
SKILL_PATTERN = re.compile(
    r"\b(sql|excel|python|java|javascript|analysis|analytics|numerical|reasoning|"
    r"statistics|communication|leadership|attention|detail|coding|testing|"
    r"personality|competency|simulation|accounting|finance|customer|salesforce|"
    r"cloud|aws|azure|security|data|typing|language|problem solving)\b",
    re.IGNORECASE,
)


def _last_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("Conversation history must contain at least one user message.")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _build_catalog_context(entries: list[dict]) -> str:
    lines = ["CATALOG CONTEXT (use only these for recommendations):"]
    for index, entry in enumerate(entries, start=1):
        name = _clean_text(entry.get("name"))
        url = _clean_text(entry.get("url"))
        test_type = _clean_text(entry.get("test_type"))
        description = _clean_text(entry.get("description"))
        lines.append(
            f"{index}. Name: {name} | URL: {url} | Type: {test_type} | Description: {description}"
        )
    return "\n".join(lines)


def _conversation_text(messages: list[dict]) -> str:
    return " ".join(str(message.get("content", "")) for message in messages)


def _retrieval_query(messages: list[dict]) -> str:
    user_messages = [
        _clean_text(message.get("content"))
        for message in messages
        if message.get("role") == "user" and _clean_text(message.get("content"))
    ]
    if user_messages:
        return ". ".join(user_messages)
    return _last_user_message(messages)


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def _recommendations_from_entries(entries: list[dict], top_k: int = 5) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for entry in entries:
        name = _clean_text(entry.get("name"))
        url = _clean_text(entry.get("url"))
        test_type = _clean_text(entry.get("test_type"))
        if (
            name
            and url.startswith("https://www.shl.com/")
            and test_type in VALID_TEST_TYPES
            and url not in seen_urls
        ):
            recommendations.append({"name": name, "url": url, "test_type": test_type})
            seen_urls.add(url)
        if len(recommendations) >= top_k:
            break
    return recommendations


def _missing_context_labels(text: str) -> list[str]:
    missing: list[str] = []
    if not ROLE_PATTERN.search(text):
        missing.append("role or job title")
    if not SENIORITY_PATTERN.search(text):
        missing.append("seniority level")
    if not SKILL_PATTERN.search(text):
        missing.append("key skills")
    return missing


def _local_reply(messages: list[dict], retrieved_entries: list[dict]) -> dict:
    last_user_message = _last_user_message(messages)
    full_text = _conversation_text(messages)

    if _contains_any(last_user_message, INJECTION_PATTERNS):
        return {
            "reply": "I can only help with SHL assessment selection.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    if _contains_any(last_user_message, OFF_TOPIC_PATTERNS):
        return {
            "reply": "I can only help with SHL assessment selection. Tell me the role, seniority, and skills you want to assess.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    missing = _missing_context_labels(full_text)
    should_force_recommend = len(messages) >= 6
    if missing and not should_force_recommend:
        if len(missing) == 1:
            missing_text = missing[0]
        else:
            missing_text = ", ".join(missing[:-1]) + f", and {missing[-1]}"
        return {
            "reply": f"Got it. To narrow this to the right SHL assessments, please share the {missing_text}.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    recommendations = _recommendations_from_entries(retrieved_entries, top_k=5)
    if not recommendations:
        return {
            "reply": "I could not find a strong catalog match yet. Please share the role, seniority, and skills you want to assess.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    names = ", ".join(item["name"] for item in recommendations[:3])
    extra = "" if len(recommendations) <= 3 else f", plus {len(recommendations) - 3} more"
    return {
        "reply": f"Based on the SHL catalog, I recommend starting with {names}{extra}. These are selected from the retrieved catalog matches for your role and skill requirements.",
        "recommendations": recommendations,
        "end_of_conversation": False,
    }


def _to_gemini_messages(messages: list[dict]) -> list[dict]:
    gemini_messages: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", ""))
        gemini_role = "model" if role == "assistant" else "user"
        gemini_messages.append({"role": gemini_role, "parts": [{"text": content}]})
    return gemini_messages


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_response_text(response: Any) -> str:
    try:
        return str(response.text)
    except Exception as text_error:
        primary_error = text_error

    try:
        parts = response.candidates[0].content.parts
        return "".join(getattr(part, "text", "") for part in parts)
    except Exception as exc:
        raise ValueError("Gemini response did not contain text.") from primary_error or exc


def _validate_agent_response(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Agent response must be a JSON object.")

    required_keys = {"reply", "recommendations", "end_of_conversation"}
    if not required_keys.issubset(payload.keys()):
        raise ValueError("Agent response is missing required keys.")

    reply = payload["reply"]
    recommendations = payload["recommendations"]
    end_of_conversation = payload["end_of_conversation"]

    if not isinstance(reply, str):
        raise ValueError("reply must be a string.")
    if not isinstance(recommendations, list):
        raise ValueError("recommendations must be a list.")
    if not isinstance(end_of_conversation, bool):
        raise ValueError("end_of_conversation must be a boolean.")
    if len(recommendations) > 10:
        raise ValueError("recommendations must contain at most 10 items.")

    validated_recommendations: list[dict[str, str]] = []
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            raise ValueError("Each recommendation must be an object.")
        for key in ("name", "url", "test_type"):
            if key not in recommendation or not isinstance(recommendation[key], str):
                raise ValueError("Each recommendation must include name, url, and test_type strings.")

        name = recommendation["name"].strip()
        url = recommendation["url"].strip()
        test_type = recommendation["test_type"].strip()
        if not name or not url or not test_type:
            raise ValueError("Recommendation fields cannot be empty.")
        if not url.startswith("https://www.shl.com/"):
            raise ValueError("Recommendation URL must start with https://www.shl.com/.")
        if test_type not in VALID_TEST_TYPES:
            raise ValueError("Recommendation test_type must be one of A, P, B, C, K, or S.")

        validated_recommendations.append({"name": name, "url": url, "test_type": test_type})

    return {
        "reply": reply.strip(),
        "recommendations": validated_recommendations,
        "end_of_conversation": end_of_conversation,
    }


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(env_path)
    except Exception:
        return


async def get_reply(messages: list[dict]) -> dict:
    try:
        _load_environment()
        retrieval_query = _retrieval_query(messages)
        retrieved_entries = retriever.search(retrieval_query, top_k=10)

        provider = os.getenv("AGENT_PROVIDER", "gemini").strip().lower()
        if provider in LOCAL_PROVIDERS:
            return _local_reply(messages, retrieved_entries)

        catalog_context = _build_catalog_context(retrieved_entries)
        enriched_system_prompt = f"{catalog_context}\n\n{SYSTEM_PROMPT}"
        gemini_messages = _to_gemini_messages(messages)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            system_instruction=enriched_system_prompt,
        )
        response = await asyncio.wait_for(
            model.generate_content_async(
                gemini_messages,
                generation_config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
                request_options={"timeout": 10},
            ),
            timeout=14,
        )

        response_text = _strip_markdown_fences(_extract_response_text(response))
        parsed = json.loads(response_text)
        return _validate_agent_response(parsed)
    except Exception as exc:
        print(f"Agent error: {exc}")
        try:
            retrieval_query = _retrieval_query(messages)
            return _local_reply(messages, retriever.search(retrieval_query, top_k=10))
        except Exception as fallback_exc:
            print(f"Local fallback error: {fallback_exc}")
            return dict(SAFE_FALLBACK)
