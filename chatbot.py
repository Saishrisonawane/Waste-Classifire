"""
chatbot.py — EcoScan Ollama Chatbot
====================================
Sends waste classification results to a local Ollama model
and returns disposal / recycling advice.

Make sure Ollama is running:
    ollama serve
    ollama pull llama3.2   (or mistral / gemma2 etc.)

Usage (standalone test):
    python chatbot.py
"""

import requests
import json

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"          # change to any model you have pulled


def build_prompt(label: str, confidence: float, user_question: str = "") -> str:
    """
    Build a context-rich prompt for the LLM based on scan result.
    """
    category_context = {
        "Organic": (
            "The waste item has been classified as ORGANIC (biodegradable). "
            "Organic waste includes food scraps, vegetable peels, fruit rinds, "
            "eggshells, coffee grounds, tea bags, garden waste, leaves, and similar matter."
        ),
        "Recyclable": (
            "The waste item has been classified as RECYCLABLE. "
            "Recyclable waste includes plastic bottles, cardboard, paper, glass bottles, "
            "metal cans, aluminium foil, tetra packs, and similar materials."
        ),
    }

    context = category_context.get(label, "The waste type is unknown.")

    base_prompt = f"""You are EcoBot, an expert AI assistant for waste management and environmental sustainability.

Context: {context}
Classification confidence: {confidence:.1f}%

Your role:
- Give clear, actionable disposal instructions for this type of waste
- Mention specific bins or disposal methods (e.g., green bin, blue bin, compost pit)
- Share 2-3 eco-friendly tips related to this waste type
- Keep the response friendly, concise (under 200 words), and formatted with short paragraphs

"""

    if user_question.strip():
        base_prompt += f"User's specific question: {user_question.strip()}\n\nAnswer:"
    else:
        base_prompt += (
            f"Explain how to properly dispose of this {label.lower()} waste "
            f"and share eco-tips to reduce its environmental impact.\n\nAnswer:"
        )

    return base_prompt


def ask_ollama(label: str, confidence: float, user_question: str = "") -> str:
    """
    Send a prompt to the local Ollama model and stream the response.

    Returns:
        str: The model's full text response, or an error message.
    """
    prompt = build_prompt(label, confidence, user_question)

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,          # set True if you want streaming later
        "options": {
            "temperature": 0.5,
            "top_p": 0.9,
            "num_predict": 80,   # 🔥 reduce tokens (important)
        "num_ctx": 512   # max tokens
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120            # Ollama can be slow on first load
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Sorry, I couldn't generate a response.")

    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Ollama is not running. Please start it with:\n"
            "   ollama serve\n"
            "Then pull a model: ollama pull llama3.2"
        )
    except requests.exceptions.Timeout:
        return "⚠️ The AI model took too long to respond. Please try again."
    except Exception as e:
        return f"⚠️ Error communicating with Ollama: {str(e)}"


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing EcoBot with Ollama...\n")

    # Test 1: Organic
    print("=" * 50)
    print("Test 1 — Organic waste, no user question")
    print("=" * 50)
    reply = ask_ollama("Organic", 91.5)
    print(reply)

    # Test 2: Recyclable with follow-up question
    print("\n" + "=" * 50)
    print("Test 2 — Recyclable waste, with user question")
    print("=" * 50)
    reply = ask_ollama("Recyclable", 88.2, "Can I recycle dirty plastic bags?")
    print(reply)