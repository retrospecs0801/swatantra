import ollama, json, re

MODEL = "gemma4:e4b"

def ask_json(prompt, retries=3):
    """Ask the model for JSON, parse it safely, retry if it's malformed."""
    for attempt in range(retries):
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt + "\n\nReturn ONLY valid JSON. No explanation, no markdown fences."}],
            options={"temperature": 0},
        )
        raw = response["message"]["content"]
        raw = re.sub(r"```(?:json)?", "", raw).strip()      # strip ``` fences
        start, end = raw.find("{"), raw.rfind("}")          # keep first { .. last }
        try:
            return json.loads(raw[start:end + 1])
        except (ValueError, json.JSONDecodeError):
            print(f"  (attempt {attempt + 1}: bad JSON, retrying)")
    raise ValueError("Model never returned valid JSON:\n" + raw)