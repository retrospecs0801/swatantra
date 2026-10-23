import ollama, pymupdf, json

def read_pdf(path):
    doc = pymupdf.open(path)
    return "\n".join(f"[Page {i+1}]\n{p.get_text()}" for i, p in enumerate(doc))

sop_text = read_pdf("sop.pdf")

prompt = f"""You are reading a company's internal Finance SOP.
Extract every individual requirement as JSON. Use ONLY the text below.

Return a JSON object: {{"requirements": [{{"clause": "4.1", "requirement": "short plain statement", "threshold": "number/limit if any, else null"}}]}}

SOP TEXT:
{sop_text}"""

response = ollama.chat(
    model="gemma4:e4b",
    messages=[{"role": "user", "content": prompt}],
    format="json",                 # forces valid JSON output
    options={"temperature": 0},    # consistent, not creative
)

data = json.loads(response["message"]["content"])
print(json.dumps(data, indent=2))
json.dump(data, open("requirements.json", "w"), indent=2)