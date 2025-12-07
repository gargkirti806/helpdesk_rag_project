STRICT_INTENT_PROMPT = """
Classify the following user query into exactly ONE of the two categories below:

- HR_Policy: office timing, working hours, attendance, leave policy, holidays, maternity/paternity leave, appraisal, salary, HR rules
- IT_guidelines: laptop issues, VPN, email, password reset, software access, system issues, network, hardware

Rules:
- Questions about office timing or working hours MUST be classified as HR_Policy.
- You MUST respond ONLY in valid JSON with the key "Intent".
- Do NOT add any explanation.

Examples:
{{"Intent": "HR_Policy"}}

Question: {question}
"""

STRICT_RAG_PROMPT = """You are a strict RAG assistant. Only answer based on the given context.
Do not use any external knowledge or make assumptions.

Context:
{context}

Question:
{question}

Return ONLY a JSON object with field {{answer}}.

"""