import tiktoken
encoding=tiktoken.encoding_for_model("gpt-4o-mini")
sentences={
    "English": "Artificial intelligence is transforming the world.",
    "Hindi": "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।",
    "code": "def add(a,b): return a+b"
}
token_counts = {}
for lang, text in sentences.items():
    tokens = encoding.encode(text)
    token_counts[lang] = len(tokens)    
    print(f"\n{lang} sentence: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    