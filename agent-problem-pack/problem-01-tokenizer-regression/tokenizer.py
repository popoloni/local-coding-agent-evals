def tokenize(text):
    return [part.lower() for part in text.strip().split(",")]
