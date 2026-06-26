from tokenizer import tokenize


def test_empty_input_returns_empty_list():
    assert tokenize("   ") == []


def test_empty_csv_fields_are_not_tokens():
    assert tokenize("Alpha,,BETA") == ["alpha", "beta"]
