from rag.prompts import SUMMARIZE

def test_prompts():
    assert "{context}" in SUMMARIZE
    assert "{question}" in SUMMARIZE
