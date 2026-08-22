"""Language model provider boundary.

Domain code depends only on the types in :mod:`app.llm.contracts`; no module
outside this package may import an Ollama- or AvalAI-specific object.
"""
