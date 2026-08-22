"""Ariadne fine-tuning pipeline.

Runs entirely outside the Ariadne web service. It reads versioned dataset files
exported by ``app.active_learning`` and produces a LoRA adapter plus an
evaluation report. It never connects to Ariadne's database and never promotes a
model on its own.

Importing this package pulls in no machine-learning dependency: transformers,
peft, trl, bitsandbytes and torch are imported lazily inside the functions that
need them.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
