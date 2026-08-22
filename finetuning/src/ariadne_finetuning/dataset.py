"""Dataset loading and validation.

The pipeline only ever reads files that Ariadne's exporter produced: three
conversational JSONL splits, a manifest and the authoritative target JSON
Schema. Nothing here talks to the application database.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLITS: tuple[str, ...] = ("train", "validation", "test")

_NON_WORD = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class DatasetError(ValueError):
    """The exported dataset is missing, malformed or internally inconsistent."""


@dataclass(frozen=True)
class ChatSample:
    """One conversational SFT sample."""

    prompt: str
    completion: str

    @property
    def request(self) -> str:
        """The user's own words, without the exporter's boilerplate context.

        Every exported prompt ends with the same dataset catalogue and
        instruction block. Comparing full prompts would make unrelated
        questions look like near-duplicates of each other.
        """
        return self.prompt.split("\n\n", 1)[0].strip() or self.prompt.strip()

    @property
    def prompt_key(self) -> str:
        return hashlib.sha256(_normalize(self.request).encode("utf-8")).hexdigest()

    def reference(self) -> dict[str, Any]:
        """Parsed assistant target."""
        payload = json.loads(self.completion)
        if not isinstance(payload, dict):
            raise DatasetError("assistant target must be a JSON object")
        return payload

    def as_messages(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.completion},
            ]
        }


@dataclass(frozen=True)
class DatasetBundle:
    """Three frozen splits plus their provenance."""

    dataset_version: str
    task: str
    manifest: dict[str, Any]
    target_schema: dict[str, Any]
    train: tuple[ChatSample, ...]
    validation: tuple[ChatSample, ...]
    test: tuple[ChatSample, ...]

    def split(self, name: str) -> tuple[ChatSample, ...]:
        if name not in SPLITS:
            raise DatasetError(f"unknown split '{name}'")
        return {"train": self.train, "validation": self.validation, "test": self.test}[name]

    @property
    def test_digest(self) -> str:
        """Fingerprint of the frozen evaluation set.

        Baseline and post-training reports must carry the same value, or the
        comparison between them is meaningless.
        """
        digest = hashlib.sha256()
        for sample in self.test:
            digest.update(sample.prompt.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(sample.completion.encode("utf-8"))
            digest.update(b"\x1e")
        return digest.hexdigest()


def _normalize(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def load_jsonl(path: Path) -> tuple[ChatSample, ...]:
    """Read one conversational JSONL split."""
    if not path.exists():
        raise DatasetError(f"missing dataset file: {path}")
    samples: list[ChatSample] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}:{number} is not valid JSON") from exc
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list) or len(messages) != 2:
            raise DatasetError(f"{path.name}:{number} must hold exactly two messages")
        user, assistant = messages
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise DatasetError(f"{path.name}:{number} must be a user/assistant pair")
        samples.append(
            ChatSample(prompt=str(user["content"]), completion=str(assistant["content"]))
        )
    return tuple(samples)


def load_bundle(directory: Path, *, task: str, dataset_version: str) -> DatasetBundle:
    """Load the split files, manifest and target schema for one version."""
    stem = f"{task}_{dataset_version}"
    manifest_path = directory / f"{stem}.manifest.json"
    schema_path = directory / f"{stem}.schema.json"
    if not manifest_path.exists():
        raise DatasetError(f"missing manifest: {manifest_path}")
    if not schema_path.exists():
        raise DatasetError(f"missing target schema: {schema_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    bundle = DatasetBundle(
        dataset_version=dataset_version,
        task=task,
        manifest=manifest,
        target_schema=schema,
        train=load_jsonl(directory / f"{stem}.train.jsonl"),
        validation=load_jsonl(directory / f"{stem}.validation.jsonl"),
        test=load_jsonl(directory / f"{stem}.test.jsonl"),
    )
    if manifest.get("dataset_version") != dataset_version:
        raise DatasetError("manifest dataset_version does not match the requested version")
    return bundle


def leakage_report(bundle: DatasetBundle, *, threshold: float = 0.85) -> list[str]:
    """Report any evaluation example that also appears in training.

    Ariadne's exporter already splits by group, so a finding here means the
    dataset was assembled by hand or by an older exporter.
    """
    findings: list[str] = []
    train_keys = {sample.prompt_key for sample in bundle.train}
    train_requests = [
        (_tokens(sample.request), _numbers(sample.request)) for sample in bundle.train
    ]

    for name in ("validation", "test"):
        for index, sample in enumerate(bundle.split(name)):
            if sample.prompt_key in train_keys:
                findings.append(f"{name}[{index}] duplicates a training prompt")
                continue
            tokens = _tokens(sample.request)
            numbers = _numbers(sample.request)
            # Differing numbers (radius, limits) mean a different expected plan,
            # which mirrors how the exporter groups candidates.
            if any(
                other_numbers == numbers and _jaccard(tokens, other_tokens) >= threshold
                for other_tokens, other_numbers in train_requests
            ):
                findings.append(f"{name}[{index}] is a near-duplicate of a training prompt")
    return findings


def validate_bundle(bundle: DatasetBundle) -> list[str]:
    """Structural checks run before any GPU time is spent."""
    problems: list[str] = []
    if not bundle.train:
        problems.append("the training split is empty")
    if not bundle.test:
        problems.append("the test split is empty; there would be nothing to evaluate")
    for name in SPLITS:
        for index, sample in enumerate(bundle.split(name)):
            try:
                sample.reference()
            except (json.JSONDecodeError, DatasetError):
                problems.append(f"{name}[{index}] assistant target is not a JSON object")
    problems.extend(leakage_report(bundle))
    return problems


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_normalize(text).split())


def _numbers(text: str) -> frozenset[str]:
    return frozenset(_NUMBER.findall(text))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))
