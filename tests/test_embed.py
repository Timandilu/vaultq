from __future__ import annotations

from vaultq import embed


def test_contextual_text_is_bounded_by_estimated_tokens(monkeypatch) -> None:
    monkeypatch.setattr(embed, "_contextual_text_max_tokens", lambda: 24)

    text = " ".join(["abcdefgh"] * 100)

    bounded = embed._bounded_contextual_text(text)

    assert embed._estimate_embedding_tokens(bounded) <= 24
    assert len(bounded.split()) < len(text.split())


def test_contextual_batches_use_actual_outbound_text_estimate(monkeypatch) -> None:
    monkeypatch.setattr(embed, "use_contextualized_chunk_embeddings", lambda: True)
    monkeypatch.setattr(embed, "_embedding_window_tokens", lambda: 32)
    documents = [
        {"source_key": "document:1", "token_count": 1, "text": " ".join(["abcdefgh"] * 10)},
        {"source_key": "document:1", "token_count": 1, "text": " ".join(["ijklmnop"] * 10)},
    ]

    batches = embed._embedding_batches(documents)

    assert len(batches) == 2
    for batch in batches:
        assert sum(embed._document_embedding_token_count(doc) for doc in batch) <= 32
