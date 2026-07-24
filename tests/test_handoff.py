from __future__ import annotations

import pytest
import respx
import httpx

from champollion_agents.handoff import HandoffPayload, post_handoff


@pytest.mark.unit
def test_handoff_payload_serialises():
    p = HandoffPayload(
        output_dir="/data/out",
        combined_embeddings="/data/out/combined",
        run_id="abc-123",
        dataset="TEST",
        n_subjects=42,
    )
    d = p.model_dump()
    assert d["event"] == "pipeline_complete"
    assert d["n_subjects"] == 42


@pytest.mark.unit
async def test_post_handoff_sends_correct_payload():
    payload = HandoffPayload(
        output_dir="/data/out",
        combined_embeddings="/data/out/combined",
        run_id="abc-123",
        dataset="TEST",
        n_subjects=42,
    )
    with respx.mock:
        route = respx.post("http://localhost:8002/runs").mock(
            return_value=httpx.Response(202, json={"run_id": "analyst-run-1"})
        )
        result = await post_handoff(payload, analyst_url="http://localhost:8002")

    assert route.called
    assert result == {"run_id": "analyst-run-1"}
    sent = route.calls[0].request
    import json
    body = json.loads(sent.content)
    assert body["message"]["event"] == "pipeline_complete"
    assert body["message"]["dataset"] == "TEST"
