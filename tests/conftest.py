from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    """Load .env and configure SSL before any test runs."""
    # httpx uses ssl.create_default_context() internally. On CEA infrastructure
    # a corporate CA intercepts TLS, so we patch create_default_context to
    # always load the custom CA bundle from SSL_CERT_FILE.
    import ssl as _ssl

    ssl_cert = os.environ.get("SSL_CERT_FILE")
    if ssl_cert and Path(ssl_cert).is_file():
        _orig_create_default_context = _ssl.create_default_context

        def _create_default_context_with_custom_ca(*args, **kwargs):
            # Drop cafile/capath so create_default_context loads system CAs.
            # SSL_CERT_FILE being set in the env causes httpx to pass cafile=
            # which SKIPS certifi (and GTS Root R4), breaking public-CA sites.
            kwargs.pop("cafile", None)
            kwargs.pop("capath", None)
            kwargs.pop("cadata", None)
            ctx = _orig_create_default_context(*args, **kwargs)
            # Add certifi's bundle so well-known public CAs (GTS Root R4 etc.)
            # are trusted even when SSL_CERT_FILE is set.
            try:
                import certifi

                ctx.load_verify_locations(cafile=certifi.where())
            except ImportError:
                pass
            # Add the corporate CA bundle last.
            ctx.load_verify_locations(cafile=ssl_cert)
            return ctx

        _ssl.create_default_context = _create_default_context_with_custom_ca

    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Convert billing / auth errors into skips so CI stays green."""
    outcome = yield
    if outcome.excinfo:
        exc = outcome.excinfo[1]
        msg = str(exc)
        skip_phrases = (
            "credit balance is too low",
            "quota",
            "authentication required",
            "not logged in",
            "invalid api key",
        )
        if any(p in msg.lower() for p in skip_phrases):
            outcome.force_exception(
                pytest.skip.Exception(f"Auth/billing issue — check credentials: {msg[:200]}")
            )


@pytest.fixture(scope="session")
def sdk_available():
    """Skip integration tests when the claude CLI is not available."""
    import shutil

    if not shutil.which("claude"):
        pytest.skip("claude CLI not found in PATH — skipping SDK integration test")


@pytest.fixture(scope="session")
def anthropic_api_key():
    """Skip integration tests when no Anthropic API key is available."""
    key = os.environ.get("CHAMPOLLION_LLM_API_KEY") or os.environ.get("CHAMPOLLION_API_KEY", "")
    if not key.startswith("sk-ant-"):
        pytest.skip("Anthropic API key not available — skipping integration test")
    return key


@pytest.fixture
def tmp_output_dir(tmp_path):
    d = tmp_path / "derivatives"
    d.mkdir()
    return d


@pytest.fixture
def fake_embeddings_csv(tmp_output_dir):
    """Create a minimal full_embeddings.csv for indexer tests."""
    import pandas as pd

    region_dir = tmp_output_dir / "combined_embeddings" / "SC-sylv_left"
    region_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {f"dim{i}": [float(i + j) for j in range(3)] for i in range(128)},
        index=["sub-001", "sub-002", "sub-003"],
    )
    df.index.name = "subject_id"
    csv_path = region_dir / "full_embeddings.csv"
    df.to_csv(csv_path)
    return tmp_output_dir / "combined_embeddings"
