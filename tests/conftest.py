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
    """Convert Anthropic billing / quota errors into xfails so CI stays green."""
    outcome = yield
    if outcome.excinfo:
        exc = outcome.excinfo[1]
        msg = str(exc)
        if "credit balance is too low" in msg or "quota" in msg.lower():
            outcome.force_exception(
                pytest.skip.Exception(
                    "Anthropic API credits exhausted — "
                    "top up at console.anthropic.com/settings/billing"
                )
            )


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
