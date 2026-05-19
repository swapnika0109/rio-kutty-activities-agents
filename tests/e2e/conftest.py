"""
E2E test fixtures for the rio-kutty story pipeline.

Fixture chain:
  session: test_bucket, retry_limit
  function: firestore_test_client (per-test to keep gRPC channels loop-aligned)
  function (autouse): cleanup_firestore, cleanup_storage
  session: report_writer plugin registration

Tests run against the `(default)` Firestore database — the same one workflows
write to. Cleanup only deletes IDs each test tracks in its own `created_*_ids`
list, so pre-existing data is never touched.
"""

from __future__ import annotations

import os
import pytest

from google.cloud import firestore as _firestore
from google.cloud import storage as _storage

from tests.e2e.helpers.firestore_helper import bulk_cleanup
from tests.e2e.helpers.storage_helper import delete_test_blobs, get_test_bucket


# ---------------------------------------------------------------------------
# pytest plugin registration
# ---------------------------------------------------------------------------

pytest_plugins = ["tests.e2e.helpers.report_writer"]


# ---------------------------------------------------------------------------
# Tests run against the same `(default)` database the workflows write to, so
# they can read what was just produced. Cleanup fixtures only delete docs
# whose IDs were appended during the test (per-test tracking lists), so
# pre-existing data in `(default)` is not touched.
# ---------------------------------------------------------------------------

@pytest.fixture
async def firestore_test_client():
    """Async Firestore client connected to the `(default)` database — same one the workflows use.

    Function-scoped (not session-scoped) because pytest-asyncio creates a fresh
    event loop per test. A session-scoped AsyncClient holds gRPC channels bound
    to the first test's loop; reusing it in later tests triggers
    `AttributeError: 'InterceptedUnaryUnaryCall' object has no attribute
    '_interceptors_task'` from grpc.aio's GC when those loops close. Per-test
    clients cost ~0.5s of setup but make the gRPC lifecycle clean."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "riokutty")
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        client = _firestore.AsyncClient.from_service_account_json(
            credentials_path, project=project, database="(default)"
        )
    else:
        client = _firestore.AsyncClient(project=project, database="(default)")
    try:
        yield client
    finally:
        # Close the transport on the same loop it was opened on, before
        # pytest-asyncio tears the loop down. Suppresses the grpc.aio
        # `_interceptors_task` AttributeError from GC against a closed loop.
        close = getattr(client, "close", None)
        if close is not None:
            try:
                result = close()
                import asyncio
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Session-scoped: GCS bucket handle
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_bucket():
    """GCS bucket handle for kutty_bucket."""
    return get_test_bucket()


# ---------------------------------------------------------------------------
# Session-scoped: configurable retry limit
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def retry_limit():
    """Returns the configured PARALLEL_WORKFLOW_MAX_RETRIES (default 4)."""
    from src.utils.config import get_settings
    return get_settings().PARALLEL_WORKFLOW_MAX_RETRIES


# ---------------------------------------------------------------------------
# Function-scoped: resource tracking + autouse cleanup
# ---------------------------------------------------------------------------

@pytest.fixture
def created_topic_ids() -> list[tuple[str, str]]:
    """Tracks (theme, topics_id) tuples created during a test."""
    return []


@pytest.fixture
def created_story_ids() -> list[tuple[str, str]]:
    """Tracks (theme, story_id) tuples created during a test."""
    return []


@pytest.fixture
def created_activity_story_ids() -> list[str]:
    """Tracks story_ids whose activities_v1 docs should be deleted."""
    return []


@pytest.fixture
def created_checkpoint_ids() -> list[str]:
    """Tracks LangGraph thread_ids to delete from workflow_checkpoints."""
    return []


@pytest.fixture
def created_gcs_blobs() -> list[str]:
    """Tracks GCS blob paths (under test/) created during a test."""
    return []


@pytest.fixture(autouse=True)
async def cleanup_firestore(
    firestore_test_client,
    created_topic_ids,
    created_story_ids,
    created_activity_story_ids,
    created_checkpoint_ids,
):
    """Delete all Firestore docs created during the test after it completes."""
    yield
    await bulk_cleanup(
        firestore_test_client,
        topic_ids=created_topic_ids,
        story_ids=created_story_ids,
        activity_story_ids=created_activity_story_ids,
        checkpoint_thread_ids=created_checkpoint_ids,
    )


@pytest.fixture(autouse=True)
async def cleanup_storage(test_bucket, created_gcs_blobs):
    """Delete all GCS blobs created under test/images/ and test/audio/ during the test."""
    yield
    if created_gcs_blobs:
        for blob_path in created_gcs_blobs:
            blob = test_bucket.blob(blob_path)
            if blob.exists():
                blob.delete()
    else:
        # Fallback: sweep all test/ prefixes in case tracking was not used
        await delete_test_blobs(test_bucket)
