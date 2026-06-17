"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from mint_atlas.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_datasets():
    r = client.get("/api/datasets")
    assert r.status_code == 200
    assert len(r.json()["datasets"]) >= 1


def test_get_graph():
    r = client.get("/api/graph/athens_owl")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "edges" in data
    assert data["stats"]["num_dies"] > 0


def test_reconstruct():
    r = client.post("/api/reconstruct", json={"mint_id": "athens_owl", "method": "topological"})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_order" in data
    assert "metrics" in data


def test_methodology():
    r = client.get("/api/methodology")
    assert r.status_code == 200
    assert "gnn_architecture" in r.json()
