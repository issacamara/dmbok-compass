"""Content-free aggregate metrics owned by the administration surface."""

from __future__ import annotations

from hashlib import sha256
from threading import Lock
from typing import Protocol

from firebase_admin import firestore

from app.contracts.api import AggregateMetric
from app.identity import _firebase_app


class AggregateMetricRepository(Protocol):
    def list_metrics(self) -> list[AggregateMetric]: ...

    def replace(self, metric: AggregateMetric) -> AggregateMetric: ...


class InMemoryAggregateMetricRepository:
    """Deterministic aggregate store for tests and local development."""

    def __init__(self, metrics: list[AggregateMetric] | None = None) -> None:
        self._metrics = {metric.metric_name: metric for metric in metrics or []}
        self._lock = Lock()

    def list_metrics(self) -> list[AggregateMetric]:
        with self._lock:
            return sorted(self._metrics.values(), key=lambda metric: metric.metric_name)

    def replace(self, metric: AggregateMetric) -> AggregateMetric:
        """Replace one pre-aggregated metric without accepting arbitrary payloads."""
        with self._lock:
            self._metrics[metric.metric_name] = metric
        return metric


class FirestoreAggregateMetricRepository:
    """Server-only read access to the content-free aggregate metric collection."""

    def __init__(self) -> None:
        client = firestore.client(app=_firebase_app())
        self.metrics = client.collection("aggregate_metrics")

    def list_metrics(self) -> list[AggregateMetric]:
        values = [AggregateMetric.model_validate(snapshot.to_dict()) for snapshot in self.metrics.stream()]
        return sorted(values, key=lambda metric: metric.metric_name)

    def replace(self, metric: AggregateMetric) -> AggregateMetric:
        document_id = sha256(metric.metric_name.encode("utf-8")).hexdigest()
        self.metrics.document(document_id).set(metric.model_dump())
        return metric
