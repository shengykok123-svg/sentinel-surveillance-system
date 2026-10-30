"""
System health and resource monitoring API routes for the Sentinel backend.

This module provides endpoints for operational monitoring.  It does **not**
use a shared router prefix because the health-check sits at the root level
while the resource endpoint lives under ``/api``:

* **GET /health**         -- Lightweight health check that always returns
  ``{"status": "ok"}`` if the server is running.
* **GET /api/resources**  -- Detailed system metrics including CPU, RAM,
  GPU utilisation, disk usage, and network throughput.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_resource_service, get_current_user
from app.services.resource_service import ResourceService


router = APIRouter(tags=["system"])


@router.get("/health")
async def health():
    """Simple health check endpoint.

    **HTTP Method:** GET
    **Path:** /health
    **Auth required:** No

    Returns a minimal JSON object to confirm the server process is alive
    and able to handle requests.  Intended for use by load balancers,
    container orchestrators (e.g. Docker health checks, Kubernetes
    liveness probes), and uptime monitors.

    Returns:
        dict: ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@router.get("/api/resources", dependencies=[Depends(get_current_user)])
async def get_resources(resource_service: ResourceService = Depends(get_resource_service)):
    """Return current system resource utilisation metrics.

    **HTTP Method:** GET
    **Path:** /api/resources
    **Auth required:** Yes (Bearer token)

    Collects and returns a snapshot of system-level metrics via the
    :class:`ResourceService`, including:

    * **CPU** -- Per-core and aggregate utilisation percentages.
    * **RAM** -- Total, used, and available memory.
    * **GPU** -- Utilisation, memory usage, and temperature (if a
      compatible GPU is present).
    * **Disk** -- Usage statistics for the primary storage volume.
    * **Network** -- Bytes sent/received and throughput rates.

    Returns:
        dict: A comprehensive resource-metrics payload.
    """
    return resource_service.get_resources()
