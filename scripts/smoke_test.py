"""Smoke-test a running Docker image through its public HTTP API."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _request(path: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = Request(
        f"http://127.0.0.1:8000{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")
        payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path} returned a non-object JSON response")
        return payload


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} IMAGE", file=sys.stderr)
        return 2
    image = sys.argv[1]
    container_id = subprocess.check_output(
        ["docker", "run", "-d", "-p", "8000:8000", image], text=True
    ).strip()
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                if _request("/health").get("status") == "healthy":
                    break
            except (HTTPError, OSError, TimeoutError, RuntimeError):
                time.sleep(0.5)
        else:
            raise RuntimeError("Container did not become healthy within 45 seconds")

        info = _request("/model-info")
        classes = info.get("supported_classes")
        if not isinstance(classes, list) or not classes:
            raise RuntimeError("/model-info did not return supported classes")
        prediction = _request(
            "/predict",
            "POST",
            {
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2,
            },
        ).get("prediction")
        if prediction not in classes:
            raise RuntimeError("Prediction is not one of the supported classes")
        print(f"Container smoke test passed: {prediction}")
        return 0
    finally:
        subprocess.run(["docker", "rm", "--force", container_id], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
