"""Polymarket + Kalshi MLB markets (live picks only — never used in backtest)."""
from __future__ import annotations

import base64
import datetime
import logging
import os
from typing import Any

import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polymarket  (unchanged — was already correct)
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _polymarket_get(path: str, params: dict[str, Any] | None = None) -> Any:
    resp = requests.get(f"{config.POLYMARKET_BASE}{path}", params=params or {}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_polymarket() -> pd.DataFrame:
    """Pull active MLB markets from the Polymarket CLOB API.

    Prices come back on the 0-1 scale, already no-vig (LMSR-style market).
    """
    snapshot_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    try:
        data = _polymarket_get("/markets", {"active": "true", "closed": "false"})
    except requests.RequestException as exc:
        logger.warning("polymarket fetch failed: %s", exc)
        return pd.DataFrame()
    for market in (data or {}).get("data", []):
        raw_tags = market.get("tags") or []
        # API returns tags as either strings or dicts with a "label" key.
        tags = [
            (t.get("label", "") if isinstance(t, dict) else str(t)).lower()
            for t in raw_tags
        ]
        if not any("mlb" in t or "baseball" in t for t in tags):
            continue
        for token in market.get("tokens", []):
            rows.append({
                "snapshot_ts": snapshot_ts,
                "market_id": market.get("condition_id"),
                "question": market.get("question"),
                "outcome": token.get("outcome"),
                "price": float(token.get("price") or 0.0),  # already 0-1
                "end_date": market.get("end_date_iso"),
            })
    df = pd.DataFrame(rows)
    out = config.RAW_DIR / "polymarket.parquet"
    df.to_parquet(out, index=False)
    logger.info("wrote %d polymarket rows to %s", len(df), out)
    return df


# ---------------------------------------------------------------------------
# Kalshi — RSA-signed requests (v2 API, current as of 2025)
# ---------------------------------------------------------------------------

def _load_kalshi_private_key():
    """Load RSA private key from the path specified in config/env."""
    key_path = getattr(config, "KALSHI_PRIVATE_KEY_PATH", None) or os.getenv(
        "KALSHI_PRIVATE_KEY_PATH"
    )
    if not key_path:
        raise RuntimeError(
            "KALSHI_PRIVATE_KEY_PATH not set. "
            "Point it to the .key file downloaded when you created your API key at "
            "https://kalshi.com/account/profile"
        )
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _kalshi_headers(method: str, path: str) -> dict[str, str]:
    """Build the three signed headers required by every Kalshi v2 request.

    Kalshi requires:
        KALSHI-ACCESS-KEY       — your Key ID (UUID from the dashboard)
        KALSHI-ACCESS-TIMESTAMP — current time in milliseconds
        KALSHI-ACCESS-SIGNATURE — RSA-PSS signature of (timestamp + METHOD + path)

    The path must have query parameters stripped before signing.
    """
    key_id = getattr(config, "KALSHI_KEY_ID", None) or os.getenv("KALSHI_KEY_ID")
    if not key_id:
        raise RuntimeError(
            "KALSHI_KEY_ID not set. Copy the Key ID UUID from "
            "https://kalshi.com/account/profile after creating an API key."
        )

    timestamp_ms = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000))

    # Strip query string before signing
    path_no_query = path.split("?")[0]
    msg = (timestamp_ms + method.upper() + path_no_query).encode("utf-8")

    private_key = _load_kalshi_private_key()
    signature = private_key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )

    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        "Content-Type": "application/json",
    }


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _kalshi_get(path: str, params: dict[str, Any] | None = None) -> Any:
    """Signed GET against the Kalshi v2 REST API."""
    # Build full path string (with query params) for the URL, but sign without them
    query = ""
    if params:
        query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    full_path = path + query

    headers = _kalshi_headers("GET", path)  # sign path without query
    resp = requests.get(
        f"{config.KALSHI_BASE}{full_path}",
        headers=headers,
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_kalshi() -> pd.DataFrame:
    """Pull active MLB markets from Kalshi.

    yes_ask is used as the implied probability (already on 0-1 scale after
    converting from cents).
    """
    snapshot_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    try:
        events = _kalshi_get("/events", params={"series_ticker": "KXMLB", "status": "open"})
        for event in events.get("events", []):
            for market in event.get("markets", []):
                rows.append({
                    "snapshot_ts": snapshot_ts,
                    "ticker": market.get("ticker"),
                    "event_ticker": event.get("event_ticker"),
                    "title": market.get("title"),
                    "yes_ask": (market.get("yes_ask") or 0) / 100.0,   # cents → prob
                    "yes_bid": (market.get("yes_bid") or 0) / 100.0,
                    "close_ts": market.get("close_time"),
                })
    except Exception as exc:  # noqa: BLE001
        logger.warning("kalshi fetch failed: %s", exc)
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = config.RAW_DIR / "kalshi.parquet"
    df.to_parquet(out, index=False)
    logger.info("wrote %d kalshi rows to %s", len(df), out)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_polymarket()
    fetch_kalshi()