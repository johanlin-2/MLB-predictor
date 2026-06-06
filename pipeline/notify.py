"""Analyse today's picks with Claude and send a summary via iMessage.

Called automatically at the end of pipeline.predict.run() when
ANTHROPIC_API_KEY and NOTIFY_PHONE are set in .env.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _send_imessage(phone: str, message: str) -> None:
    script = f"""
    tell application "Messages"
        set targetService to 1st service whose service type = iMessage
        set targetBuddy to buddy "{phone}" of targetService
        send "{message}" to targetBuddy
    end tell
    """
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"AppleScript error: {result.stderr.strip()}")


def _build_prompt(picks: pd.DataFrame, date_str: str) -> str:
    lines = [f"MLB picks for {date_str}. {len(picks)} games today.\n"]

    flagged = picks[picks.get("flag_bet", False) == True]  # noqa: E712
    if not flagged.empty:
        lines.append("FLAGGED BETS (edge threshold passed):")
        for _, r in flagged.iterrows():
            lines.append(
                f"  {r['visiting_team']} @ {r['home_team']} | "
                f"model win prob: {r['model_win_prob']:.1%} | "
                f"best book: {r.get('best_book','?')} {r.get('best_price','?')} | "
                f"edge: {r.get('edge_vs_best', 0):.1%} | "
                f"stake: ${r.get('recommended_stake_usd', 0):.0f}"
            )
    else:
        lines.append("No bets passed the flagging threshold today.")

    lines.append("\nAll games summary:")
    for _, r in picks.iterrows():
        lines.append(
            f"  {r['visiting_team']} @ {r['home_team']}: "
            f"model={r['model_win_prob']:.1%} home win, "
            f"predicted {r.get('predicted_away_runs', '?'):.1f}-{r.get('predicted_home_runs', '?'):.1f}, "
            f"edge={r.get('edge_vs_best', 0):.1%}"
        )

    lines.append(
        "\nWrite a full daily MLB betting breakdown covering every game. "
        "Structure it as:\n"
        "1. TOP PICK (or 'NO EDGE TODAY') — one sentence on the best opportunity.\n"
        "2. FULL SLATE — for each game: matchup, model win prob, predicted score, "
        "best line, edge %, and a one-line take on whether it's worth a look.\n"
        "3. SUMMARY — 1-2 sentences on overall slate quality.\n"
        "Be direct and analytical. No filler. No emojis."
    )
    return "\n".join(lines)


def send_daily_summary(picks_path: Path) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    phone = os.getenv("NOTIFY_PHONE")

    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set — skipping notify")
        return
    if not phone:
        logger.info("NOTIFY_PHONE not set — skipping notify")
        return

    import anthropic  # lazy import — only needed when keys are present

    picks = pd.read_csv(picks_path)
    if picks.empty:
        logger.info("picks file empty — skipping notify")
        return

    date_str = picks_path.stem.replace("picks_", "")
    prompt = _build_prompt(picks, date_str)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system="You are a sharp sports-betting analyst. Be concise and direct.",
        messages=[{"role": "user", "content": prompt}],
    )
    summary = response.content[0].text.strip()
    logger.info("Claude summary: %s", summary)

    try:
        _send_imessage(phone, summary)
        logger.info("iMessage sent to %s", phone)
    except RuntimeError as exc:
        logger.error("iMessage failed: %s", exc)
