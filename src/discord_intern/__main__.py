from __future__ import annotations

import argparse
import asyncio
import logging

from discord_intern.adapters.discord import DiscordBotAdapter
from discord_intern.ai import MockAIClient
from discord_intern.config import YamlConfigLoader
from discord_intern.config.models import ConfigLoadRequest
from discord_intern.logging import init_logging

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="discord-intern", description="Discord Intern bot runner")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Disable loading .env (env overrides still apply)",
    )
    parser.add_argument(
        "--mock-reply-text",
        default=None,
        help="Override the default mock reply text",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Run the bot for N seconds then exit (useful for smoke testing).",
    )
    return parser


async def _stop_adapter_gracefully(adapter: DiscordBotAdapter, *, timeout_seconds: float = 15.0) -> None:
    try:
        await asyncio.wait_for(asyncio.shield(adapter.stop()), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("app.shutdown_timeout timeout_seconds=%s", timeout_seconds)
    except Exception:
        logger.exception("app.shutdown_error")


async def _run_async(args: argparse.Namespace) -> None:
    loader = YamlConfigLoader()
    request = ConfigLoadRequest(
        yaml_path=args.config,
        dotenv_path=None if args.no_dotenv else ".env",
    )
    config = await loader.load(request)

    init_logging(config.logging)
    logger.info("app.starting dry_run=%s", config.app.dry_run)

    ai_client = MockAIClient(reply_text=args.mock_reply_text) if args.mock_reply_text else MockAIClient()
    adapter = DiscordBotAdapter(config=config, ai_client=ai_client)
    try:
        if args.run_seconds is not None:
            await adapter.run_for(seconds=args.run_seconds)
        else:
            await adapter.start()
    finally:
        await _stop_adapter_gracefully(adapter)


def main() -> None:
    args = _build_parser().parse_args()
    try:
        asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        logger.info("app.interrupted_by_user")


if __name__ == "__main__":
    main()
