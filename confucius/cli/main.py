# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import asyncio
import signal

import click

from confucius.analects.code.entry import CodeAssistEntry  # noqa: F401
from confucius.analects.osworld.entry import OSWorldEntry  # noqa: F401

from confucius.lib.confucius import Confucius
from confucius.lib.entry_repl import run_entry_repl


async def _run_repl(entry_name: str, *, verbose: bool) -> None:
    """
    Start a minimal REPL that routes user input to the specified entry.

    Similar spirit to confucius.cli.common_cli.run_lite but simplified for the
    GitHub library variant and without any metagen handling.
    """
    cf: Confucius = Confucius(verbose=verbose)

    # run_entry_repl resolves to None upon completion
    task: asyncio.Task[None] = asyncio.create_task(
        run_entry_repl(cf, entry_name=entry_name)
    )

    async def on_interrupt(_cf: Confucius, _task: asyncio.Task[None]) -> None:
        # If we successfully canceled a running analect, then no need to exit the program.
        # We only exit if that is not the case (fall back to cancel current task).
        if not await _cf.cancel_task():
            _task.cancel()

    loop = asyncio.get_event_loop()
    try:
        loop.add_signal_handler(
            signal.SIGINT, lambda: asyncio.create_task(on_interrupt(cf, task))
        )
    except NotImplementedError:
        # Some environments may not support signal handlers (e.g. Windows, certain REPLs)
        pass

    await task


@click.group()
def main() -> None:  # noqa: D401
    """Confucius GitHub CLI"""
    pass


@main.command("code")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logs")
def code_cmd(verbose: bool) -> None:
    """
    Launch a REPL for CodeAssistEntry.

    Usage: <CLI_NAME> code
    Then type your input line-by-line to interact.
    """
    asyncio.run(_run_repl("Code", verbose=verbose))


@main.command("osworld")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logs")
def osworld_cmd(verbose: bool) -> None:
    """
    Launch a REPL for OSWorldEntry.

    Usage: <CLI_NAME> osworld
    Then type your input line-by-line to interact.
    """
    asyncio.run(_run_repl("OSWorld", verbose=verbose))


if __name__ == "__main__":
    # Allow running via `python -m confucius.cli.main` or directly as a script
    main()
