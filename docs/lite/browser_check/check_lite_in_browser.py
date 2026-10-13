#!/usr/bin/env python
"""Check that the JupyterLite site starts and installs the toolbox in a real browser.

Opens a notebook in a fresh browser context, waits for the Pyodide kernel, runs the
notebook's bootstrap cell (the first code cell) and requires its status line
``Running in browser using MVTB v...`` to appear.  Also fails if JupyterLab itself
could not start (``Plugin '...' failed to activate`` console errors), which happens
before any kernel exists and which the Node/Pyodide install check cannot see.

Usage::

    python check_lite_in_browser.py --base-url http://localhost:8000/lite
    python check_lite_in_browser.py --base-url https://petercorke.github.io/machinevision-toolbox-python/lite

Exits 0 if every requested browser passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

SUCCESS_TEXT = "Running in browser using MVTB"
APP_FAILURE_MARKERS = ("failed to activate", "No setting registry")


class CheckFailed(Exception):
    """The site did not behave as expected; the message says how."""


def check_once(
    playwright,
    browser_name: str,
    base_url: str,
    notebook: str,
    timeout_s: float,
    screenshot: Path | None,
) -> str:
    """Run the check once in one browser, returning the success status line."""
    browser = getattr(playwright, browser_name).launch()
    try:
        # a fresh context has no storage, cache or service workers
        page = browser.new_context().new_page()

        console_errors: list[str] = []
        http_errors: list[str] = []
        page.on(
            "console",
            lambda m: console_errors.append(m.text) if m.type == "error" else None,
        )
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
        page.on(
            "response",
            lambda r: http_errors.append(f"{r.status} {r.url}") if r.status >= 400 else None,
        )

        def fail(reason: str) -> CheckFailed:
            if screenshot is not None:
                try:
                    page.screenshot(path=str(screenshot), full_page=True)
                except PlaywrightError:
                    pass
            lines = [reason]
            if console_errors:
                lines.append("console errors (first 10):")
                lines += [f"  {t[:200]}" for t in console_errors[:10]]
            if http_errors:
                lines.append("HTTP errors (first 10):")
                lines += [f"  {t}" for t in http_errors[:10]]
            return CheckFailed("\n".join(lines))

        url = f"{base_url.rstrip('/')}/lab/index.html?path={notebook}"
        page.goto(url, wait_until="load", timeout=timeout_s * 1000)

        try:
            page.wait_for_selector(".jp-Notebook .jp-CodeCell", timeout=60_000)
        except PlaywrightTimeout:
            raise fail(f"JupyterLab did not show the notebook: {url}") from None

        app_failures = [
            t for t in console_errors if any(m in t for m in APP_FAILURE_MARKERS)
        ]
        if app_failures:
            raise fail("JupyterLab plugins failed to activate")

        # Run before the kernel is idle and the request is silently lost, leaving the
        # cell at "[ ]:" forever, so wait for it explicitly.
        try:
            page.get_by_text("Python (Pyodide) | Idle").wait_for(timeout=timeout_s * 1000)
        except PlaywrightTimeout:
            raise fail("the Pyodide kernel never became idle") from None

        # the first code cell is the generated bootstrap cell
        cell = page.locator(".jp-Notebook .jp-CodeCell").first
        cell.locator(".jp-InputPrompt").click()
        page.locator(
            '[data-command="notebook:run-cell-and-select-next"]'
        ).first.click()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            outputs = " ".join(cell.locator(".jp-OutputArea").all_inner_texts())
            if SUCCESS_TEXT in outputs:
                for line in outputs.splitlines():
                    if SUCCESS_TEXT in line:
                        return line.strip()
                return SUCCESS_TEXT
            if "Traceback" in outputs:
                raise fail("the bootstrap cell raised:\n" + outputs[-1500:])
            time.sleep(2)

        outputs = " ".join(cell.locator(".jp-OutputArea").all_inner_texts())
        raise fail(
            f"no '{SUCCESS_TEXT}' within {timeout_s:.0f} s; output so far:\n"
            + outputs[-800:]
        )
    finally:
        browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", required=True, help="URL of the lite site root")
    parser.add_argument("--notebook", default="intro.ipynb")
    parser.add_argument(
        "--browsers", nargs="+", default=["chromium", "webkit"], help="Playwright browsers"
    )
    parser.add_argument("--timeout", type=float, default=300, help="seconds per attempt")
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="extra attempts after a failure (deploy propagation, network flakiness)",
    )
    parser.add_argument("--retry-wait", type=float, default=60, help="seconds between attempts")
    parser.add_argument("--screenshot-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.screenshot_dir is not None:
        args.screenshot_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    with sync_playwright() as p:
        for name in args.browsers:
            shot = args.screenshot_dir / f"{name}.png" if args.screenshot_dir else None
            for attempt in range(args.retries + 1):
                try:
                    status = check_once(
                        p, name, args.base_url, args.notebook, args.timeout, shot
                    )
                except (CheckFailed, PlaywrightError) as e:
                    print(f"[{name}] attempt {attempt + 1} FAILED:\n{e}\n", flush=True)
                    if attempt < args.retries:
                        time.sleep(args.retry_wait)
                        continue
                    ok = False
                else:
                    print(f"[{name}] PASS: {status}", flush=True)
                break

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
