"""Focused browser checks for the rendered noisy-regression writeup.

Render the page first. On the server, run with temporary dependencies:
uv run --no-project --with pytest --with playwright python -m pytest \
    tests/test_noisy_regression_writeup.py --basetemp=/tmp/noisy-writeup-checks
"""

import functools
import http.server
import json
import math
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="module")
def browser_page():
    docs = Path(__file__).resolve().parents[1] / "docs"
    assert (docs / "noisy-regression.html").is_file(), (
        "Render the Quarto page before running these checks"
    )
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(docs)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(
            viewport={"width": 1440, "height": 1050}, device_scale_factor=1
        )
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"http://127.0.0.1:{server.server_port}/noisy-regression.html")
        page.wait_for_selector("#nr-input-chart svg")
        yield page, errors
        browser.close()
    server.shutdown()
    server.server_close()
    thread.join()


def set_range(page, selector, value):
    page.locator(selector).evaluate(
        "(element, value) => { element.value = value; element.dispatchEvent(new Event('input', {bubbles: true})); }",
        str(value),
    )


def expected_digits(value):
    midpoints = [-3 + (i + 0.5) * 6 / 255 for i in range(255)]
    index = sum(value >= boundary for boundary in midpoints)
    return list(f"{index:02X}")


def test_complete_prompts_match_continuous_example(browser_page):
    page, _ = browser_page
    example = json.loads(page.locator("#nr-example-data").text_content())
    assert len(example["rows"]) == 64
    for row in [*example["rows"], example["query"]]:
        assert row["signal"] == pytest.approx(
            sum(w * x for w, x in zip(example["w"], row["x"], strict=True))
        )
        assert row["y"] == pytest.approx(row["signal"] + row["noise"])
    for version, prompt_length, answer_length in [("new", 649, 3), ("legacy", 519, 2)]:
        page.select_option("#nr-prompt-format", version)
        tokens = page.locator("#nr-prompt-text").text_content().split()
        answer = page.locator("#nr-answer-text").text_content().split()
        assert len(tokens) == prompt_length
        assert len(answer) == answer_length
        assert answer[:2] == expected_digits(example["query"]["y"])
        assert [token for token in tokens if not token.startswith("[")] == [
            digit
            for row in example["rows"]
            for scalar in [*row["x"], row["y"]]
            for digit in expected_digits(scalar)
        ] + [
            digit
            for scalar in example["query"]["x"]
            for digit in expected_digits(scalar)
        ]
        assert tokens.count("[X]") == tokens.count("[Y]") == 65
        assert tokens.count("[EOO]") == (64 if version == "new" else 0)
        assert tokens.count("[SEP]") == (65 if version == "new" else 0)
    page.select_option("#nr-prompt-format", "new")


def test_quantization_center_boundary_and_probability_conservation(browser_page):
    page, _ = browser_page
    widget = page.locator("#nr-quantization")
    page.click("#nr-q-default")
    assert float(widget.get_attribute("data-same-bin-probability")) > 1 - 1e-10
    page.click("#nr-q-boundary")
    assert float(widget.get_attribute("data-same-bin-probability")) == pytest.approx(
        0.5, abs=1e-7
    )
    for levels, range_, log_sigma in [(16, 1, 0), (256, 3, -2), (1024, 8, -4)]:
        page.select_option("#nr-levels", str(levels))
        page.select_option("#nr-range", str(range_))
        set_range(page, "#nr-q-sigma", log_sigma)
        assert float(widget.get_attribute("data-probability-sum")) == pytest.approx(
            1, abs=1e-8
        )
    page.click("#nr-q-default")


def test_geometry_information_and_population_controls(browser_page):
    page, _ = browser_page
    widget = page.locator("#nr-geometry")
    page.click("#nr-g-default")
    set_range(page, "#nr-n", 1)
    one_observation_trace = float(widget.get_attribute("data-posterior-trace"))
    set_range(page, "#nr-n", 64)
    many_observations_trace = float(widget.get_attribute("data-posterior-trace"))
    assert 0 < many_observations_trace < one_observation_trace
    set_range(page, "#nr-g-sigma", -1)
    assert float(widget.get_attribute("data-posterior-trace")) > many_observations_trace
    page.check("#nr-show-quantized")
    assert (
        "Decoded-input ridge query prediction"
        in page.locator("#nr-g-detail").text_content()
    )
    before = page.locator("#nr-g-readout").text_content()
    page.click("#nr-new-task")
    assert page.locator("#nr-g-readout").text_content() != before
    page.select_option("#nr-dimension", "64")
    assert "0.09375" in page.locator("#nr-pop-readout").text_content()
    page.select_option("#nr-pop-mode", "conditional")
    assert page.locator("#nr-dimension").is_disabled()
    assert "exactly Gaussian" in page.locator("#nr-pop-readout").text_content()
    page.select_option("#nr-pop-mode", "population")
    page.select_option("#nr-dimension", "2")
    page.click("#nr-g-default")


def test_rendered_layout_links_and_screenshots(browser_page, tmp_path):
    page, errors = browser_page
    assert page.locator(".quarto-unresolved-ref").count() == 0
    assert page.locator("#nr-example-rows tr").count() == 65
    for selector in [
        "#nr-population",
        "#nr-quantization",
        "#nr-geometry",
        "#nr-prompts",
    ]:
        page.locator(selector).screenshot(
            path=str(tmp_path / f"desktop-{selector[1:]}.png")
        )
    page.screenshot(path=str(tmp_path / "desktop-page.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate(
        "document.documentElement.scrollWidth <= window.innerWidth + 1"
    )
    for selector in [
        "#nr-population",
        "#nr-quantization",
        "#nr-geometry",
        "#nr-prompts",
    ]:
        page.locator(selector).screenshot(
            path=str(tmp_path / f"mobile-{selector[1:]}.png")
        )
    page.locator("#nr-prompts details summary").first.click()
    assert page.locator("#nr-prompt-text").is_visible()
    assert page.evaluate(
        "document.documentElement.scrollWidth <= window.innerWidth + 1"
    )
    invalid_geometry = page.locator(".nr-plot svg").evaluate_all(
        "elements => elements.some(element => /(?:NaN|Infinity)/.test(element.innerHTML))"
    )
    assert not invalid_geometry
    assert not errors
    page.set_viewport_size({"width": 1440, "height": 1050})


def test_static_numerical_example():
    assert expected_digits(1) == ["A", "A"]
    assert expected_digits(0) == ["8", "0"]
    assert [expected_digits(y) for y in [0.601, -0.801, -1.3995, 0.701]] == [
        ["9", "9"],
        ["5", "D"],
        ["4", "4"],
        ["9", "D"],
    ]
    assert math.exp(-3 * math.sqrt(2)) == pytest.approx(0.01436959609)
