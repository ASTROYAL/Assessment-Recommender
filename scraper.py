from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, Tag


SHL_ROOT = "https://www.shl.com/"
REQUESTED_CATALOG_URL = "https://www.shl.com/solutions/products/productcatalog/"
CURRENT_CATALOG_URL = "https://www.shl.com/products/product-catalog/"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "catalog.json"
PAGE_SIZE = 12
REQUEST_DELAY_SECONDS = 1
VALID_TEST_TYPES = ("A", "P", "B", "C", "K", "S")
TEST_TYPE_PRIORITY = ("A", "P", "B", "C", "K", "S")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in (clean_text(part) for part in value.split(",")) if item]


def fetch(client: httpx.Client, url: str, *, params: dict[str, str | int] | None = None) -> str:
    try:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.text
    finally:
        time.sleep(REQUEST_DELAY_SECONDS)


def find_individual_table(soup: BeautifulSoup) -> Tag | None:
    for table in soup.find_all("table"):
        header_text = clean_text(" ".join(th.get_text(" ", strip=True) for th in table.find_all("th")))
        if "Individual Test Solutions" in header_text:
            return table
    return None


def extract_test_type_from_container(container: Tag | None) -> str | None:
    if container is None:
        return None

    codes: list[str] = []
    for key in container.select(".product-catalogue__key"):
        code = clean_text(key.get_text(" ", strip=True)).upper()
        if code in VALID_TEST_TYPES:
            codes.append(code)

    if not codes:
        raw_text = clean_text(container.get_text(" ", strip=True)).upper()
        codes = [code for code in re.findall(r"\b[A-Z]\b", raw_text) if code in VALID_TEST_TYPES]

    for preferred in TEST_TYPE_PRIORITY:
        if preferred in codes:
            return preferred
    return None


def infer_test_type_from_text(*values: str | Iterable[str] | None) -> str | None:
    text_parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text_parts.append(value)
        else:
            text_parts.extend(str(item) for item in value)
    text = " ".join(text_parts).lower()

    keyword_map: list[tuple[str, str]] = [
        ("simulation", "S"),
        ("coding", "S"),
        ("exercise", "S"),
        ("knowledge", "K"),
        ("skills", "K"),
        ("skill", "K"),
        ("personality", "P"),
        ("behavior", "P"),
        ("behaviour", "P"),
        ("biodata", "B"),
        ("situational judgement", "B"),
        ("situational judgment", "B"),
        ("competenc", "C"),
        ("ability", "A"),
        ("aptitude", "A"),
        ("reasoning", "A"),
    ]
    for keyword, code in keyword_map:
        if keyword in text:
            return code
    return None


def parse_listing_items(html: str) -> list[dict[str, str | None]]:
    soup = BeautifulSoup(html, "html.parser")
    table = find_individual_table(soup)
    if table is None:
        return []

    items: list[dict[str, str | None]] = []
    for row in table.find_all("tr"):
        link = row.find("a", href=re.compile(r"/products/product-catalog/view/"))
        if not link:
            continue

        href = link.get("href")
        name = clean_text(link.get_text(" ", strip=True))
        if not href or not name:
            continue

        cells = row.find_all("td")
        test_type = extract_test_type_from_container(cells[-1]) if cells else None
        absolute_url = urljoin(SHL_ROOT, href)
        items.append({"name": name, "url": absolute_url, "test_type": test_type})

    return items


def section_text(soup: BeautifulSoup, heading: str) -> str:
    target = heading.lower()
    for h4 in soup.find_all("h4"):
        if clean_text(h4.get_text(" ", strip=True)).lower() != target:
            continue

        row = h4.find_parent(class_="product-catalogue-training-calendar__row") or h4.parent
        if row is None:
            return ""

        paragraphs: list[str] = []
        for paragraph in row.find_all("p", recursive=True):
            paragraph_text = clean_text(paragraph.get_text(" ", strip=True))
            if not paragraph_text:
                continue
            if paragraph_text.startswith("Test Type:") or paragraph_text.startswith("Remote Testing:"):
                continue
            paragraphs.append(paragraph_text)
        return clean_text(" ".join(paragraphs))

    return ""


def extract_duration(soup: BeautifulSoup) -> str | None:
    text = section_text(soup, "Assessment length")
    return text or None


def extract_detail_test_type(soup: BeautifulSoup) -> str | None:
    for node in soup.find_all(string=re.compile(r"Test Type:", re.IGNORECASE)):
        parent = node.find_parent("p")
        test_type = extract_test_type_from_container(parent)
        if test_type:
            return test_type
    return None


def parse_detail_page(html: str, *, fallback_name: str, url: str, listing_test_type: str | None) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    name = clean_text(h1.get_text(" ", strip=True)) if h1 else fallback_name
    description = section_text(soup, "Description")
    job_levels = split_csv(section_text(soup, "Job levels"))
    languages = split_csv(section_text(soup, "Languages"))
    duration = extract_duration(soup)
    test_type = (
        extract_detail_test_type(soup)
        or listing_test_type
        or infer_test_type_from_text(name, description, job_levels)
        or "K"
    )

    return {
        "name": name,
        "url": url,
        "description": description,
        "test_type": test_type,
        "job_levels": job_levels,
        "languages": languages,
        "duration": duration,
    }


def discover_catalog_url(client: httpx.Client) -> tuple[str, str]:
    candidate_urls = (REQUESTED_CATALOG_URL, CURRENT_CATALOG_URL)
    last_error: Exception | None = None

    for candidate in candidate_urls:
        try:
            print(f"Checking catalog URL: {candidate}")
            html = fetch(client, candidate, params={"start": 0, "type": 1})
        except Exception as exc:
            last_error = exc
            print(f"Catalog URL unavailable: {candidate} ({exc})")
            continue

        if parse_listing_items(html):
            print(f"Using catalog URL: {candidate}")
            return candidate, html
        print(f"No individual test rows found at: {candidate}")

    message = "Unable to locate SHL Individual Test Solutions catalog."
    if last_error:
        message = f"{message} Last error: {last_error}"
    raise RuntimeError(message)


def scrape_catalog() -> list[dict]:
    print("Starting SHL Individual Test Solutions scrape...")
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30.0) as client:
        catalog_url, first_page_html = discover_catalog_url(client)
        assessments: list[dict] = []
        seen_urls: set[str] = set()
        start = 0
        current_html: str | None = first_page_html

        while True:
            if current_html is None:
                print(f"Fetching catalog page start={start}")
                current_html = fetch(client, catalog_url, params={"start": start, "type": 1})
            else:
                print(f"Parsing catalog page start={start}")

            listing_items = parse_listing_items(current_html)
            new_items = [item for item in listing_items if item["url"] not in seen_urls]
            if not new_items:
                print("No new Individual Test Solutions found; pagination complete.")
                break

            print(f"Found {len(new_items)} individual tests on page start={start}.")
            for item in new_items:
                seen_urls.add(str(item["url"]))
                print(f"Scraping detail page {len(assessments) + 1}: {item['name']}")
                try:
                    detail_html = fetch(client, str(item["url"]))
                    assessment = parse_detail_page(
                        detail_html,
                        fallback_name=str(item["name"]),
                        url=str(item["url"]),
                        listing_test_type=item["test_type"] if isinstance(item["test_type"], str) else None,
                    )
                    assessments.append(assessment)
                except Exception as exc:
                    print(f"Failed to scrape {item['url']}: {exc}")

            start += PAGE_SIZE
            current_html = None

    if not assessments:
        raise RuntimeError("No SHL Individual Test Solutions were scraped.")

    return assessments


def main() -> None:
    catalog = scrape_catalog()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(catalog)} assessments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
