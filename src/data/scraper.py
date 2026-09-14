import os
import warnings
import re
from typing import List
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class WebsiteScraper:
    """Handles web scraping, noise removal, and text chunking for the RAG pipeline."""

    # Tag names that are almost always non-content (semantic tags)
    NOISE_TAGS = ["script", "style", "footer", "nav", "header", "noscript",
                  "iframe", "form", "button", "aside", "svg"]

    # class/id name fragments commonly used for menus, sidebars, ads, etc.
    # (matched case-insensitively as substrings)
    NOISE_SELECTORS = [
        "nav", "menu", "navbar", "sidebar", "widget", "breadcrumb",
        "footer", "header", "topbar", "advert", "ads", "banner",
        "social", "share", "cookie", "popup", "modal", "search-box",
        "pagination", "related-post", "comment"
    ]

    CONTENT_SELECTORS = [
        {"name": "main"},
        {"name": "article"},
        {"attrs": {"id": "content"}},
        {"attrs": {"id": "main-content"}},
        {"attrs": {"class": "entry-content"}},
        {"attrs": {"class": "post-content"}},
        {"attrs": {"class": "content-area"}},
    ]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def get_website_data(self, url: str) -> List[Document]:
        """Main execution pipeline to fetch, clean, and chunk website data."""
        print(f"Fetching data from: {url} ...")
        try:
            clean_text = self._extract_and_clean_text(url)
            if not clean_text:
                print("Error: Website se koi data nahi mila.")
                return []

            return self._create_chunks(clean_text, url)

        except Exception as e:
            print(f"Scraping Error: {e}")
            return []

    def _remove_noise_tags(self, soup: BeautifulSoup) -> None:
        """Removes semantic noise tags (nav, footer, script, etc.)."""
        for tag in soup.find_all(self.NOISE_TAGS):
            tag.decompose()

    # Max text length (chars) a "noise" element is allowed to have.
    
    MAX_NOISE_TEXT_LEN = 400

    def _remove_noise_by_class_id(self, soup: BeautifulSoup) -> None:
        """Removes elements whose class/id matches known noise patterns
        (menus, sidebars, ads, etc. that aren't wrapped in semantic tags).
        Uses word-boundary matching (not raw substring) so e.g. 'nav'
        doesn't accidentally match 'navigation-wrapper' or 'main-nav-section',
        and skips large tags to avoid deleting real content by accident."""
        pattern = re.compile(
            r'(?:^|[\s_-])(?:' + "|".join(self.NOISE_SELECTORS) + r')(?:[\s_-]|$)',
            re.IGNORECASE
        )

        def is_noise(tag) -> bool:
            class_val = " ".join(tag.get("class", []))
            id_val = tag.get("id", "") or ""
            combined = f"{class_val} {id_val}".strip()
            if not combined:
                return False
            return bool(pattern.search(combined))

        for tag in soup.find_all(is_noise):
            if len(tag.get_text(strip=True)) > self.MAX_NOISE_TEXT_LEN:
                continue 
            tag.decompose()

    def _find_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Tries to isolate the main content container. Falls back to the
        full (already noise-stripped) soup if nothing matches."""
        for selector in self.CONTENT_SELECTORS:
            name = selector.get("name")
            attrs = selector.get("attrs", {})
            found = soup.find(name=name, attrs=attrs) if name or attrs else None
            if found and len(found.get_text(strip=True)) > 200:
                return found
        return soup

    def _extract_and_clean_text(self, url: str) -> str:
        """Fetches HTML, removes unwanted tags/sections, and strips
        irregular whitespace, returning only the meaningful page text."""
        loader = WebBaseLoader(url)
        soup = loader.scrape()

        # Step 1: remove obvious semantic noise tags
        self._remove_noise_tags(soup)

        # Step 2: remove non-semantic noise (menus/sidebars via class/id)
        self._remove_noise_by_class_id(soup)

        # Step 3: try to isolate the main content block
        content_node = self._find_main_content(soup)

        raw_text = content_node.get_text(separator='\n', strip=True)

       
        if len(raw_text) < 200:
            print("Warning: Cleaning removed too much text, falling back to raw page text.")
            fallback_loader = WebBaseLoader(url)
            fallback_soup = fallback_loader.scrape()
            for tag in fallback_soup(["script", "style"]):
                tag.decompose()
            raw_text = fallback_soup.get_text(separator='\n', strip=True)

        # Step 4: whitespace cleanup
        clean_text = re.sub(r'\n+', '\n', raw_text)
        clean_text = re.sub(r'[ \t]+', ' ', clean_text).strip()

        return clean_text

    def _create_chunks(self, clean_text: str, url: str) -> List[Document]:
        """Wraps text in a Document object and splits it into optimal chunks."""
        doc = Document(page_content=clean_text, metadata={"source": url})
        chunks = self.text_splitter.split_documents([doc])

        print(f"Success! Total chunks created: {len(chunks)}")
        return chunks