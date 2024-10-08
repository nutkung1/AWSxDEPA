from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import json
import urllib.parse
import re


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=chrome_options
    )


def get_page_content(driver, url):
    driver.get(url)
    time.sleep(2)  # Wait for JavaScript to load
    return driver.page_source


def extract_links(soup, base_url):
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urllib.parse.urljoin(base_url, href)
        if full_url.startswith(base_url) and not full_url.endswith(
            (".pdf", ".jpg", ".png")
        ):
            links.append(full_url)
    return list(set(links))  # Remove duplicates


def clean_text(text):
    # Remove non-printable and non-ASCII characters
    text = re.sub(r"[^\x20-\x7E\s]", "", text)
    # Replace multiple whitespace with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def crawl_website(base_url, output_file):
    driver = setup_driver()
    visited = set()
    to_visit = [base_url]

    with open(output_file, "w", encoding="utf-8") as file:
        while to_visit:
            url = to_visit.pop(0)
            if url in visited:
                continue

            print(f"Visiting: {url}")
            visited.add(url)

            try:
                content = get_page_content(driver, url)
                soup = BeautifulSoup(content, "html.parser")

                # Extract title and main content
                title = clean_text(soup.title.string if soup.title else "No title")
                main_content = clean_text(
                    " ".join(
                        [
                            p.text
                            for p in soup.find_all(
                                ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
                            )
                        ]
                    )
                )

                # Create a JSON object
                data = {
                    "URL": url,
                    "Title": title,
                    "Content": main_content,
                }

                # Write the JSON object as a single line in the JSONL file
                json.dump(data, file, ensure_ascii=False)
                file.write("\n")

                # Find new links
                new_links = extract_links(soup, base_url)
                to_visit.extend([link for link in new_links if link not in visited])

            except Exception as e:
                print(f"Error crawling {url}: {e}")

    driver.quit()


if __name__ == "__main__":
    base_url = "https://www.jventures.co.th/"
    output_file = "Data/jventures_crawl_results.jsonl"
    crawl_website(base_url, output_file)
    print(f"Crawling completed. Data saved to {output_file}")
