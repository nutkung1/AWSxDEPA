from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import csv
import urllib.parse


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


def crawl_website(base_url):
    driver = setup_driver()
    visited = set()
    to_visit = [base_url]
    data = []

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
            title = soup.title.string if soup.title else "No title"
            main_content = " ".join(
                [
                    p.text
                    for p in soup.find_all(
                        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
                    )
                ]
            )

            data.append(
                {
                    "URL": url,
                    "Title": title,
                    "Content": main_content,  # No longer limiting to 500 characters
                }
            )

            # Find new links
            new_links = extract_links(soup, base_url)
            to_visit.extend([link for link in new_links if link not in visited])

        except Exception as e:
            print(f"Error crawling {url}: {e}")

    driver.quit()
    return data


def save_to_csv(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["URL", "Title", "Content"])
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    base_url = "https://www.jventures.co.th/"
    crawled_data = crawl_website(base_url)
    save_to_csv(crawled_data, "Data/jventures_crawl_results.csv")
    print(f"Crawling completed. Data saved to jventures_crawl_results.csv")
