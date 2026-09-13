import csv
from pathlib import Path

import feedparser


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "recent_real_news.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# REAL, REPUTABLE NEWS RSS FEEDS (free, no API key)
# ==================================================
#
# Using many feeds across multiple outlets and
# categories, since each feed only exposes its latest
# ~20-50 items. More feeds = more articles collected
# per run, and more topic/style diversity.

RSS_FEEDS = [
    # BBC
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://feeds.bbci.co.uk/news/uk/rss.xml",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "http://feeds.bbci.co.uk/news/health/rss.xml",
    "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
    "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "http://feeds.bbci.co.uk/sport/rss.xml",

    # NPR
    "https://feeds.npr.org/1001/rss.xml",
    "https://feeds.npr.org/1004/rss.xml",
    "https://feeds.npr.org/1006/rss.xml",
    "https://feeds.npr.org/1007/rss.xml",
    "https://feeds.npr.org/1019/rss.xml",
    "https://feeds.npr.org/1128/rss.xml",
    "https://feeds.npr.org/1131/rss.xml",

    # Al Jazeera
    "https://www.aljazeera.com/xml/rss/all.xml",

    # The Guardian
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/us-news/rss",
    "https://www.theguardian.com/business/rss",
    "https://www.theguardian.com/technology/rss",
    "https://www.theguardian.com/science/rss",
    "https://www.theguardian.com/sport/rss",

    # Deutsche Welle
    "https://rss.dw.com/rdf/rss-en-all",

    # CNBC
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/19746125/device/rss/rss.html",

    # Reuters / AP (may be limited, kept as bonus sources)
    "https://www.reutersagency.com/feed/?best-topics=top-news&post_type=best",
    "https://apnews.com/apf-topnews?output=rss",
]


def fetch_recent_real_news():

    rows = []
    seen_titles = set()

    for feed_url in RSS_FEEDS:

        print(f"Fetching: {feed_url}")

        try:
            feed = feedparser.parse(feed_url)
        except Exception as error:
            print(f"  Failed to fetch {feed_url}: {error}")
            continue

        for entry in feed.entries:

            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()

            if not title or title.lower() in seen_titles:
                continue

            text = f"{title}. {summary}".strip()

            if len(text) < 40:
                continue

            seen_titles.add(title.lower())
            rows.append({"text": text})

        print(f"  Collected so far: {len(rows)} unique articles")

    return rows


if __name__ == "__main__":

    print("\nCollecting current real news from RSS feeds...\n")

    articles = fetch_recent_real_news()

    if not articles:
        print("\nNo articles collected. Check your internet connection or feed URLs.")
    else:

        with open(OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as file:

            writer = csv.DictWriter(file, fieldnames=["text"])
            writer.writeheader()

            for row in articles:
                writer.writerow(row)

        print(f"\nSaved {len(articles)} real news articles to:\n{OUTPUT_PATH}")
        print("\nNOTE: Run this script periodically (e.g. weekly) to keep this file current.")