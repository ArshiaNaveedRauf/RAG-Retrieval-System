from src.data.scraper import WebsiteScraper
from config import Url

def test_my_scraper():
    print("\n--- SCRAPER TOOL TEST START ---")
    
    # 1. Initialize the scraper
    scraper = WebsiteScraper()
    
    # 2. Fetch the website data
    url = "https://www.neduet.edu.pk/"
    chunks = scraper.get_website_data(url)

    assert isinstance(chunks,list)
    assert len(chunks) > 0
    



