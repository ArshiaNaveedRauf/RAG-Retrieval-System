from core.scraper import WebsiteScraper

def test_my_scraper():
    print("\n--- SCRAPER TOOL TEST START ---")
    
    # 1. Initialize the scraper
    scraper = WebsiteScraper()
    
    # 2. Fetch the website data
    url = "https://www.neduet.edu.pk/"
    chunks = scraper.get_website_data(url)
    
    # 3. Check the output
    if chunks:
        print("\n--- FIRST CHUNK DATA (Cleaned) ---")
        # Print only the first chunk for testing purposes
        print(chunks[0].page_content)
        print("----------------------------------\n")
        print("Test Successful! Data cleaned and divided into chunks.")
    else:
        print("Test Failed! No data found.")

if __name__ == "__main__":
    test_my_scraper()