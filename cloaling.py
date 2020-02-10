
from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(storage={"root_dir": "images"})
crawler.crawl(keyword="321系", max_num=1500)
