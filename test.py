from icrawler.builtin import BingImageCrawler
import sys

args = sys.argv

crawler = BingImageCrawler(parser_threads=2, downloader_threads=4,storage={"root_dir": "225"})
crawler.session.verify = False
crawler.crawl(keyword="225系", max_num=1500,)