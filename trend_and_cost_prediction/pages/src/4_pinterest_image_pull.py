from pinscrape import scraper, Pinterest
import os

def download_images_from_pinterest(keyword="latest saree design closeup", output_folder="images", images_to_download=2, proxies=None, number_of_workers=5):

    if proxies is None:
        proxies = {}

    p = Pinterest(proxies=proxies)
    images_url = p.search(keyword, images_to_download)
    p.download(url_list=images_url, number_of_workers=number_of_workers, output_folder=output_folder)

# Example usage within this script
if __name__ == "__main__":
    # Default usage
    download_images_from_pinterest()

    # Customized usage
    # download_images_from_pinterest(keyword="saree design", images_to_download=10, output_folder="images")
