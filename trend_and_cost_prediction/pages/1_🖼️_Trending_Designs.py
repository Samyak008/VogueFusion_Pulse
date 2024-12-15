import streamlit as st
from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os

# Function to download images
def download_images_with_icrawler(keyword, limit, output_folder="downloads"):
    crawler = GoogleImageCrawler(storage={"root_dir": output_folder})
    crawler.crawl(keyword=keyword, max_num=limit)
    return output_folder

# Streamlit App
def main():
    st.title("Trending Designs")
    st.write("Enter a keyword to search for images.")

    # Input Section
    keyword = st.text_input("Enter the search query", placeholder="e.g., trending ethnic traditional cotton saree printing design patterns template")
    limit = st.number_input("Number of images to download", min_value=1, max_value=50, value=5)
    submit = st.button("Search and Download Images")

    # Output Section
    if submit and keyword:
        with st.spinner("Downloading images..."):
            output_folder = "downloads"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Download images
            folder_path = download_images_with_icrawler(keyword, limit, output_folder)

            # Display the images
            image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('png', 'jpg', 'jpeg'))]

            if image_files:
                st.success(f"Downloaded {len(image_files)} images!")
                for image_file in image_files:
                    img = Image.open(image_file)
                    st.image(img, caption=os.path.basename(image_file), use_column_width=True)

                    # Download button
                    with open(image_file, "rb") as file:
                        st.download_button(
                            label="Download this image",
                            data=file,
                            file_name=os.path.basename(image_file),
                            mime="image/jpeg"
                        )

                    # Delete the image after processing
                    os.remove(image_file)

                # Clean up the folder if empty
                if not os.listdir(folder_path):
                    os.rmdir(folder_path)
            else:
                st.error("No images found. Try a different keyword.")

# Run the app
if __name__ == "__main__":
    main()
