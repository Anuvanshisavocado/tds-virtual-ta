import os
import requests
from bs4 import BeautifulSoup

def scrape_example_html_content():
    """
    Creates dummy HTML files and then 'scrapes' them to simulate
    extracting course content. In a real scenario, you would modify
    this to fetch content from actual course URLs.
    """
    output_dir = "data/course_content"
    os.makedirs(output_dir, exist_ok=True)

    # Define some dummy HTML content representing course lectures/notes
    dummy_files = [
        ("data/course_content/tds_intro.html", "<h1>Introduction to TDS</h1><p>Welcome to Tools in Data Science. This course covers Python, Pandas, NumPy, and more. Make sure to install Anaconda.</p>"),
        ("data/course_content/tds_week1_notes.html", "<h2>Week 1 Notes: Data Structures</h2><p>Remember to install Anaconda and Jupyter notebooks for the first week's assignments. We'll cover Lists, Tuples, Dictionaries, and Sets in Python. Pay special attention to their immutability properties.</p>"),
        ("data/course_content/tds_pandas_basics.html", "<h3>Pandas Basics</h3><p>Pandas is a powerful library for data manipulation. Key objects are DataFrame and Series. You can filter DataFrames using boolean indexing: `df[df['column'] > value]`.</p>")
    ]

    print("Creating dummy HTML files for course content simulation...")
    for filename, content in dummy_files:
        # Create the dummy HTML file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created dummy file: {filename}")

    print("\n'Scraping' text content from dummy HTML files...")
    # Now, process these dummy HTML files to extract plain text,
    # mimicking a real scraping process
    for filename, _ in dummy_files:
        with open(filename, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Use BeautifulSoup to parse the HTML and extract text
        soup = BeautifulSoup(html_content, 'html.parser')
        # Get all text, separating paragraphs with newlines
        text_content = soup.get_text(separator='\n', strip=True)

        # Save the extracted text to a .txt file
        output_file_name = os.path.basename(filename).replace(".html", ".txt")
        output_file_path = os.path.join(output_dir, output_file_name)
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"Extracted and saved: {output_file_path}")

if __name__ == "__main__":
    scrape_example_html_content()
    print("\nCourse content 'scraping' simulation complete.")