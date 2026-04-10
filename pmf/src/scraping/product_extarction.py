

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
from PIL import Image
from io import BytesIO

# ---------- Helper function ----------
# def download_and_save(url, filename):
def download_and_save(url):

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Use relative path to existing directory
        filename = r"..\..\data\scrape_input\product_image\product_image.jpg"

        # Ensure JPG format (even if AVIF or PNG)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(filename, "JPEG")
        print(f"✅ Saved {filename}")
    else:
        print(f"❌ Failed to download {url}")

# def product_image_extraction(search_text, filename):
def product_image_extraction(search_text):

    # ---------- Selenium setup ----------
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)

    url = "https://www.thermofisher.com/in/en/home/life-science/lab-equipment.html"

    driver.get(url)
    wait = WebDriverWait(driver, 15)

    # 1. Accept cookies
    try:
        accept_btn = wait.until(EC.element_to_be_clickable((By.ID, "truste-consent-button")))
        accept_btn.click()
        print("Cookies accepted ✅")
    except:
        print("No cookie popup found ❌")

    # 2. Search product
    search_box = wait.until(EC.presence_of_element_located((By.ID, "suggest1")))
    search_box.clear()
    search_box.send_keys(search_text)

    search_button = driver.find_element(By.ID, "searchButton")
    search_button.click()

    # 3. Wait for product cards
    wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "search-card")))


    # 5. Open first product card in new tab
    first_card_link = driver.find_element(By.CSS_SELECTOR, ".search-card .search-result-title-brand a")
    product_url = first_card_link.get_attribute("href")

    driver.execute_script("window.open(arguments[0], '_blank');", product_url)

    # Switch to new tab
    driver.switch_to.window(driver.window_handles[1])

    # 6. Get first product page image
    product_img = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".pdp-item-image")))
    prod_img_url = product_img.get_attribute("src")

    # Handle relative URL
    if prod_img_url.startswith("/"):
        prod_img_url = "https://www.thermofisher.com" + prod_img_url

    # download_and_save(prod_img_url, filename)
    download_and_save(prod_img_url)


    driver.quit()


# search_text = "Thermo Scientific TDE Series -86°C Ultra-Low Temperature (ULT) Freezer" #"Thermo Scientific TSX Universal"
# filename = r"C:\Users\naveen.jallepalli\OneDrive - Thermo Fisher Scientific\Documents\TFS_POC\CER\CER_Final_with_database_git\RegulatoryDocGen\data\scrape_input\product_image\product_image2.jpg"

# product_image_extraction(search_text,filename)
# , filename)
# product_image_extraction(search_text)

