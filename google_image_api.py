# Documentation: https://developers.google.com/custom-search/v1/using_rest
import requests

api_key = "AIzaSyCuYr2tC0X9D89i5sRFQlzPtNTuDzPVvGU"
search_term = "AUDI A6 AVANT QUATTRO 3 TDI"
search_type = "image"
image_type = "photo"
num_results = 1
custom_search_engine_id = "025b404de865d4d24"

url = "https://customsearch.googleapis.com/customsearch/v1"

params = {
    "key": api_key,
    "q": search_term,
    "searchType": search_type,
    "imgType": image_type,
    "num": num_results,
    "cx": custom_search_engine_id 
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an exception for bad status codes

    search_results = response.json()

    # Process the search results
    if "items" in search_results and search_results["items"]:
        first_image_result = search_results["items"][0]
        print("First image result found:")
        print(f"Title: {first_image_result.get('title')}")
        print(f"Link: {first_image_result.get('link')}")
        print(f"Snippet: {first_image_result.get('snippet')}")
        # You can access other information like image dimensions, context link, etc.
    else:
        print("No image results found for the query using your custom search engine.")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")