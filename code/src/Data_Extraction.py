import requests

def get_entity_id(name):
    """Search for an entity and return its QID."""
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "format": "json"
    }
    
    response = requests.get(search_url, params=params).json()
    
    if "search" in response and response["search"]:
        return response["search"][0]["id"]  # Return first result's ID
    return None

def get_entity_details(name, language="en"):
    """Get full entity details in a single language."""
    entity_id = get_entity_id(name)
    if not entity_id:
        return None  # No entity found

    details_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "languages": language,  # Restrict results to one language
        "format": "json"
    }
    
    response = requests.get(details_url, params=params).json()
    if response is None:
      response = {}
    return (entity_id, response)

def get_wikipedia_title(entity_id, entity_details, language="en"):

    sitelinks = entity_details["entities"].get(entity_id, {}).get("sitelinks", {})
    wiki_key = f"{language}wiki"
    
    if wiki_key in sitelinks:
        title = sitelinks[wiki_key]["title"]
        return title
    
    return None

def get_wikipedia_content(title):
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "format": "json"
    }
        
    response = requests.get(search_url, params=params).json()

    pages = response["query"]["pages"]
    page_content = next(iter(pages.values()))
    return (page_content["title"], page_content["extract"])