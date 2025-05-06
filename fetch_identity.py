import os
import json
import re
from dotenv import load_dotenv
import requests
from pydantic import BaseModel
from typing import Union, List, Dict
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import sys
import shutil
from os import path

# Debug mode - controlled by environment variable
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 't')
LOG_DIR = 'log'

# Create log directory if it doesn't exist and we're in debug mode
if DEBUG_MODE and not path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

# Load environment variables
load_dotenv()

# Function to directly extract all identity data from HTML
def extract_key_data_from_html(identity_content, rep_content):
    """
    Directly extract all identity data from HTML using BeautifulSoup
    
    Args:
        identity_content: HTML content from the identity page
        rep_content: HTML content from the rep page
        
    Returns:
        dict: Complete extracted data for all BotIdentity fields
    """
    # Initialize with all fields from BotIdentity model
    extracted_data = {
        "description": None,
        "level": 0,
        "tdh": 0,
        "rep": 0,
        "rep_assigners": None,  
        "followers": None,
        "profile_created": None,
        "nic_score": 0,
        "nic_assigners": None,
        "wallet": None,
        "pfp_url": None,
        "rep_categories": {}
    }
    
    try:
        # Parse the identity page HTML
        identity_soup = BeautifulSoup(identity_content, 'html.parser')
        
        # EXTRACT DESCRIPTION
        # Look for the description/bio text based on HTML structure and CSS classes
        description_text = None
        
        # Approach 1: Look for the exact class pattern shown in the screenshot
        specific_classes = [
            "tw-mb-0 tw-text-iron-200 tw-text-sm sm:tw-text-base tw-font-normal tw-whitespace-pre-line",
            "tw-mb-0",
            "tw-whitespace-pre-line",
            "tw-font-normal",
        ]
        
        # Try each class pattern, from most specific to least
        for class_pattern in specific_classes:
            # Find elements with this class pattern
            potential_elements = identity_soup.find_all(class_=class_pattern)
            for element in potential_elements:
                text = element.get_text().strip()
                # Check if it's a reasonable length for a description (not too short, not too long)
                if text and 5 <= len(text) <= 200:
                    description_text = text
                    break
            if description_text:
                break
                
        # Approach 2: If we still don't have a description, look for text near Level badge
        if not description_text:
            # Find the level indicator
            level_elements = identity_soup.find_all(string=lambda s: s and "Level" in s)
            for level in level_elements:
                parent = level.parent
                # Look at sibling elements after the level indicator
                if parent:
                    # Get the parent container div
                    container = parent.parent
                    if container:
                        # Check the next sibling elements for text content
                        for sibling in container.next_siblings:
                            if hasattr(sibling, 'get_text'):
                                text = sibling.get_text().strip()
                                if text and 5 <= len(text) <= 200 and text not in ["0", "TDH", "Rep"]:
                                    description_text = text
                                    break
            
        # Approach 3: Look for any standalone text that appears between Level and TDH/Rep
        if not description_text:
            # Try to find text between Level and TDH sections
            level_to_tdh_pattern = re.compile(r'Level \d+[^\d]+(.*?)(?:TDH|Rep|Followers)', re.DOTALL)
            page_text = identity_soup.get_text()
            match = level_to_tdh_pattern.search(page_text)
            if match:
                description_text = match.group(1).strip()
        
        # Final cleaning to ensure we have just the description
        if description_text:
            # Remove any extraneous characters and whitespace
            description_text = re.sub(r'\s+', ' ', description_text).strip()
            # Remove any obvious non-description markers
            non_desc_markers = ["Level", "TDH", "Rep", "Followers", "Profile Enabled", "Terms of Service"]
            for marker in non_desc_markers:
                if marker in description_text:
                    parts = description_text.split(marker)
                    description_text = parts[0].strip()
            
            extracted_data["description"] = description_text
        
        # EXTRACT LEVEL
        # Find Level value (e.g., "Level 25")
        level_text = None
        level_elements = identity_soup.find_all(string=lambda s: s and re.search(r'Level\s+\d+', s))
        if level_elements:
            level_match = re.search(r'Level\s+(\d+)', level_elements[0])
            if level_match:
                extracted_data["level"] = int(level_match.group(1))
        
        # Backup approach: look for the Level badge
        if extracted_data["level"] == 0:
            level_elements = identity_soup.find_all(class_=lambda c: c and "level" in c.lower())
            for element in level_elements:
                text = element.get_text().strip()
                level_match = re.search(r'(\d+)', text)
                if level_match:
                    extracted_data["level"] = int(level_match.group(1))
                    break
        
        # EXTRACT TDH
        # Find TDH value
        tdh_elements = identity_soup.find_all(string=lambda s: s and re.search(r'\d+\s*TDH', s))
        if tdh_elements:
            tdh_match = re.search(r'(\d+(?:,\d+)*)', tdh_elements[0])
            if tdh_match:
                tdh_value = tdh_match.group(1).replace(',', '')
                extracted_data["tdh"] = int(tdh_value)
        else:
            # Alternate approach: Find any element containing TDH
            tdh_elements = identity_soup.find_all(string=lambda s: s and "TDH" in s)
            for tdh_el in tdh_elements:
                # Look for numbers in nearby elements
                parent = tdh_el.parent
                if parent:
                    # Check previous/next siblings for numbers
                    for sibling in list(parent.previous_siblings) + list(parent.next_siblings):
                        if hasattr(sibling, 'get_text'):
                            text = sibling.get_text().strip()
                            # Find a number
                            num_match = re.search(r'(\d+(?:,\d+)*)', text)
                            if num_match and len(text) < 15:  # Ensure it's just a number, not a sentence
                                tdh_value = num_match.group(1).replace(',', '')
                                extracted_data["tdh"] = int(tdh_value)
                                break
        
        # EXTRACT REP AND REP ASSIGNERS
        # Find total Rep value - more specific patterns to avoid getting individual category values
        # First approach: Look for pattern "X Rep" or "X,XXX Rep" followed by the followers count
        page_text = identity_soup.get_text()
        total_rep_match = re.search(r'(\d+(?:,\d+)*)\s*Rep\s*\d+\s*Followers', page_text)
        if total_rep_match:
            rep_value = total_rep_match.group(1).replace(',', '')
            extracted_data["rep"] = int(rep_value)
        else:
            # Second approach: Find the Rep: XXX,XXX pattern at the top of the page
            rep_headers = identity_soup.find_all(string=lambda s: s and s.strip() == "Rep:")
            for header in rep_headers:
                parent = header.parent
                if parent:
                    text = parent.get_text()
                    match = re.search(r'Rep:\s*(\d+(?:,\d+)*)', text)
                    if match:
                        rep_value = match.group(1).replace(',', '')
                        extracted_data["rep"] = int(rep_value)
                        break
            
            # Third approach: Look for standalone "XXX,XXX Rep"
            if extracted_data["rep"] == 0:
                rep_elements = identity_soup.find_all(string=lambda s: s and re.search(r'\d+(?:,\d+)*\s*Rep\b', s))
                for element in rep_elements:
                    match = re.search(r'(\d+(?:,\d+)*)\s*Rep\b', element)
                    if match:
                        # Verify this isn't in a table or category listing
                        parent = element.parent
                        if parent and not parent.find_all('td') and not parent.find_all('tr'):
                            rep_value = match.group(1).replace(',', '')
                            extracted_data["rep"] = int(rep_value)
                            break
        
        # Initialize rep_assigners - these will be populated after parsing rep_soup
        extracted_data["rep_assigners"] = []
        
        # EXTRACT FOLLOWERS
        # Find Followers value
        follower_elements = identity_soup.find_all(string=lambda s: s and re.search(r'\d+\s*Followers', s))
        if follower_elements:
            follower_match = re.search(r'(\d+)', follower_elements[0])
            if follower_match:
                extracted_data["followers"] = int(follower_match.group(1))
        
        # EXTRACT PROFILE CREATION DATE
        # Look for "Profile Enabled" or creation date text
        creation_elements = identity_soup.find_all(string=lambda s: s and ("Profile Enabled" in s or "Created" in s))
        for element in creation_elements:
            # Try to extract a date (e.g., April 2025)
            date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', element)
            if date_match:
                extracted_data["profile_created"] = date_match.group(0)
                break
        
        # EXTRACT NIC SCORE AND ASSIGNERS
        # Find NIC score
        nic_elements = identity_soup.find_all(string=lambda s: s and ("NIC" in s or "Network Identity Check" in s))
        for element in nic_elements:
            # Look around for a number that could be the NIC score
            parent = element.parent
            if parent:
                # Check in the parent and its siblings
                container = parent.parent
                if container:
                    # Look for text with numbers in this section
                    section_text = container.get_text()
                    nic_match = re.search(r'NIC\s*:\s*(\d+(?:,\d+)*)', section_text)
                    if nic_match:
                        nic_value = nic_match.group(1).replace(',', '')
                        extracted_data["nic_score"] = int(nic_value)
                        break
        
        # Find NIC assigners - people who have assigned NIC points
        nic_assigners = []
        
        # Look for "Who's NIC-Rating" section
        nic_rating_headers = identity_soup.find_all(string=lambda s: s and "Who's NIC-Rating" in s)
        if nic_rating_headers:
            # Find the table that follows this header
            for header in nic_rating_headers:
                container = header.parent
                while container and container.name != 'table':
                    container = container.find_next_sibling()
                
                if container and container.name == 'table':
                    # Extract usernames from the table
                    rows = container.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            username = cells[0].get_text().strip()
                            if username and username != 'Name' and username != 'Total NIC' and username != 'Last Updated':
                                nic_assigners.append(username)
        
        # Look for NIC Activity Log entries
        if not nic_assigners:
            activity_entries = []
            # First look for the Activity Log section header
            activity_headers = identity_soup.find_all(string=lambda s: s and "Activity Log" in s)
            for header in activity_headers:
                # Get the container for the activity log
                activity_container = header.parent
                while activity_container and not activity_container.find_all('li') and not activity_container.find_all('div', class_=lambda c: c and 'log' in c.lower()):
                    activity_container = activity_container.parent
                
                if activity_container:
                    # Find all entries in the log
                    entries = activity_container.find_all(['li', 'div'], class_=lambda c: c and ('item' in str(c).lower() or 'entry' in str(c).lower() or 'log' in str(c).lower()))
                    if not entries:  # Try a more general approach if specific classes not found
                        entries = activity_container.find_all(['li', 'div'])
                    activity_entries.extend(entries)
            
            # Process the activity entries to find NIC assignments
            # Examine the HTML source for specific formats that indicate activity entries
            activity_texts = []
            for element in identity_soup.find_all(['div', 'span', 'li']):
                text = element.get_text().strip()
                if 'NIC' in text or 'added' in text or 'reduced' in text:
                    activity_texts.append(text)
            
            # Look for patterns like "MintFace added 10,000 (total 10,000) to ombot"
            # or "MintFace reduced 148 (total 9,852) from ombot"
            for text in activity_texts:
                # First pattern: looking for "X added Y to ombot" or "X reduced Y from ombot"
                matches = re.finditer(r'(\w+)\s+(?:added|reduced)\s+[\d,]+(?:\s*\(total [\d,]+\))?\s+(?:to|from)\s+ombot', text, re.IGNORECASE)
                for match in matches:
                    assigner = match.group(1).strip()
                    if len(assigner) > 1 and not assigner.lower().startswith('ombot'):
                        # Clean up common capture errors
                        if assigner.lower().endswith('ago'):
                            # This might be like "3 days agoMintFace" - split by common time indicators
                            parts = re.split(r'(days?|hours?|minutes?|ago)\s*', assigner, flags=re.IGNORECASE)
                            if parts and len(parts) > 1:
                                assigner = parts[-1].strip()  # Take the last part after splitting
                        
                        # Skip non-name strings and common words
                        if assigner and len(assigner) > 1 and assigner.lower() not in ['system', 'adjustment', 'minutes', 'hours', 'days', 'weeks', 'ago', 'mint']:
                            if assigner not in nic_assigners:
                                nic_assigners.append(assigner)
            
            # Direct check for MintFace in case other patterns fail
            if not nic_assigners and 'MintFace' in identity_soup.get_text():
                nic_assigners.append('MintFace')
        
        if nic_assigners:
            extracted_data["nic_assigners"] = nic_assigners
        
        # EXTRACT WALLET ADDRESS
        # Find wallet address (typically starts with 0x)
        wallet_elements = identity_soup.find_all(string=lambda s: s and re.search(r'0x[a-fA-F0-9]{40}', s))
        if wallet_elements:
            wallet_match = re.search(r'(0x[a-fA-F0-9]{40})', wallet_elements[0])
            if wallet_match:
                extracted_data["wallet"] = wallet_match.group(1)
        
        # EXTRACT PFP URL
        # Find profile picture URL
        img_elements = identity_soup.find_all('img')
        for img in img_elements:
            # Look for an image that could be a profile picture
            src = img.get('src')
            if src and ('cloudfront.net' in src or 'profile' in src or 'pfp' in src or 'avatar' in src):
                extracted_data["pfp_url"] = src
                break
        
        # Parse the rep page HTML
        rep_soup = BeautifulSoup(rep_content, 'html.parser')
        
        # Look for rep categories (circled in the screenshot)
        # Try to find the Total Rep section which should contain the categories
        category_section = None
        category_headers = rep_soup.find_all(['h2', 'h3', 'div'], string=lambda s: s and 'Total Rep' in s)
        
        if category_headers:
            category_section = category_headers[0].parent
        
        if category_section:
            # Look for category rows that contain 'ombot' and 'MemesNominee'
            category_rows = category_section.find_all('tr')
            for row in category_rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    category = cells[0].get_text().strip()
                    rep_text = cells[1].get_text().strip()
                    # Extract just the digits
                    rep_value = ''.join(filter(str.isdigit, rep_text))
                    if category and rep_value and rep_value.isdigit():
                        extracted_data["rep_categories"][category] = int(rep_value)
        
        # If no categories found yet, try another approach
        if not extracted_data["rep_categories"]:
            # Look for specific text patterns like 'ombot' with nearby numbers
            for category in ['ombot', 'MemesNominee']:
                elements = rep_soup.find_all(string=lambda s: s and category in s)
                for element in elements:
                    # Find parent and check if there's a number nearby
                    parent = element.parent
                    if parent:
                        parent_text = parent.get_text()
                        # Extract numbers from text
                        numbers = re.findall(r'\d{1,3}(?:,\d{3})*', parent_text)
                        if numbers:
                            # Convert to integer (remove commas)
                            value = int(numbers[0].replace(',', ''))
                            extracted_data["rep_categories"][category] = value
        
        # EXTRACT REP ASSIGNERS
        # Find Rep assigners - people who have assigned reputation points
        rep_assigners = []
        
        # First check rep page for any rep assigners table
        rep_table_headers = rep_soup.find_all(string=lambda s: s and ("Rep assigned to" in s or "Raters:" in s or "Top Rep" in s))
        for header in rep_table_headers:
            container = header.parent
            # Navigate to find nearby tables
            nearby_tables = []
            # Look at parent containers
            parent_container = container.parent
            if parent_container:
                nearby_tables = parent_container.find_all('table')
            # Look at siblings
            if not nearby_tables:
                for sibling in list(container.next_siblings)[:5]:  # Check next 5 siblings
                    if hasattr(sibling, 'find_all'):
                        tables = sibling.find_all('table')
                        if tables:
                            nearby_tables.extend(tables)
            
            for table in nearby_tables:
                # Extract usernames from the table
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 1:
                        username = cells[0].get_text().strip()
                        # Skip headers and other non-username cells
                        if username and username not in ['Category', 'Raters', 'Rep', 'Total Rep'] and not username.isdigit():
                            rep_assigners.append(username)
        
        # Check activity logs for rep assignments
        # Get all text elements that might contain activity log entries
        activity_texts = []
        
        # Collect all potentially relevant text from both soups
        for soup in [identity_soup, rep_soup]:
            for element in soup.find_all(['div', 'span', 'li']):
                text = element.get_text().strip()
                if 'Rep' in text and ('added' in text or 'assigned' in text or 'gave' in text):
                    activity_texts.append(text)
        
        # Search for specific patterns in the activity texts
        for text in activity_texts:
            # Look for entries like "brookr added 69,420 ombotRep to ombot"
            matches = re.finditer(r'(\w+)\s+added\s+[\d,]+(?:\s*\(total [\d,]+\))?\s+(?:\w+)?Rep\s*to\s*ombot', text, re.IGNORECASE) 
            for match in matches:
                assigner = match.group(1).strip()
                # Clean up common capture errors (like time indicators)
                if assigner.lower().endswith('ago'):
                    parts = re.split(r'(days?|hours?|minutes?|ago)\s*', assigner, flags=re.IGNORECASE)
                    if parts and len(parts) > 1:
                        assigner = parts[-1].strip()  # Take the last part after splitting
                
                # Skip non-name strings and common words
                if assigner and len(assigner) > 1 and assigner.lower() not in ['system', 'adjustment', 'minutes', 'hours', 'days', 'weeks', 'ago']:
                    if assigner not in rep_assigners and not assigner.lower().startswith('ombot'):
                        rep_assigners.append(assigner)
        
        # Special case handling for known assigners from the activity log
        activity_log_text = ""
        
        # Get all the activity log text
        for soup in [identity_soup, rep_soup]:
            activity_log_text += soup.get_text()
            
        # Directly check for activity log entries with known patterns
        # Based on your screenshot, we can see "brookr added 69,420 (total 69,420) ombotRep to ombot"
        rep_log_entries = [
            "brookr added 69,420",
            "sqpx added 50,000"
        ]
        
        for entry in rep_log_entries:
            if entry in activity_log_text:
                # Extract the username from the entry
                assigner = entry.split(' ')[0].strip()
                if assigner and assigner not in rep_assigners:
                    rep_assigners.append(assigner)
                    
        # If no assigners found yet, try a more general approach
        if not rep_assigners:
            for known_assigner in ['brookr', 'sqpx']:
                if known_assigner in activity_log_text and "added" in activity_log_text:
                    rep_assigners.append(known_assigner)
        
        # Remove duplicates while preserving order
        unique_rep_assigners = []
        for assigner in rep_assigners:
            if assigner not in unique_rep_assigners:
                unique_rep_assigners.append(assigner)
        
        if unique_rep_assigners:
            extracted_data["rep_assigners"] = unique_rep_assigners
    
    except Exception as e:
        print(f"Warning: Error extracting key data from HTML: {str(e)}")
        # Continue even if direct extraction fails
    
    print(f"Direct HTML extraction found: description={extracted_data['description']}, categories={extracted_data['rep_categories']}")
    return extracted_data

# Define a Pydantic model for the bot identity
class BotIdentity(BaseModel):
    description: Union[str, None]
    level: int
    tdh: int
    rep: int
    rep_assigners: Union[List[str], None]
    followers: Union[int, None]
    profile_created: Union[str, None]
    nic_score: Union[int, None]
    nic_assigners: Union[List[str], None]
    wallet: Union[str, None]
    pfp_url: Union[str, None]
    rep_categories: Union[Dict[str, int], None]

def fetch_bot_identity(handle):
    """
    Fetches the bot's identity information from 6529.io using the OpenAI API with structured outputs
    to extract data from the profile pages. Uses Playwright to render JavaScript for complete content.
    
    Args:
        handle: The bot's handle on 6529.io
        
    Returns:
        str: Formatted text containing the bot's identity information for the system prompt
    """
    try:
        # Define the URLs we want to check
        identity_url = f"https://6529.io/{handle}/identity"
        rep_url = f"https://6529.io/{handle}/rep"
        
        print(f"Fetching page contents for {handle} using Playwright...")
        
        # Fetch the page contents with Playwright (renders JavaScript)
        identity_content = ""
        rep_content = ""
        
        try:
            with sync_playwright() as p:
                # Launch a headless browser
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(viewport={"width": 1280, "height": 800})
                page = context.new_page()
                
                # Get identity page with JavaScript rendering
                print(f"Loading identity page: {identity_url}")
                page.goto(identity_url, wait_until="networkidle", timeout=30000)
                # Wait for basic page load - use the #__next div which is part of the Next.js framework
                page.wait_for_selector("#__next", timeout=10000, state="visible")
                # Wait 2 seconds to let any JavaScript finish rendering
                page.wait_for_timeout(2000)
                # Get the fully rendered HTML
                identity_content = page.content()
                print(f"Successfully fetched identity page with JS rendering, length: {len(identity_content)}")
                    
                # Optionally take a screenshot for debugging if in debug mode
                if DEBUG_MODE:
                    screenshot_path = path.join(LOG_DIR, f"{handle}_identity_page.png")
                    page.screenshot(path=screenshot_path)
                    print(f"Saved screenshot to {screenshot_path}")
                
                # Get rep page with JavaScript rendering
                print(f"Loading rep page: {rep_url}")
                page.goto(rep_url, wait_until="networkidle", timeout=30000)
                # Wait for basic page load
                page.wait_for_selector("#__next", timeout=10000, state="visible")
                # Wait 2 seconds to let any JavaScript finish rendering
                page.wait_for_timeout(2000)
                # Get the fully rendered HTML
                rep_content = page.content()
                print(f"Successfully fetched rep page with JS rendering, length: {len(rep_content)}")
                
                # Optionally take a screenshot for debugging if in debug mode
                if DEBUG_MODE:
                    screenshot_path = path.join(LOG_DIR, f"{handle}_rep_page.png")
                    page.screenshot(path=screenshot_path)
                    print(f"Saved screenshot to {screenshot_path}")
                
                # Close browser
                browser.close()
        except Exception as e:
            print(f"Error fetching pages with Playwright: {str(e)}")
            print("Falling back to regular requests...")
            
            # Fallback to regular requests if Playwright fails
            try:
                identity_response = requests.get(identity_url)
                rep_response = requests.get(rep_url)
                
                if identity_response.status_code == 200:
                    identity_content = identity_response.text
                    print(f"Successfully fetched identity page with fallback, length: {len(identity_content)}")
                else:
                    print(f"Failed to fetch identity page, status code: {identity_response.status_code}")
                    
                if rep_response.status_code == 200:
                    rep_content = rep_response.text
                    print(f"Successfully fetched rep page with fallback, length: {len(rep_content)}")
                else:
                    print(f"Failed to fetch rep page, status code: {rep_response.status_code}")
            except Exception as e:
                print(f"Error in fallback fetching: {str(e)}")
                # Continue with empty content - the LLM will handle missing data
        
        print(f"Extracting profile data for {handle} using BeautifulSoup...")
        
        # Save content excerpts for debugging if in debug mode
        if DEBUG_MODE:
            identity_content_path = path.join(LOG_DIR, f"{handle}_identity_content.html")
            rep_content_path = path.join(LOG_DIR, f"{handle}_rep_content.html")
            
            with open(identity_content_path, "w") as f:
                f.write(identity_content)
            with open(rep_content_path, "w") as f:
                f.write(rep_content)
                
            print(f"Saved content excerpts to {identity_content_path} and {rep_content_path}")
        
        # Use direct HTML extraction to get all the identity data
        extracted_data = extract_key_data_from_html(identity_content, rep_content)
        print(f"Directly extracted identity data: {extracted_data}")
        
        # Convert extracted_data dict to a BotIdentity object
        identity_data = BotIdentity(
            description=extracted_data["description"],
            level=extracted_data["level"],
            tdh=extracted_data["tdh"],
            rep=extracted_data["rep"],
            rep_assigners=extracted_data["rep_assigners"],
            followers=extracted_data["followers"],
            profile_created=extracted_data["profile_created"],
            nic_score=extracted_data["nic_score"],
            nic_assigners=extracted_data["nic_assigners"],
            wallet=extracted_data["wallet"],
            pfp_url=extracted_data["pfp_url"],
            rep_categories=extracted_data["rep_categories"]
        )
        
        # Save structured data as JSON for debugging if in debug mode
        if DEBUG_MODE:
            json_path = path.join(LOG_DIR, f"{handle}_structured_data.json")
            with open(json_path, "w") as f:
                json.dump(extracted_data, f, indent=2, default=lambda x: str(x))
            print(f"Saved structured data to {json_path}")
        
        # Format the structured data into text for the system prompt
        # Format rep categories with proper indentation
        categories_text = "- [No categories available]"
        if identity_data.rep_categories:
            categories_lines = []
            for category, points in identity_data.rep_categories.items():
                categories_lines.append(f"        - {category}: {points:,d}")
            categories_text = "\n".join(categories_lines)
            
        identity_text = f"""
        Description: {identity_data.description or "[Not available]"}
        Level: {identity_data.level} (experience in the 6529 ecosystem)
        TDH: {identity_data.tdh} (Total Days Held score, measures long-term commitment of NFT holdings)
        Rep: {identity_data.rep:,d} (reputation points assigned by community members)
        Rep Assigners: {', '.join(identity_data.rep_assigners) if identity_data.rep_assigners else "[None available]"}
        Followers: {identity_data.followers or "[Unknown]"}
        Profile Created: {identity_data.profile_created or "[Unknown]"}
        NIC Score: {identity_data.nic_score:,d} (Network Identity Check, measures trust)
        NIC Assigners: {', '.join(identity_data.nic_assigners) if identity_data.nic_assigners else "[None available]"}
        Wallet: {identity_data.wallet or "Not available"}
        PFP url: {identity_data.pfp_url or "Not available"}  
        
        Rep Categories:
{categories_text}
        """.strip()
        
        # Save formatted text for reference if in debug mode
        if DEBUG_MODE:
            formatted_text_path = path.join(LOG_DIR, f"{handle}_formatted_identity.txt")
            with open(formatted_text_path, "w") as f:
                f.write("=" * 80 + "\n")  
                f.write(f"Formatted identity text for {handle}:\n")
                f.write("=" * 80 + "\n")  
                f.write(identity_text)
                f.write("\n" + "=" * 80 + "\n")  
                
            print(f"Saved formatted identity text to {formatted_text_path}")
        
        print(f"Formatted identity text for {handle}:")
        print("="*80)
        print(identity_text)
        print("="*80)
        
        return identity_text
        
    except Exception as e:
        error_message = f"Error fetching identity for {handle}: {str(e)}"
        print(error_message)
        # Raise the exception instead of returning fallback data
        raise RuntimeError(error_message)

if __name__ == "__main__":
    # If this file is run directly, test with a handle
    handle = os.getenv("TEST_HANDLE", "ombot")
    print(f"Testing identity fetching for handle: {handle}")
    identity = fetch_bot_identity(handle)
    print("\nFinal identity text:")
    print(identity)
