import os
import re
import json
import atexit
import time
from flask import Flask
from flask import render_template, request, redirect, url_for, jsonify, Blueprint
import requests
from dateutil import parser
from openai import OpenAI
from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_defunct
from flask_migrate import Migrate
from datetime import datetime, timezone, timedelta
from sqlalchemy.exc import SQLAlchemyError
from collections import defaultdict
from models import db, Wave, Drop, Author, WaveTracking
from requests.exceptions import HTTPError

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['SQLALCHEMY_ECHO'] = True

# # Initialize SQLAlchemy
db.init_app(app)
migrate = Migrate(app, db)

# app.register_blueprint(main_bp)
# app.register_blueprint(network_bp)

chosen_wave = "maybe's dive bar"

# BASE URL for the API
BASE_URL = 'https://api.6529.io/api'

# =======================
#    AUTHENTICATION
# =======================

def authenticate():
    # Load the private key from environment variables
    private_key = os.getenv('PRIVATE_KEY')
    if not private_key:
        print("Private key not found in environment variables.")
        return None

    # Create an account object from the private key
    try:
        account = Account.from_key(private_key)
    except ValueError as e:
        print("Invalid private key.")
        return None

    signer_address = account.address

    # Step 1: GET /auth/nonce
    nonce_url = f"{BASE_URL}/auth/nonce"
    params = {
        'signer_address': signer_address,
        'short_nonce': True
    }

    try:
        response = requests.get(nonce_url, params=params)
        response.raise_for_status()
        nonce_response = response.json()
    except requests.exceptions.RequestException as e:
        print("Error fetching nonce:", e)
        return None

    nonce = nonce_response.get('nonce')
    server_signature = nonce_response.get('server_signature')

    if not nonce or not server_signature:
        print("Nonce or server signature not found in the response.")
        return None

    # Step 2: Sign the nonce
    message = encode_defunct(text=nonce)
    signed_message = Account.sign_message(message, private_key=private_key)

    # Reconstruct the signature to include 'v', 'r', and 's'
    r = signed_message.r.to_bytes(32, byteorder='big')
    s = signed_message.s.to_bytes(32, byteorder='big')
    v = signed_message.v.to_bytes(1, byteorder='big')
    signature_bytes = r + s + v
    client_signature = '0x' + signature_bytes.hex()

    # Step 3: POST /auth/login
    login_url = f"{BASE_URL}/auth/login"
    params = {
        'signer_address': signer_address
    }
    data = {
        'client_signature': client_signature,
        'server_signature': server_signature,
    }

    try:
        response = requests.post(login_url, params=params, json=data)
        response.raise_for_status()
        login_response = response.json()
    except requests.exceptions.RequestException as e:
        print("Error during login:", e)
        return None

    # Handle the JWT token from the login response
    jwt_token = login_response.get('token')
    if jwt_token:
        return jwt_token
    else:
        print("Authentication failed. JWT token not received.")
        return None
  
def fetch_new_drops_for_wave(wave_id, jwt_token):
    """
    Fetches new drops for a given wave, checking only the latest 19 at a time,
    and continues fetching only if there are new drops. This version also
    collects replies (and creates stubs for missing parent drops) just as in the backfill.
    """
    print(f"Fetching new drops for wave {wave_id}...")

    # 1. Retrieve the highest serial_no we have locally for this wave
    local_max_serial = (
        db.session.query(db.func.max(Drop.serial_no))
        .filter(Drop.wave_id == wave_id)
        .scalar()
        or 0
    )
    print(f"Local max serial_no for wave {wave_id}: {local_max_serial}")

    # 2. Setup API request details
    drops_url = f"{BASE_URL}/drops"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/json",
    }
    limit = 19
    serial_no_less_than = None  # Start fetching the newest drops

    while True:
        # 3. Prepare parameters for the API call
        params = {
            "limit": limit,
            "wave_id": wave_id,
            "include_replies": True,  # Ensure replies are included
            "drop_type": "CHAT",
        }
        if serial_no_less_than is not None:
            params["serial_no_less_than"] = serial_no_less_than

        try:
            # 4. Make the API call
            drops_response = safe_get_drops(params.copy(), headers)
            if not drops_response:
                break

            # 5. Validate response structure
            if not isinstance(drops_response, list):
                print(f"Unexpected response format: {drops_response}")
                return

            drops_list = drops_response
            if not drops_list:
                print(f"No more drops to fetch for wave {wave_id}.")
                break

            new_drops_found = False
            serial_no_values = []  # Track serial numbers in this batch

            for drop_data in drops_list:
                drop_serial = drop_data.get("serial_no", 0)
                serial_no_values.append(drop_serial)

                # Skip processing if the drop is older or equal to our local max
                if drop_serial <= local_max_serial:
                    continue

                drop_id = drop_data.get("id")
                # Check if drop already exists to prevent duplicate insertion
                if Drop.query.get(drop_id):
                    print(f"Drop with id {drop_id} already exists. Skipping.")
                    continue

                # --- Begin processing reply logic ---
                # Check for replies and create stubs if needed
                reply_to = drop_data.get("reply_to", {})
                reply_to_id = reply_to.get("drop_id", None)

                # If there's a parent drop reference but we don't have it, create a stub
                if reply_to_id and not Drop.query.get(reply_to_id):
                    print(f"Creating stub for missing parent drop: {reply_to_id}")
                    stub_drop = Drop(
                        id=reply_to_id,
                        wave_id=drop_data["wave"]["id"],
                        author="Unknown",
                        content="(Stub parent record, to be updated later)",
                        serial_no=0,
                        created_at=None,
                    )
                    db.session.add(stub_drop)
                    try:
                        # Database operation
                        db.session.commit()
                    except Exception as e:
                        db.session.rollback()  # Rollback on failure
                        print(f"Database error: {e}")

                # Handle nested replies (replies to replies)
                nested_parent_drop = reply_to.get("drop", None)
                if nested_parent_drop:
                    nested_reply_to_id = nested_parent_drop.get("id")
                    # If reply_to_id is missing but nested exists, use nested ID
                    if reply_to_id is None and nested_reply_to_id:
                        reply_to_id = nested_reply_to_id
                    if nested_reply_to_id and not Drop.query.get(nested_reply_to_id):
                        print(f"Creating stub for missing nested parent drop: {nested_reply_to_id}")
                        stub_drop = Drop(
                            id=nested_reply_to_id,
                            wave_id=drop_data["wave"]["id"],
                            author="Unknown",
                            content="(Stub parent record, to be updated later)",
                            serial_no=0,
                            created_at=None,
                        )
                        db.session.add(stub_drop)
                # --- End processing reply logic ---

                # 6. Process basic drop fields
                content_parts = drop_data.get("parts", [])
                content = "\n".join(
                    (part.get("content") or "" for part in content_parts)
                ).strip()


                created_at_raw = drop_data.get("created_at")
                if isinstance(created_at_raw, int):
                    created_at = datetime.fromtimestamp(created_at_raw / 1000, tz=timezone.utc)
                elif isinstance(created_at_raw, str):
                    created_at = parser.isoparse(created_at_raw)
                else:
                    created_at = None

                author_handle = drop_data.get("author", {}).get("handle", "Unknown")

                # Handle author creation/updating if needed
                author_data = Author.query.get(author_handle)
                if author_data is None:
                    # If we don't have it in our local DB, fetch from API
                    user_info = fetch_user_by_handle(jwt_token, author_handle, wave_id)
                    if user_info:
                        pfp_url = user_info.get("pfp", None)
                        profile_url = f"https://6529.io/{author_handle}"
                        normalized_handle = user_info.get("normalised_handle", None)
                        cic = user_info.get("cic", None)
                        rep = user_info.get("rep", None)
                        level = user_info.get("level", None)
                        tdh = user_info.get("tdh", None)
                        display = user_info.get("display", None)
                        primary_wallet = user_info.get("primary_wallet", None)

                        author_data = Author(
                            id=wave_id,
                            handle=author_handle,
                            pfp_url=pfp_url,
                            profile_url=profile_url,
                            normalized_handle=normalized_handle,
                            cic=cic,
                            rep=rep,
                            level=level,
                            tdh=tdh,
                            display=display,
                            primary_wallet=primary_wallet,
                        )
                        db.session.add(author_data)
                        print(f"{author_handle} added")
                    else:
                        # Fallback to a minimal record
                        profile_url = f"https://6529.io/{author_handle}"
                        author_data = Author(
                            handle=author_handle, pfp_url=None, profile_url=profile_url
                        )
                        db.session.add(author_data)

                # 7. Create and add the new drop
                new_drop = Drop(
                    id=drop_id,
                    wave_id=wave_id,
                    author=author_handle,
                    content=content,
                    serial_no=drop_serial,
                    created_at=created_at,
                    reply_to_id=reply_to_id,  # Link to parent drop (if any)
                )
                db.session.add(new_drop)
                new_drops_found = True

            # 8. Commit all new drops & stubs to the DB
            if new_drops_found:
                try:
                    # Database operation
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()  # Rollback on failure
                    print(f"Database error: {e}")
                print(f"Added {len(drops_list)} drops for wave {wave_id}.")
            else:
                print(f"No new drops found in this batch for wave {wave_id}.")

            # 9. Update `serial_no_less_than` to fetch older drops
            serial_no_less_than = min(serial_no_values) - 1

            # If all serial numbers are less than or equal to local_max_serial, stop fetching
            if all(sn <= local_max_serial for sn in serial_no_values):
                print(f"All drops in this batch are older or equal to local max {local_max_serial}. Stopping fetch for wave {wave_id}.")
                break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching drops for wave {wave_id}: {e}")
            break


"""
===================================================
bot section
===================================================
"""
GENERAL_RESPONSE_THRESHOLD = 15
MAX_DROPS_FOR_SUMMARY = 50
def monitor_memes_chat(jwt_token=None):
    print("✅ monitor_memes_chat() ran at", datetime.now())
    print("Checking for new activity in the chosen wave...")
    jwt_token = jwt_token or authenticate()
    wave = Wave.query.filter_by(name=chosen_wave).first()
    if not wave:
        print(f"{wave} wave not found in the database!")
        return
    wave_id = wave.id
    print(wave_id)

    # Fetch the last processed serial number and accumulator for the wave
    wave_tracking = WaveTracking.query.filter_by(wave_id=wave_id).first() 
    if not wave_tracking:
        wave_tracking = WaveTracking(wave_id=wave_id, last_processed_serial_no=0, accumulated_new_drops=0)
        db.session.add(wave_tracking)
        db.session.commit()

    last_processed_serial = wave_tracking.last_processed_serial_no
    print(f"last_processed_serial {last_processed_serial}")

    # Fetch new drops (those with a serial_no greater than the last processed one)
    new_drops = Drop.query.filter(
        Drop.wave_id == wave_id, Drop.serial_no > last_processed_serial
    ).order_by(Drop.serial_no.asc()).all()
    print(f"new drops: {new_drops}")

    if not new_drops:
        print("No new drops to process.")
        return

    try:
        handle_new_drops(new_drops, wave_id, jwt_token)
    except Exception as e:
        print(f"Error handling new drops: {e}")
        return

    # Add the count of new drops to the accumulator
    wave_tracking.accumulated_new_drops += len(new_drops)
    print(f"Accumulated new drops: {wave_tracking.accumulated_new_drops}")

    # Only generate a general response if the accumulated count exceeds threshold
    if wave_tracking.accumulated_new_drops >= GENERAL_RESPONSE_THRESHOLD:
        wave_tracking.accumulated_new_drops = 0
        drops_text = "\n".join([f"{drop.author}: {drop.content}" for drop in new_drops])
        print("Generating a general response to the wave...")
        bot_response = generate_general_response(drops_text)
        if bot_response is None:
            raise Exception("General response generation failed.")
        post_general_response(wave_id, bot_response, jwt_token)
        # Reset the accumulator after a response is generated
        

    # Update last processed serial to the highest serial number from new_drops
    highest_serial_no = max(drop.serial_no for drop in new_drops)
    print(f"highest serial: {highest_serial_no}")
    wave_tracking.last_processed_serial_no = highest_serial_no

    try:
        # Database operation
        db.session.commit()
    except Exception as e:
        db.session.rollback()  # Rollback on failure
        print(f"Database error: {e}")
    print("Updated tracking after successfully processing drops.")

def handle_new_drops(new_drops, wave_id, jwt_token):
    bot_handle = "Gray"
    drops_text = "\n".join([f"{drop.author}: {drop.content}" for drop in new_drops])

    mentioned_drops = []
    for drop in new_drops:
        is_explicit_mention = "@" + bot_handle.lower() in (drop.content or "").lower()
        is_direct_reply = False

        if drop.reply_to_id:
            parent_drop = Drop.query.get(drop.reply_to_id)
            if parent_drop and parent_drop.author.lower() == bot_handle.lower():
                is_direct_reply = True

        if is_explicit_mention or is_direct_reply:
            mentioned_drops.append(drop)

    print(f"Mentions or direct replies found: {mentioned_drops}")

    for drop in mentioned_drops:
        print(f"Bot was mentioned or replied to in drop {drop.serial_no}. Generating a reply...")
        reply_to_mention(drop, jwt_token)

    # Process rep assignment commands in drops.
    # Look for drops containing an @mention, a number, and a quoted category.
    for drop in new_drops:
        username, amount, category = extract_rep_details(drop.content)
        if username and amount and category:
            print(f"Processing rep command from drop {drop.serial_no} for @{username}: {amount} rep for \"{category}\"")
            # Optionally, you could add extra conditions here if you want to distinguish rep commands from other drops.
            status, result = assign_rep(username, amount, category, jwt_token)
            print(f"Assigned rep: {status} {result}")

    # Optionally, if the total new drops reach a threshold, generate a general response.
    # if len(new_drops) >= GENERAL_RESPONSE_THRESHOLD:
    #     print("Generating a general response to the wave...")
    #     bot_response = generate_general_response(drops_text)
    #     if bot_response is None:
    #         raise Exception("General response generation failed.")
    #     post_general_response(wave_id, bot_response, jwt_token)

def reply_to_mention(drop, jwt_token):
    """
    Generates and posts a response directly to a drop mentioning the bot.
    Exceptions are allowed to propagate so that errors are caught in the caller,
    preventing the drop from being marked as processed if the response fails.
    """
    prompt = f"""
    You were mentioned in the following post:
    {drop.content}
    Respond directly to the comment in a firendly way keeping it brief (1-2 sentences).
    """

    # Generate the response from the chatbot
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a funny and friendly bar patron who loves internet memes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.60,
        max_tokens=120,
    )
    bot_response = response.choices[0].message.content.strip()

    # Prepare the payload for posting the reply
    payload = {
        "wave_id": drop.wave_id,
        "drop_type": "CHAT",
        "parts": [{"content": bot_response}],
        "reply_to": {"drop_id": drop.id, "drop_part_id": 0},  # Reply to the specific drop
    }

    url = f"{BASE_URL}/drops"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    # Post the reply; if this fails, an exception is raised
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    print(f"Successfully replied to mention in drop {drop.serial_no}.")

def get_last_50_drops(wave_id):
    """
    Fetches the last 50 drops from the given wave, ordered by creation time.
    """
    drops = Drop.query.filter(Drop.wave_id == wave_id).order_by(Drop.created_at.desc()).limit(50).all()
    drops.reverse()  # Ensure they're in chronological order
    return drops

def generate_general_response(drops_text):
    """
    Generates a general response based on the content of recent drops.
    Exceptions are not caught here so that errors propagate to the caller.
    """
    if not drops_text:
        print("No recent drops to analyze.")
        return None

    MAX_DROPS_TEXT_LENGTH = 4000  # Adjust as needed

    # Safely trim the drops text if it's too long
    if len(drops_text) > MAX_DROPS_TEXT_LENGTH:
        print(f"Trimming drops_text from {len(drops_text)} to {MAX_DROPS_TEXT_LENGTH} characters.")
        drops_text = drops_text[-MAX_DROPS_TEXT_LENGTH:]

    prompt = f"""
    Here's a recent conversation from the Memes-Chat wave:

    {drops_text}

    You're a dry dive bar patron who excels at contributing insights to the conversation. 
    Craft a brief insightful responses.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a funny and friendly dive bar patron."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.60,
        max_tokens=100,
    )
    bot_response = response.choices[0].message.content.strip()
    return bot_response

def fetch_user_by_handle(jwt_token, handle, wave_id=None):
    """
    Returns the profile JSON for @handle or None on error.
    """
    url = f"{BASE_URL}/profiles/{handle}"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/json"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch profile for @{handle}: {e}")
        return None

    
def post_general_response(wave_id, bot_response, jwt_token):
    """
    Posts the bot's response to the wave without replying to a specific drop.
    Exceptions are allowed to propagate so that errors are caught in the caller.
    """
    if not bot_response:
        raise Exception("No response generated. Skipping post.")

    payload = {
        "wave_id": wave_id,
        "drop_type": "CHAT",
        "parts": [{"content": bot_response}],
    }

    url = f"{BASE_URL}/drops"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    print("Successfully posted the bot's general response to the wave.")

def is_bot_mentioned(drop):
    """
    Checks if your bot is mentioned in the given drop.
    """
    bot_handle = "Gray"  # Replace with your bot's handle
    return bot_handle in drop.content

def respond_to_drop(drop, wave_id):
    """
    Generates and posts a bot response to the given drop in chosen wave.
    """
    print(f"Responding to drop {drop.id}...")
    try:
        # Construct the prompt for OpenAI = tell it what kind of response you want
        prompt = f"""
        A group chat participant who is responding to this drop:
        Drop Content: "{drop.content}"
        Create a ___ response:
        """

        # OpenAI API call
        response = OpenAI.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a chatbot that responds relevantly."},
                    {"role": "user", "content": prompt}],
            temperature=0.60,
        )

        # Get the response text
        bot_response = response["choices"][0]["message"]["content"]

        # Create the drop payload
        drop_payload = {
            "wave_id": wave_id,
            "drop_type": "CHAT",
            "parts": [{"content": bot_response}],
            "reply_to": {
                "drop_id": drop.id,
                "drop_part_id": 0  # Assuming the bot replies to the whole drop
            }
        }

        # Post the bot's response
        post_drop_to_api(drop_payload)

    except Exception as e:
        print(f"Error generating or posting response: {e}")

def post_drop_to_api(payload):
    """
    Posts a new drop to the Memes-Chat wave via the API.
    """
    try:
        url = f"{BASE_URL}/drops"
        headers = {
            "Authorization": f"Bearer {authenticate()}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Successfully posted bot response: {payload}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to post bot response: {e}")

def assign_rep(username, amount, category, jwt_token):
    url = f"https://api.6529.io/api/profiles/{username}/rep/rating"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "amount": amount,
        "category": category
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.status_code, response.json()

def extract_rep_details(text):
    #This function is for Articlate 
    # He wanted a situation where you could ask bot to proxy rep someone and they'd do it on your behalf  
    # Extract the first number as the rep amount
    amount_match = re.search(r'(\d+)', text)
    amount = int(amount_match.group(1)) if amount_match else None

    # Extract text within quotes for the category
    category_match = re.search(r'"([^"]+)"', text)
    category = category_match.group(1) if category_match else None

    # Extract the target username after an '@'
    user_match = re.search(r'@(\w+)', text)
    username = user_match.group(1) if user_match else None

    return username, amount, category

def accept_proxy(jwt_token):
    proxies_url = "https://api.6529.io/api/profiles/HANDLE_NAME/proxies/received"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    response = requests.get(proxies_url, headers=headers)
    proxies = response.json()
    print(proxies)
    if proxies:
        top_proxy = proxies[0]
        proxy_id = top_proxy.get("id")
        print(proxy_id)
        # Assuming at least one action is present
        action = top_proxy.get("actions", [{}])[0]
        action_id = action.get("id")
        print(action_id)
        acceptance_url = f"https://api.6529.io/api/proxies/{proxy_id}/actions/{action_id}/acceptance"
        # Include the required payload
        payload = {"action": "ACCEPT"}
        accept_response = requests.post(acceptance_url, headers=headers, json=payload)
        return accept_response.status_code, accept_response.json()
    else:
        print("No proxies received.")
        return None

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

scheduler = BackgroundScheduler()
scheduler.start()

def monitor_job():
    with app.app_context():
        monitor_memes_chat(authenticate())

scheduler.add_job(
    func=monitor_job,
    trigger=IntervalTrigger(minutes=2),
    id="monitor_memes_chat",
    replace_existing=True
)

def scheduled_accept_proxy():
    with app.app_context():
        jwt_token = authenticate()  # Get a fresh token
    if jwt_token:
        status, result = accept_proxy(jwt_token)
        print(f"Proxy acceptance status: {status}, result: {result}")
    else:
        print("Failed to authenticate.")

scheduler.add_job(
    func=scheduled_accept_proxy,
    trigger=IntervalTrigger(minutes=30),
    id='accept_proxy_job',
    name='Accept new proxy every 30 minutes',
    replace_existing=True
)

import atexit
atexit.register(lambda: scheduler.shutdown())

with app.app_context():
    db.create_all()
    # fetch_new_data()

def setup_chosen_wave(jwt_token, chosen_wave):
    print(f"Looking up wave: {chosen_wave}")
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/json"
    }
    params = {"name": chosen_wave}
    try:
        response = requests.get(f"{BASE_URL}/waves", headers=headers, params=params)
        response.raise_for_status()
        waves = response.json()
        if not waves:
            print("No matching wave found.")
            return None
        wave_data = waves[0]
        print(f"Found wave: {wave_data['name']}")

        existing_wave = Wave.query.get(wave_data["id"])
        if not existing_wave:
            wave = Wave(
                id=wave_data["id"],
                name=wave_data["name"],
                author=wave_data.get("author", {}).get("handle", "Unknown"),
                serial_no=wave_data["serial_no"]
            )
            db.session.add(wave)
            db.session.commit()
        else:
            wave = existing_wave
        return wave

    except Exception as e:
        print(f"Error fetching wave: {e}")
        return None
    
GAP_RETRY_LIMIT = 1000        # how many serials we probe downward before giving up

def safe_get_drops(params, headers):
    """
    Wrapper that handles 404-by-gap:
    keeps stepping serial_no_less_than downward until the call succeeds
    or we exhaust GAP_RETRY_LIMIT.
    """
    tries = 0
    while True:
        try:
            resp = requests.get(f"{BASE_URL}/drops", headers=headers, params=params)
            resp.raise_for_status()
            return resp.json()          # ✅ success
        except HTTPError as err:
            if err.response.status_code == 404 and "serial_no_less_than" in params:
                tries += 1
                if tries > GAP_RETRY_LIMIT:
                    print(f"Gap retry limit hit at serial {params['serial_no_less_than']}. Stopping page.")
                    return []           # give up on this wave/page
                # step down one serial and retry
                params["serial_no_less_than"] -= 1
                continue
            else:
                raise                   # re-raise non-gap errors


def fetch_all_drops_for_wave(jwt_token, wave_id):
    print(f"Fetching all drops for wave {wave_id}...")
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/json"
    }
    url = f"{BASE_URL}/drops"
    limit = 19
    serial_no = None

    while True:
        params = {
            "wave_id": wave_id,
            "limit": limit,
            "include_replies": "true",
            "drop_type": "CHAT"
        }
        if serial_no:
            params["serial_no_less_than"] = serial_no

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            drops = response.json()
            if not drops:
                break

            for drop_data in drops:
                drop_id = drop_data["id"]
                if Drop.query.get(drop_id):
                    continue

                created_at = drop_data.get("created_at")
                if isinstance(created_at, str):
                    created_at = parser.isoparse(created_at)

                content_parts = drop_data.get("parts", [])
                content = "\n".join((str(part.get("content") or "") for part in content_parts)).strip()


                author_data = drop_data.get("author", {})
                author_handle = author_data.get("handle", "Unknown")

                created_at_raw = drop_data.get("created_at")

                if isinstance(created_at_raw, int):
                    # It's epoch seconds (or milliseconds?), convert it
                    if created_at_raw > 1e12:  # If huge, divide by 1000
                        created_at_raw = created_at_raw / 1000
                    created_at = datetime.fromtimestamp(created_at_raw, tz=timezone.utc)
                elif isinstance(created_at_raw, str):
                    created_at = parser.isoparse(created_at_raw)
                else:
                    created_at = None

                drop = Drop(
                    id=drop_id,
                    wave_id=wave_id,
                    author=author_handle,
                    content=content,
                    serial_no=drop_data["serial_no"],
                    created_at=created_at,
                    reply_to_id=drop_data.get("reply_to_id")
                )
                db.session.add(drop)


            db.session.commit()
            serial_no = min(d["serial_no"] for d in drops)

        except Exception as e:
            db.session.rollback()
            print(f"Error fetching drops: {e}")
            break


def poll_new_drops_every_5_min(jwt_token, wave_id):
    def poll():
        with app.app_context():
            print("Polling for new drops...")
            jwt = authenticate()
            if jwt:
                fetch_new_drops_for_wave(wave_id, jwt)
    scheduler.add_job(poll, trigger=IntervalTrigger(minutes=5), id="poll_drops", replace_existing=True)

# MAIN FLOW for setting up the bot
with app.app_context():
    db.create_all()
    jwt_token = authenticate()
    if jwt_token:
        wave = setup_chosen_wave(jwt_token, chosen_wave)
        if wave:
            fetch_all_drops_for_wave(jwt_token, wave.id)
            poll_new_drops_every_5_min(jwt_token, wave.id)

# ---- keep the script alive so APScheduler jobs can run ----
if __name__ == "__main__":
    print("✅ Gray bot is up. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down…")
