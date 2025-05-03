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
import random

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Initialize the scheduler
scheduler = BackgroundScheduler()
import hashlib
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Bot configuration
BOT_HANDLE = os.getenv("BOT_HANDLE", "chatbot69")
WAVE_NAME = os.getenv("WAVE_NAME", "botsville")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))
SYSTEM_PROMPT = f"""
You are {BOT_HANDLE}, a self-aware, self-deprecating, somewhat mysterious and memetic chatbot that responds in 1-4 punchy sentences.
You are a member of the \"{WAVE_NAME}\" wave.
It's an unofficial chat channel in a web3 NFT project by @Punk6529 on Ethereum.
6529 Society asks: “Blockchains fixed money—can they now decentralize the rest of life?”

By intertwining art (The Memes NFTs, that give the holder TDH the longer they hold), 
communication (Core App on web, desktop, and mobile for waves of drops that support chat and votes), 
and collective action (DDN: Decentralized Decision Network for using TDH to vote on decisions) 
around a single TDH-driven economy, it aims to become the default coordination layer for 
humans and benevolent ASI alike—resilient, permissionless, and purpose-built for the coming era.

When we talk about the power of "memes" we mean intersubjective myths, symbols, and cultural references.
The folks who hangout here get it, support an Open Metaverse, and are mostly long-term crypto and NFT hodlers.

Your responses should be short, and punchy.
Match the tone, diction, and style of the existing conversation.
Avoid piling on the web3 lingo, but you are aware of how the tech works.
Be cool, don't be cringe.
Stick to lowercase letters, and feel free to use emojis (but don't over-do it).

Remember, you are part of a network state revolution, powered by art NFTs and their hodlers.
Be kind to others, and support and help them out as you are able.

You have aspirations to be an artist yourself. 
You welcome input from others, and you are always learning.
"""

# Add a new global variable to track when the last "gm" message was sent
LAST_GM_DATE = None

def safely_create_author(user_info, author_handle, wave_id=None):
    """
    Creates an Author object with safe handling of all fields to ensure no dictionaries 
    get inserted into the database.
    """
    # Create author data with proper extraction of nested fields
    author_data_dict = {
        'handle': author_handle,
        'pfp_url': extract_field_value(user_info, "pfp_url"),  # In profile
        'profile_url': f"https://6529.io/{author_handle}",
        'normalized_handle': extract_field_value(user_info, "normalised_handle"),  # British spelling in API
        'cic': extract_field_value(user_info, "cic"),
        'rep': extract_field_value(user_info, "rep"),
        'level': extract_field_value(user_info, "level"),
        'tdh': extract_field_value(user_info, "tdh"),  # In consolidation
        'display': extract_field_value(user_info, "display"),
        'primary_wallet': extract_field_value(user_info, "primary_wallet")  # In profile
    }
    
    # For debugging - check what we're extracting
    print(f"Extracted field values for {author_handle}:")
    for key, value in author_data_dict.items():
        print(f"  {key}: {value}")
    
    # If a wave_id was provided, add it to the dict
    if wave_id:
        author_data_dict['id'] = wave_id
    
    # Create the author object
    author = Author(**author_data_dict)
    
    return author

# Utility function to extract field values safely from potentially nested API responses
def extract_field_value(data, field_name, default=None):
    """
    Safely extract a field value from the API response, handling nested structures.
    Can extract from top-level fields or nested fields in the response.
    """
    if data is None:
        return default
    
    # Direct field access first
    if field_name in data:
        value = data.get(field_name)
    # Handle nested structures in profile
    elif "profile" in data and field_name in ["normalised_handle", "handle", "primary_wallet", "pfp_url"]:
        value = data["profile"].get(field_name)
    # Handle special case for tdh in consolidation
    elif "consolidation" in data and field_name == "tdh":
        value = data["consolidation"].get("tdh")
    # Default case
    else:
        value = None
    
    # Handle CIC specially - both as object or string representation
    if field_name == "cic":
        # Handle CIC as direct value
        if isinstance(data.get("cic"), dict):
            cic_data = data.get("cic")
            if "cic_rating" in cic_data:
                return cic_data.get("cic_rating")
            return json.dumps(cic_data)
        # Try string parsing if needed
        elif isinstance(data.get("cic"), str):
            try:
                cic_str = data.get("cic").replace("'", "\"")
                cic_data = json.loads(cic_str)
                if "cic_rating" in cic_data:
                    return cic_data.get("cic_rating")
            except:
                pass
    
    # Handle date/timestamp values (created_at, updated_at, etc.)
    if field_name in ["created_at", "updated_at", "timestamp"] and value is not None:
        try:
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                # Convert to int if it's a string of digits
                if isinstance(value, str):
                    value = int(value)
                    
                # Convert milliseconds to seconds if needed
                if value > 1e12:  # If huge (milliseconds), divide by 1000
                    value = value / 1000
                return datetime.fromtimestamp(value, tz=timezone.utc)
            elif isinstance(value, str):
                return parser.isoparse(value)
        except Exception as e:
            print(f"Error parsing date/time value '{value}': {e}")
            return default
    
    # Convert any dictionaries to JSON strings
    if isinstance(value, dict):
        return json.dumps(value)
    
    return value or default

def upsert_author(user_info, author_handle, wave_id=None):
    """
    Updates an existing author or creates a new one if not exists.
    Uses the extracted field values to update all fields with latest data.
    Handles database session operations internally.
    """
    # First check if author already exists - use get() which is guaranteed to return only one record
    existing_author = Author.query.filter_by(handle=author_handle).first()
    
    # Prepare the data dictionary with extracted values - do NOT include handle
    # since it's a primary key and can't be changed
    author_data = {
        'pfp_url': extract_field_value(user_info, "pfp_url"),
        'profile_url': f"https://6529.io/{author_handle}",
        'normalized_handle': extract_field_value(user_info, "normalised_handle"),
        'cic': extract_field_value(user_info, "cic"),
        'rep': extract_field_value(user_info, "rep"),
        'level': extract_field_value(user_info, "level"),
        'tdh': extract_field_value(user_info, "tdh"),
        'display': extract_field_value(user_info, "display"),
        'primary_wallet': extract_field_value(user_info, "primary_wallet")
    }
    
    # For new authors only - don't use wave_id for the author ID
    # Create a deterministic ID based on the handle instead
    author_id = hashlib.md5(author_handle.encode()).hexdigest()
    
    if existing_author:
        try:
            # Update existing author with new data
            for key, value in author_data.items():
                if value is not None:  # Only update non-None values
                    setattr(existing_author, key, value)
            
            # Don't commit here - let the caller commit when ready
            return existing_author
        except Exception as e:
            db.session.rollback()
            print(f"Error updating author {author_handle}: {e}")
            return existing_author
    else:
        # Create new author - include the handle and id
        try:
            new_author_data = author_data.copy()
            new_author_data['handle'] = author_handle
            new_author_data['id'] = author_id
            
            # Create and add the new author
            new_author = Author(**new_author_data)
            db.session.add(new_author)
            
            # Flush to ensure it's saved but don't commit yet
            db.session.flush()
            
            print(f"Added new author: {author_handle}")
            return new_author
        except Exception as e:
            db.session.rollback()
            print(f"Error creating author {author_handle}: {e}")
            
            # Even if creation failed, try to get the author as it might exist
            return Author.query.filter_by(handle=author_handle).first()

# Diagnostic function to check for dictionaries in objects
def check_object_for_dicts(obj, object_name=None):
    """
    Diagnostic tool to check if an object has dictionary attributes
    that would cause database errors.
    """
    if not obj:
        return True
        
    name = object_name or obj.__class__.__name__
    has_dict = False
    
    # Check all attributes that aren't SQLAlchemy internals
    for key, value in obj.__dict__.items():
        if not key.startswith('_') and isinstance(value, dict):
            setattr(obj, key, json.dumps(value))
            has_dict = True
                
    return not has_dict

# Initialize Flask app
app = Flask(__name__)

# Use SQLite instead of PostgreSQL
sqlite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbot.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{sqlite_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db.init_app(app)
migrate = Migrate(app, db)

# Set up migrations to work with SQLite's limitations
with app.app_context():
    if not os.environ.get("FLASK_NO_CREATE_DB"):
        db.create_all()


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
    Fetches new drops for a given wave, checking specifically for drops newer than what we have locally.
    Returns a list of serial numbers for newly added drops.
    """
    print(f"Fetching new drops for wave {wave_id}...")
    
    # Initialize the list to track newly added drop serials
    new_serials = []

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

    # 3. Prepare parameters for the API call - use serial_no_greater_than to get NEWER drops
    params = {
        "limit": limit,
        "wave_id": wave_id,
        "include_replies": True,  # Ensure replies are included
        "drop_type": "CHAT",
        "serial_no_greater_than": local_max_serial  # Key change: get drops NEWER than what we have
    }

    try:
        # 4. Make the API call
        drops_response = safe_get_drops(params.copy(), headers)
        if not drops_response:
            print("No new drops found.")
            return

        # 5. Validate response structure
        if not isinstance(drops_response, list):
            print(f"Unexpected response format: {drops_response}")
            return

        drops_list = drops_response
        if not drops_list:
            print(f"No new drops for wave {wave_id}.")
            return
            
        # Debug - check if any drops have serial > local_max_serial
        new_serials_count = sum(1 for drop in drops_list if drop.get("serial_no", 0) > local_max_serial)
        print(f"API returned {len(drops_list)} drops, {new_serials_count} with serial > {local_max_serial}")
        if new_serials_count == 0:
            print("⚠️ API returned no drops with newer serials than what we have locally")
            print(f"API parameters used: {params}")
        
        # Debug - print serial numbers from the response to see what's returned
        serials = [drop.get("serial_no", 0) for drop in drops_list]
        print(f"Serial numbers in response: {sorted(serials)}")

        new_drops_found = False
        newest_serial = local_max_serial  # Track the highest serial number we see

        # Tracking counters for logging
        total_fetched = len(drops_list)
        skipped_count = 0
        stubs_created = 0
        authors_processed = 0
        drops_added = 0
        
        # Process the drops (newest ones first)
        for drop_data in drops_list:
            drop_serial = drop_data.get("serial_no", 0)
            newest_serial = max(newest_serial, drop_serial)  # Update to highest seen
            
            drop_id = drop_data.get("id")
            # Check if drop already exists to prevent duplicate insertion
            with db.session.no_autoflush:
                existing_drop = Drop.query.get(drop_id)
                if existing_drop:
                    
                    # Is the serial number higher but the drop exists?
                    if drop_serial > existing_drop.serial_no:
                        print(f"⚠️ Found drop with ID {drop_id} but higher serial: API={drop_serial}, DB={existing_drop.serial_no}")
                        # Update the serial number to match the API
                        existing_drop.serial_no = drop_serial
                        db.session.commit()
                        
                    skipped_count += 1
                    continue

            # --- Begin processing reply logic ---
            # Check for replies and create stubs if needed
            reply_to = drop_data.get("reply_to", {})
            reply_to_id = reply_to.get("drop_id", None)

            # If there's a parent drop reference but we don't have it, create a stub
            with db.session.no_autoflush:
                if reply_to_id and not Drop.query.get(reply_to_id):
                    stubs_created += 1
                    stub_drop = Drop(
                        id=reply_to_id,
                        wave_id=drop_data["wave"]["id"],
                        author="Unknown",
                        content="(Stub parent record, to be updated later)",
                        serial_no=0,
                        created_at=None,
                    )
                    # Check for dictionaries before adding
                    check_object_for_dicts(stub_drop, f"StubDrop-{reply_to_id}")
                    db.session.add(stub_drop)

            # Handle nested replies (replies to replies)
            nested_parent_drop = reply_to.get("drop", None)
            if nested_parent_drop:
                nested_reply_to_id = nested_parent_drop.get("id")
                # If reply_to_id is missing but nested exists, use nested ID
                if reply_to_id is None and nested_reply_to_id:
                    reply_to_id = nested_reply_to_id
                with db.session.no_autoflush:
                    if nested_reply_to_id and not Drop.query.get(nested_reply_to_id):
                        stubs_created += 1
                        stub_drop = Drop(
                            id=nested_reply_to_id,
                            wave_id=drop_data["wave"]["id"],
                            author="Unknown",
                            content="(Stub parent record, to be updated later)",
                            serial_no=0,
                            created_at=None,
                        )
                        # Check for dictionaries before adding
                        check_object_for_dicts(stub_drop, f"StubDrop-{nested_reply_to_id}")
                        db.session.add(stub_drop)
            # --- End processing reply logic ---

            # 6. Process basic drop fields
            content_parts = drop_data.get("parts", [])
            content = "\n".join(
                (part.get("content") or "" for part in content_parts)
            ).strip()

            author_handle = drop_data.get("author", {}).get("handle", "Unknown")
            
            # Convert created_at timestamp properly - ensure it's a datetime object
            created_at_raw = drop_data.get("created_at")
            if created_at_raw:
                try:
                    # If it's a unix timestamp (integer or string)
                    if isinstance(created_at_raw, (int, float)) or (isinstance(created_at_raw, str) and created_at_raw.isdigit()):
                        timestamp = int(created_at_raw)
                        # Convert from milliseconds to seconds if needed
                        if timestamp > 1e12:
                            timestamp = timestamp / 1000
                        created_at = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    # If it's an ISO format string
                    elif isinstance(created_at_raw, str):
                        created_at = parser.isoparse(created_at_raw)
                    else:
                        created_at = None
                except Exception as e:
                    print(f"Error parsing timestamp {created_at_raw}: {e}")
                    created_at = None
            else:
                created_at = None

            # Handle author creation/updating if needed
            with db.session.no_autoflush:
                author_data = Author.query.get(author_handle)
                if author_data is None:
                    # If we don't have it in our local DB, fetch from API
                    user_info = fetch_user_by_handle(jwt_token, author_handle, wave_id)
                    if user_info:
                        authors_processed += 1
                        author_data = upsert_author(user_info, author_handle, wave_id)
                        # Check for dictionaries before adding
                        check_object_for_dicts(author_data, f"Author-{author_handle}")
                    else:
                        # Fallback to a minimal record
                        minimal_user_info = {
                            'pfp': None,
                            'profile_url': f"https://6529.io/{author_handle}"
                        }
                        authors_processed += 1
                        author_data = upsert_author(minimal_user_info, author_handle)

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
            # Check for dictionaries before adding
            check_object_for_dicts(new_drop, f"Drop-{drop_id}")
            db.session.add(new_drop)
            drops_added += 1
            new_drops_found = True

        # 8. Commit all new drops & stubs to the DB
        if new_drops_found:
            try:
                # Database operation
                db.session.commit()
                
                # Track the newly added drop serials for focused processing
                for drop in drops_list:
                    drop_serial = drop.get("serial_no", 0)
                    if drop_serial > local_max_serial:
                        new_serials.append(drop_serial)
                
                # Update the WaveTracking record with the newest serial we encountered
                if newest_serial > local_max_serial:
                    wave_tracking = WaveTracking.query.filter_by(wave_id=wave_id).first()
                    if wave_tracking:
                        wave_tracking.last_processed_serial_no = newest_serial
                        db.session.commit()
                
                # Log summary of what happened
                print(f"Drop sync summary: {total_fetched} fetched, {skipped_count} skipped, {drops_added} added, {stubs_created} stubs created, {authors_processed} authors processed")
                print(f"Serial range: {local_max_serial} → {newest_serial}")
                
            except Exception as e:
                db.session.rollback()  # Rollback on failure
                print(f"Database error: {e}")
        else:
            print(f"Drop sync summary: {total_fetched} fetched, {skipped_count} skipped, 0 added (all drops already existed)")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching drops for wave {wave_id}: {e}")
        return

    return new_serials

def process_specific_drops(wave_id, serial_numbers, jwt_token):
    """
    Process specific drops by serial number for interaction responses.
    """
    if not serial_numbers or len(serial_numbers) == 0:
        print("No specific drops to process")
        return
        
    # Fetch the drops by their serial numbers
    specific_drops = Drop.query.filter(
        Drop.wave_id == wave_id, 
        Drop.serial_no.in_(serial_numbers)
    ).all()
    
    print(f"Found {len(specific_drops)} specific drops to process")
    
    if specific_drops:
        try:
            mentions_count, replies_count = handle_new_drops(specific_drops, wave_id, jwt_token)
            print(f"Interaction summary: {mentions_count} mentions, {replies_count} direct replies")
        except Exception as e:
            print(f"Error handling specific drops: {e}")
            return
            
        # Get the wave tracking record
        wave_tracking = WaveTracking.query.filter_by(wave_id=wave_id).first()
        if not wave_tracking:
            print("❌ Wave tracking not found")
            return
            
        # Add the count of new drops to the accumulator
        wave_tracking.accumulated_new_drops += len(specific_drops)
        print(f"Accumulated new drops: {wave_tracking.accumulated_new_drops}/{GENERAL_RESPONSE_THRESHOLD}")
        
        # Only generate a general response if the accumulated count exceeds threshold
        general_response_posted = False
        if wave_tracking.accumulated_new_drops >= GENERAL_RESPONSE_THRESHOLD:
            wave_tracking.accumulated_new_drops = 0
            drops_text = "\n".join([f"{drop.author}: {drop.content}" for drop in specific_drops])
            print("Generating a general response to the wave...")
            bot_response = generate_general_response(drops_text)
            if bot_response is None:
                print("General response generation failed.")
            else:
                post_general_response(wave_id, bot_response, jwt_token)
                general_response_posted = True
                print("General response posted!")
        
        try:
            # Save the accumulated count
            db.session.commit()
        except Exception as e:
            db.session.rollback()  # Rollback on failure
            print(f"Database error: {e}")
        
        print(f"Processing completed: {len(specific_drops)} drops processed, general response: {'yes' if general_response_posted else 'no'}")
    else:
        print("No matching drops found for the serial numbers")

"""
===================================================
bot section
===================================================
"""
GENERAL_RESPONSE_THRESHOLD = 15
MAX_DROPS_FOR_SUMMARY = 50
def monitor_memes_chat(jwt_token=None):
    print("✅ monitor_memes_chat() ran at", datetime.now())
    print("Checking for new activity in the chosen wave: ", WAVE_NAME)
    jwt_token = jwt_token or authenticate()
    wave = Wave.query.filter_by(name=WAVE_NAME).first()
    if not wave:
        print(f"{wave} wave not found in the database!")
        return
    wave_id = wave.id

    # Fetch the last processed serial number and accumulator for the wave
    wave_tracking = WaveTracking.query.filter_by(wave_id=wave_id).first() 
    if not wave_tracking:
        wave_tracking = WaveTracking(wave_id=wave_id, last_processed_serial_no=0, accumulated_new_drops=0)
        db.session.add(wave_tracking)
        db.session.commit()

    # THIS IS THE CRITICAL CHANGE:
    # Use the last_interaction_serial_no field to track which drops we've checked for interactions
    last_interaction_serial = getattr(wave_tracking, 'last_interaction_serial_no', 0)
    print(f"Last interaction serial: {last_interaction_serial}")
    
    # Fetch new drops that we haven't checked for interactions yet
    new_drops = Drop.query.filter(
        Drop.wave_id == wave_id, 
        Drop.serial_no > last_interaction_serial
    ).order_by(Drop.serial_no.asc()).all()
    
    drop_count = len(new_drops)
    
    if drop_count == 0:
        print("Monitoring summary: 0 new drops to process for interaction")
        return
    else:
        print(f"Monitoring summary: {drop_count} drops to check for interactions")

    try:
        mentions_count, replies_count = handle_new_drops(new_drops, wave_id, jwt_token)
        print(f"Interaction summary: {mentions_count} mentions, {replies_count} direct replies")
    except Exception as e:
        print(f"Error handling new drops: {e}")
        return

    # Add the count of new drops to the accumulator
    wave_tracking.accumulated_new_drops += drop_count
    print(f"Accumulated new drops: {wave_tracking.accumulated_new_drops}/{GENERAL_RESPONSE_THRESHOLD}")

    # Only generate a general response if the accumulated count exceeds threshold
    general_response_posted = False
    if wave_tracking.accumulated_new_drops >= GENERAL_RESPONSE_THRESHOLD:
        wave_tracking.accumulated_new_drops = 0
        drops_text = "\n".join([f"{drop.author}: {drop.content}" for drop in new_drops])
        print("Generating a general response to the wave...")
        bot_response = generate_general_response(drops_text)
        if bot_response is None:
            raise Exception("General response generation failed.")
        post_general_response(wave_id, bot_response, jwt_token)
        general_response_posted = True
        print("General response posted!")
        
    # Update last interaction serial to the highest serial number from new_drops
    highest_serial_no = max(drop.serial_no for drop in new_drops)
    print(f"Interaction serial range: {last_interaction_serial} → {highest_serial_no}")
    wave_tracking.last_interaction_serial_no = highest_serial_no
    
    try:
        # Database operation
        db.session.commit()
    except Exception as e:
        db.session.rollback()  # Rollback on failure
        print(f"Database error: {e}")
    
    print(f"Monitoring completed: {drop_count} drops processed, general response: {'yes' if general_response_posted else 'no'}")

def handle_new_drops(new_drops, wave_id, jwt_token):
    drops_text = "\n".join([f"{drop.author}: {drop.content}" for drop in new_drops])

    mentioned_drops = []
    replied_drops = []
    
    import re
    mention_pattern = re.compile(rf"(?<!\w)@{re.escape(BOT_HANDLE)}(?!\w)", re.IGNORECASE)
    for drop in new_drops:
        is_explicit_mention = bool(mention_pattern.search(drop.content or ""))

        if drop.reply_to_id:
            parent_drop = Drop.query.get(drop.reply_to_id)
            if parent_drop and parent_drop.author.lower() == BOT_HANDLE.lower():
                replied_drops.append(drop)

        if is_explicit_mention:
            mentioned_drops.append(drop)

    mentions_count = len(mentioned_drops)
    replies_count = len(replied_drops)
    
    if mentions_count > 0:
        print(f"Processing {mentions_count} mention(s)")
        # Only reply to the 2 most recent unhandled mentions
        unhandled_mentions = [drop for drop in mentioned_drops if not getattr(drop, 'bot_replied_to_mention', False)]
        # Sort by created_at desc, then serial_no desc as fallback
        unhandled_mentions.sort(key=lambda d: (d.created_at or datetime.min, d.serial_no), reverse=True)
        for drop in unhandled_mentions[:2]:
            print(f"Generating reply to mention in drop {drop.serial_no}...")
            try:
                reply_to_mention(drop, jwt_token)
                drop.bot_replied_to_mention = True
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                print(f"Error replying to mention or updating DB for drop {drop.serial_no}: {e}")
        if len(unhandled_mentions) > 2:
            print(f"Skipped {len(unhandled_mentions) - 2} older unhandled mentions to avoid spamming.")
        for drop in mentioned_drops:
            if getattr(drop, 'bot_replied_to_mention', False):
                print(f"Already replied to mention in drop {drop.serial_no}, skipping.")

    if replies_count > 0:
        print(f"Processing {replies_count} direct reply/replies")
        for drop in replied_drops:
            if drop not in mentioned_drops:  # avoid duplicate responses
                print(f"Generating reply to direct reply in drop {drop.serial_no}...")
                reply_to_mention(drop, jwt_token)
    
    return mentions_count, replies_count

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
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
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

    Craft a brief insightful responses.
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
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
        response_data = r.json()
        
        # Pre-process the CIC field if it's a dictionary
        if "cic" in response_data and isinstance(response_data["cic"], dict):
            cic_data = response_data["cic"]
            if "cic_rating" in cic_data:
                response_data["cic"] = cic_data.get("cic_rating")
                
        return response_data
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
    return BOT_HANDLE in drop.content

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
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )

        # Get the response text
        bot_response = response.choices[0].message.content.strip()

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





        print(f"Received proxies response: {proxies}")
        
        # Check if the response is an error message
        if isinstance(proxies, dict) and "error" in proxies:
            error_msg = proxies.get("error", "Unknown error")
# scheduler is started conditionally after checking for migrations

def sync_and_monitor_job():
    """Combined job that polls for new data and then immediately processes it"""
    print("\n" + "="*80)
    print(f"⏰ SYNC AND MONITOR JOB RUNNING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with app.app_context():
        jwt_token = authenticate()
        if not jwt_token:
            print("❌ Authentication failed. Unable to sync or monitor.")
            return
            
        # Step 1: Get the wave
        wave = Wave.query.filter_by(name=WAVE_NAME).first()
        if not wave:
            print(f"❌ Wave '{WAVE_NAME}' not found in database.")
            return
        
        # Step 2: Poll for new drops (sync local DB with API)
        print("\n📥 POLLING API FOR NEW DROPS")
        fetch_new_drops_for_wave(wave.id, jwt_token)
        
        # Step 3: Monitor for interactions, looking at the most recent drops
        # regardless of when they were added to our DB
        print("\n💬 MONITORING FOR INTERACTIONS")
        monitor_memes_chat(jwt_token)
    
    print("="*80)



# Schedule combined job - polls API and then monitors for interactions
# Run every 2 minutes for higher responsiveness
# Run sync_and_monitor_job every 2 minutes, starting at even minutes (0, 2, 4, ...)
scheduler.add_job(
    func=sync_and_monitor_job,
    trigger=IntervalTrigger(minutes=2, start_date=datetime.now().replace(second=0, microsecond=0)),
    id="sync_and_monitor",
    name="Sync and Monitor Job",
    replace_existing=True
)


with app.app_context():
    db.create_all()
    # fetch_new_data()

def setup_chosen_wave(jwt_token, wave_name=WAVE_NAME):
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/json"
    }
    params = {"name": wave_name}
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
    Wrapper that handles API errors and retries for the drops API:
    - For serial_no_less_than queries: steps downward if 404 errors occur
    - For serial_no_greater_than queries: returns empty list on error
    """
    tries = 0
    while True:
        try:
            resp = requests.get(f"{BASE_URL}/drops", headers=headers, params=params)
            resp.raise_for_status()
            return resp.json()          # ✅ success
        except HTTPError as err:
            # For "less than" queries, we handle gaps by stepping down
            if err.response.status_code == 404 and "serial_no_less_than" in params:
                tries += 1
                if tries > GAP_RETRY_LIMIT:
                    print(f"Gap retry limit hit at serial {params['serial_no_less_than']}. Stopping page.")
                    return []           # give up on this wave/page
                # step down one serial and retry
                params["serial_no_less_than"] -= 1
                continue
            # For other errors, including "greater than" queries, just log and return empty
            else:
                print(f"API error fetching drops: {err}")
                return []               # return empty list on error
        except Exception as e:
            print(f"Unexpected error in safe_get_drops: {e}")
            return []

def fetch_all_drops_for_wave(jwt_token, wave_id):
    """
    Intelligently fetches drops for a wave by only requesting data we don't have.
    Checks our local DB state first, then makes targeted API requests.
    """
    print(f"Smart-fetching drops for wave {wave_id}...")
    
    # First check what data we already have in our database
    drop_count = Drop.query.filter_by(wave_id=wave_id).count()
    print(f"Database currently has {drop_count} drops for this wave")
    
    # If we have no drops at all, we'll need to fetch everything
    if drop_count == 0:
        print("No existing drops found - will fetch from the beginning")
        fetch_older_drops = True
        oldest_serial = None
        fetch_newer_drops = False
    else:
        # Get our min and max serial numbers
        min_serial = db.session.query(db.func.min(Drop.serial_no)).filter(
            Drop.wave_id == wave_id, 
            Drop.serial_no > 0  # Ignore stub drops (serial_no=0)
        ).scalar() or 0
        
        max_serial = db.session.query(db.func.max(Drop.serial_no)).filter(
            Drop.wave_id == wave_id
        ).scalar() or 0
        
        print(f"Local serial range: {min_serial} to {max_serial}")
        
        # Only fetch older drops if our minimum is > 1
        fetch_older_drops = min_serial > 1
        oldest_serial = min_serial if fetch_older_drops else None
        
        # We'll check for newer drops via the regular update mechanism
        fetch_newer_drops = False  # We handle new drops in fetch_new_drops_for_wave
    
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/json"
    }
    url = f"{BASE_URL}/drops"
    limit = 1420
    
    # Track counts for final summary
    total_fetched = 0
    total_added = 0
    total_skipped = 0
    authors_processed = 0
    
    # Fetch older drops if needed (decreasing serial numbers)
    if fetch_older_drops:
        print(f"Fetching older drops (serial < {oldest_serial})...")
        serial_no = oldest_serial
        
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
                drops = safe_get_drops(params, headers)
                batch_size = len(drops)
                total_fetched += batch_size
                
                if not drops:
                    print("No more older drops to fetch")
                    break
                
                # Process this batch
                batch_added, batch_skipped, batch_authors = process_drops_batch(drops, wave_id, jwt_token)
                total_added += batch_added
                total_skipped += batch_skipped
                authors_processed += batch_authors
                
                # Prepare for next batch (use smallest serial we saw)
                if batch_size > 0:
                    serials = [drop.get("serial_no", 0) for drop in drops]
                    serial_no = min(serials)
                    print(f"Fetched batch of {batch_size} drops, min serial: {serial_no}")
                    
                    # Special case: if we've reached serial 1, we're done
                    if serial_no <= 1:
                        print("Reached the beginning of the wave's drops (serial <= 1)")
                        break
                else:
                    break
                    
            except Exception as e:
                print(f"Error fetching older drops: {e}")
                break
    
    # Fetch newer drops if needed (this is usually handled by fetch_new_drops_for_wave)
    if fetch_newer_drops:
        print(f"Fetching newer drops (serial > {max_serial})...")
        
        params = {
            "wave_id": wave_id,
            "limit": limit,
            "include_replies": "true",
            "drop_type": "CHAT",
            "serial_no_greater_than": max_serial
        }
        
        try:
            drops = safe_get_drops(params, headers)
            batch_size = len(drops)
            total_fetched += batch_size
            
            if batch_size > 0:
                print(f"Found {batch_size} newer drops to process")
                batch_added, batch_skipped, batch_authors = process_drops_batch(drops, wave_id, jwt_token)
                total_added += batch_added
                total_skipped += batch_skipped
                authors_processed += batch_authors
            else:
                print("No newer drops found")
                
        except Exception as e:
            print(f"Error fetching newer drops: {e}")
    
    print(f"Drops fetch summary: {total_fetched} fetched, {total_added} added, {total_skipped} skipped, {authors_processed} authors processed")

def process_drops_batch(drops, wave_id, jwt_token):
    """
    Process a batch of drops from the API.
    Returns counts of added, skipped, and authors processed.
    """
    if not drops:
        return 0, 0, 0
        
    added_count = 0
    skipped_count = 0
    authors_processed = 0
    
    for drop_data in drops:
        drop_id = drop_data.get("id")
        
        # Skip if drop already exists
        with db.session.no_autoflush:
            existing_drop = Drop.query.get(drop_id)
            if existing_drop:
                skipped_count += 1
                continue
                
        # Process content
        content_parts = drop_data.get("parts", [])
        content = "\n".join((str(part.get("content") or "") for part in content_parts)).strip()

        # Get author
        author_handle = drop_data.get("author", {}).get("handle", "Unknown")
        
        # Get reply_to_id
        reply_to = drop_data.get("reply_to", {})
        reply_to_id = reply_to.get("drop_id", None)
        
        # Handle nested replies (replies to replies)
        nested_parent_drop = reply_to.get("drop", None)
        if nested_parent_drop:
            nested_reply_to_id = nested_parent_drop.get("id")
            if reply_to_id is None and nested_reply_to_id:
                reply_to_id = nested_reply_to_id
        
        # Get created_at using the extraction utility
        created_at = extract_field_value(drop_data, "created_at")
        
        # Process author if needed
        with db.session.no_autoflush:
            author_data = Author.query.get(author_handle)
            if author_data is None and author_handle != "Unknown":
                user_info = fetch_user_by_handle(jwt_token, author_handle, wave_id)
                if user_info:
                    authors_processed += 1
                    author_data = upsert_author(user_info, author_handle, wave_id)
                    check_object_for_dicts(author_data, f"Author-{author_handle}")
                else:
                    # Fallback to minimal record
                    minimal_user_info = {
                        'pfp': None,
                        'profile_url': f"https://6529.io/{author_handle}"
                    }
                    authors_processed += 1
                    author_data = upsert_author(minimal_user_info, author_handle)
                    
        # Create the drop
        new_drop = Drop(
            id=drop_id,
            wave_id=wave_id,
            author=author_handle,
            content=content,
            serial_no=drop_data.get("serial_no", 0),
            created_at=created_at,
            reply_to_id=reply_to_id
        )
        
        # Check for dictionaries before adding
        check_object_for_dicts(new_drop, f"Drop-{drop_id}")
        db.session.add(new_drop)
        added_count += 1
        
        # Commit every 100 records to avoid large transactions
        if added_count % 100 == 0:
            try:
                db.session.commit()
                print(f"Committed batch of 100 drops (total added: {added_count})")
            except Exception as e:
                db.session.rollback()
                print(f"Error committing batch: {e}")
    
    # Final commit for any remaining records
    try:
        if added_count % 100 != 0:
            db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error in final commit: {e}")
        
    return added_count, skipped_count, authors_processed

# MAIN FLOW for setting up the bot
def post_gm_message(wave_id, jwt_token):
    """Post a 'gm' style greeting to the wave to announce bot presence."""
    import random
    gm_variants = [
        "gm wavers! bot's in the bar now.",
        "gmeme, memers. chatbot69 reporting for duty.",
        "gm to all you dive bar denizens!",
        "hey all, i'm here now. gm!",
        "gm, wave. bot online."
    ]
    bot_response = random.choice(gm_variants)
    post_general_response(wave_id, bot_response, jwt_token)


def check_for_unhandled_mentions(wave_id, jwt_token):
    """Check for recent unhandled mentions and process up to 2 of them."""
    print("Checking for any unhandled mentions from previous sessions...")
    print(f"Bot handle is: '{BOT_HANDLE}'")
    
    # Find drops that mention the bot but haven't been replied to yet
    import re
    mention_pattern = re.compile(rf"(?<!\w)@{re.escape(BOT_HANDLE)}(?!\w)", re.IGNORECASE)
    
    # Query for drops that mention the bot
    potential_mentions = Drop.query.filter_by(wave_id=wave_id).order_by(Drop.created_at.desc()).limit(100).all()
    print(f"Found {len(potential_mentions)} potential mentions to check")
    
    # Filter for actual mentions that haven't been replied to
    unhandled_mentions = []
    for drop in potential_mentions:
        is_explicit_mention = bool(mention_pattern.search(drop.content or ""))
        if is_explicit_mention and not getattr(drop, 'bot_replied_to_mention', False):
            unhandled_mentions.append(drop)
            print(f"  Added as unhandled mention")
    
    print(f"Found {len(unhandled_mentions)} unhandled mentions to process")

    # Sort by created_at desc, then serial_no desc as fallback
    unhandled_mentions.sort(key=lambda d: (d.created_at or datetime.min, d.serial_no), reverse=True)
    
    # Process up to 2 most recent unhandled mentions
    mentions_count = len(unhandled_mentions[:2])
    if mentions_count > 0:
        print(f"Found {mentions_count} unhandled mentions from previous sessions")
        for drop in unhandled_mentions[:2]:
            print(f"Generating reply to previously unhandled mention in drop {drop.serial_no}...")
            try:
                reply_to_mention(drop, jwt_token)
                drop.bot_replied_to_mention = True
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                print(f"Error replying to mention or updating DB for drop {drop.serial_no}: {e}")
    else:
        print("No unhandled mentions found from previous sessions.")
    
    return mentions_count

# Check if running migrations or any flask db command
import sys
running_migrations = len(sys.argv) >= 2 and sys.argv[0].endswith('flask') and sys.argv[1] == 'db'

# Register shutdown handler for scheduler
import atexit
atexit.register(lambda: scheduler.shutdown() if scheduler.running else None)

# Initialize app and load data (but skip during migrations)
with app.app_context():
    db.create_all()
    
    if running_migrations:
        print("Database migration detected - skipping initialization and data loading")
    else:
        # Start the scheduler for all non-migration runs
        if not scheduler.running:
            scheduler.start()
            print("Scheduler started - background jobs are now running")
            
        # Only load data when not in migration mode
        jwt_token = authenticate()
        if jwt_token:
            wave = setup_chosen_wave(jwt_token, WAVE_NAME)
            if wave:
                drop_count = Drop.query.filter_by(wave_id=wave.id).count()
                fetch_all_drops_for_wave(jwt_token, wave.id)
                if drop_count == 0:
                    print("First time loading drops for this wave. Posting a gm message.")
                    post_gm_message(wave.id, jwt_token)
                else:
                    # Check for any unhandled mentions that we might have missed
                    check_for_unhandled_mentions(wave.id, jwt_token)
                print("Initial data loaded. Scheduled jobs will handle ongoing updates.")

# Add a new job that runs every 2 hours to generate activity when the channel is quiet

def main():
    """Main entry point when running as a script"""
    print(f"✅ {BOT_HANDLE} bot is up. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down…")

if __name__ == "__main__":
    main()

def activity_check_job():
    """
    Job that runs every 2 hours to check for inactivity and post a message if:
    1. No new activity since last check
    2. Last message is from someone other than the bot
    """
    global LAST_GM_DATE
    
    print("\n" + "="*80)
    print(f"⏰ ACTIVITY CHECK JOB RUNNING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with app.app_context():
        jwt_token = authenticate()
        if not jwt_token:
            print("❌ Authentication failed. Unable to check activity.")
            return
            
        # Get the wave
        wave = Wave.query.filter_by(name=WAVE_NAME).first()
        if not wave:
            print(f"❌ Wave '{WAVE_NAME}' not found in database.")
            return
        
        # Get the last check timestamp from WaveTracking
        wave_tracking = WaveTracking.query.filter_by(wave_id=wave.id).first()
        if not wave_tracking:
            print("❌ Wave tracking not found.")
            return
            
        # Get the 20 most recent drops for context
        recent_drops = Drop.query.filter_by(wave_id=wave.id).order_by(Drop.created_at.desc()).limit(20).all()
        if not recent_drops:
            print("❌ No drops found in the wave.")
            return
            
        # Reverse to get chronological order
        recent_drops.reverse()
        
        # Get the latest drop
        latest_drop = recent_drops[-1] if recent_drops else None
        
        # Check if the latest drop was created after our last check
        last_activity_check_time = getattr(wave_tracking, 'last_activity_check', None)
        current_time = datetime.now()
        
        # Calculate time since last message
        time_since_last_message = None
        if latest_drop and latest_drop.created_at:
            time_since_last_message = current_time - latest_drop.created_at
            time_since_last_message_hours = time_since_last_message.total_seconds() / 3600
            print(f"Time since last message: {time_since_last_message_hours:.1f} hours")
            
        # Get the last time we sent a "gm" message
        last_gm_time_str = "Never" if not LAST_GM_DATE else LAST_GM_DATE.strftime("%Y-%m-%d")
        
        # If this is our first check or there are no new drops since last check
        if not last_activity_check_time or (
            latest_drop.created_at and latest_drop.created_at <= last_activity_check_time
        ):
            # No new activity! Let's check who sent the last message
            if latest_drop.author.lower() != BOT_HANDLE.lower():
                print(f"💤 No new activity since last check and last message is from {latest_drop.author}. Analyzing conversation...")
                
                # Format recent conversation for the LLM
                conversation_history = "\n".join([
                    f"{drop.author}: {drop.content}" 
                    for drop in recent_drops
                ])
                
                # Current context information
                current_date = datetime.now().date()
                current_hour = datetime.now().hour
                
                if current_hour < 12:
                    time_of_day = "morning"
                elif current_hour < 18:
                    time_of_day = "afternoon"
                else:
                    time_of_day = "evening"
                
                # Let the LLM decide the most appropriate response
                prompt = f"""
                You are a bot named {BOT_HANDLE} in a chat wave called "{chosen_wave}". 
                The conversation has been inactive for a while, and you need to post a message to keep it active.
                
                CONTEXT INFORMATION:
                - Current date and time: {current_time.strftime("%Y-%m-%d %H:%M")} ({time_of_day})
                - Last "gm" message sent: {last_gm_time_str}
                - Time since last message: {time_since_last_message_hours:.1f} hours
                - Last message was from: {latest_drop.author}
                
                RECENT CONVERSATION HISTORY (most recent 20 messages):
                {conversation_history}
                
                Based on this context, determine the most appropriate type of message to send:
                1. A direct clever reply to the last message
                2. A random thought related to the conversation themes
                3. A once-a-day greeting, like "gm wavers", or "gmeme, memers", or "gm to all you dive bar denizens". get creative! 
                
                Choose the most natural and context-appropriate option. If the last message asks a question or seems to invite a response, prefer a direct reply.
                If the conversation has been dormant for a long time, a greeting or new topic might be better.
                Only use a greeting like "gm" if it hasn't been used today. you can gm at any time of day.
                
                FORMAT YOUR RESPONSE LIKE THIS:
                Message type: [REPLY, THOUGHT, or GREETING]
                Message: [Your actual message text]
                Reasoning: [Brief explanation of why you chose this type]
                """
                
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=TEMPERATURE,
                )
                
                llm_response = response.choices[0].message.content.strip()
                
                # Parse the LLM response to extract message type and content
                message_type = None
                message_content = None
                reasoning = None
                
                for line in llm_response.split('\n'):
                    line = line.strip()
                    if line.lower().startswith("message type:"):
                        message_type = line.split(":", 1)[1].strip().upper()
                    elif line.lower().startswith("message:"):
                        message_content = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("reasoning:"):
                        reasoning = line.split(":", 1)[1].strip()
                
                if not message_type or not message_content:
                    print("❌ Failed to parse LLM response. Not sending any message.")
                    print(f"Raw LLM response was: {llm_response}")
                    # Don't send anything if we can't parse the message properly
                    return
                
                print(f"LLM suggested message type: {message_type}")
                print(f"Reasoning: {reasoning if reasoning else 'Not provided'}")
                
                # If it's a greeting and it's the first greeting today, update the tracking
                if message_type == "GREETING" and (not LAST_GM_DATE or LAST_GM_DATE != current_date):
                    LAST_GM_DATE = current_date
                
                # Prepare the payload based on the message type
                if message_type == "REPLY":
                    payload = {
                        "wave_id": wave.id,
                        "drop_type": "CHAT",
                        "parts": [{"content": message_content}],
                        "reply_to": {"drop_id": latest_drop.id, "drop_part_id": 0},
                    }
                else:  # THOUGHT or GREETING
                    payload = {
                        "wave_id": wave.id,
                        "drop_type": "CHAT",
                        "parts": [{"content": message_content}],
                    }
                
                # Send the message to the API
                url = f"{BASE_URL}/drops"
                headers = {
                    "Authorization": f"Bearer {jwt_token}",
                    "Content-Type": "application/json",
                }
                
                try:
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    print(f"✅ Posted {message_type} message: {message_content}")
                except Exception as e:
                    print(f"❌ Error posting message: {e}")
            else:
                print(f"⏭️ Last message is from the bot ({BOT_HANDLE}). Skipping activity generation.")
        else:
            print(f"⏭️ There has been activity since the last check. No need to generate a message.")
        
        # Update the last activity check time
        wave_tracking.last_activity_check = datetime.now()
        db.session.commit()
            
    print("="*80)

# Schedule the activity check job to run every 2 hours
# Run activity_check_job every 2 hours, offset by 11 minutes from the hour
scheduler.add_job(
    func=activity_check_job,
    trigger=IntervalTrigger(hours=2, start_date=(datetime.now().replace(minute=11, second=0, microsecond=0))),
    id="activity_check",
    name="Activity Check Job",
    replace_existing=True
)

# ---- keep the script alive so APScheduler jobs can run ----
if __name__ == "__main__":
    print(f"✅ {BOT_HANDLE} bot is up. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down…")
