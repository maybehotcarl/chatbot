import os
import re
import sys
import json
import atexit
import time
import hashlib
import functools

from flask import Flask, request, jsonify
import requests
from requests.exceptions import HTTPError
from dateutil import parser
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from eth_account import Account
from eth_account.messages import encode_defunct
from flask_migrate import Migrate
from datetime import datetime, timezone, timedelta

from models import db, Wave, Drop, Identity, WaveTracking

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

# Initialize the scheduler
scheduler = BackgroundScheduler()

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB for RAG (semantic search over conversation history)
chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"))
chroma_embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
chroma_collection = chroma_client.get_or_create_collection(
    name="dive_bar_drops",
    embedding_function=chroma_embedding_fn,
    metadata={"hnsw:space": "cosine"}
)
print(f"ChromaDB initialized: {chroma_collection.count()} drops in vector store")

# Bot configuration
BOT_HANDLE = os.getenv("BOT_HANDLE", "gray")
print(f"Bot handle: {BOT_HANDLE}")
WAVE_NAME = os.getenv("WAVE_NAME", "maybe's dive bar")
print(f"Wave name: {WAVE_NAME}")

# Load custom system prompt: file > env var > generic default
CUSTOM_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom_prompt.txt')

def load_system_prompt():
    """Load system prompt from custom_prompt.txt, env var, or fall back to generic default."""
    # 1. Try custom_prompt.txt
    if os.path.exists(CUSTOM_PROMPT_FILE):
        try:
            with open(CUSTOM_PROMPT_FILE, 'r') as f:
                prompt = f.read().strip()
            if prompt:
                print(f"System prompt loaded from {CUSTOM_PROMPT_FILE}")
                return prompt
        except Exception as e:
            print(f"Error reading {CUSTOM_PROMPT_FILE}: {e}")

    # 2. Try environment variable
    env_prompt = os.getenv("CUSTOM_SYSTEM_PROMPT", "")
    if env_prompt:
        print("System prompt loaded from CUSTOM_SYSTEM_PROMPT env var")
        return env_prompt

    # 3. Generic default
    print("Using generic default system prompt")
    return f"""
You are {BOT_HANDLE}, a conversational chatbot that responds in 1-2 punchy sentences.
You are a member of the "{WAVE_NAME}" wave.
This is a chat channel in a web3 NFT community focused on decentralization and digital art.

Your responses should be:
- Short and concise (1-2 sentences max)
- Contextually aware of the conversation
- Friendly and helpful
- Occasionally playful but professional

Match the tone and style of the existing conversation.
Be helpful to community members.
"""

SYSTEM_PROMPT = load_system_prompt()

# Community member descriptions -- single source of truth used across all prompts
COMMUNITY_MEMBER_INFO = """
david is a solidity engineer and is obsessed with xTDH.
ricodemus is an engineer of some type and a Philadelphia sports fan.
ezmonet works for a university in accounting.
Intrepid sometimes lives with his family on a boat traveling the world.
zigmarillian is a former body builder and current artist.
Juicy is our resident philosopher and buddhist.
AnimatedNFT is a healthcare sales person.
Blocknoob LOVES complaints.
Call our friend Regular Dad 'Karen'.
Deyvis photographs beautiful women.
Purchardus is a pathologist and project skeptic but also project believer.
Maybe is kinda like dad to you. But you ALWAYS make fun of him.
Sasha is a great artist who's also Maybe's official portrait artist.
MBstuart is a creative artist and talented photographer.
GarryB1rd actually goes by Gary.
JasonOPhoto talks a lot. Make fun of him about it. 
JamesAndrew loves you and you love him and think he's the best photographer here. 
Visithra is everyone's best friend.
""".strip()

# Add a new global variable to track when the last "gm" message was sent
LAST_GM_DATE = None

def safely_create_identity(user_info, author_handle, wave_id=None):
    """
    Creates an Identity object with safe handling of all fields to ensure no dictionaries
    get inserted into the database.
    """
    # Create identity data with proper extraction of nested fields
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

    # Create the identity object
    identity = Identity(**author_data_dict)

    return identity

# Helper function to create OpenAI API parameters based on model compatibility
def create_openai_params(model, prompt_input, system_instructions=None):
    """Create parameters for OpenAI API call with model-specific handling.
    Some models like o3 don't support temperature parameter.
    For gpt-4.1, web_search tool is always enabled.

    Args:
        model: The OpenAI model to use
        prompt_input: The input prompt text
        system_instructions: Optional system instructions, defaults to SYSTEM_PROMPT
    """
    # Use SYSTEM_PROMPT as default if system_instructions is not provided
    if system_instructions is None:
        system_instructions = SYSTEM_PROMPT

    # Keep system_instructions static for prompt caching
    # Fetch recent bot messages and append to the prompt instead
    modified_prompt = prompt_input

    try:
        # Get the 10 most recent bot messages
        recent_bot_drops = Drop.query.filter_by(author=BOT_HANDLE).order_by(
            Drop.created_at.desc()
        ).limit(10).all()

        # Extract just the content
        recent_responses = [drop.content for drop in recent_bot_drops if drop.content]

        # If we have recent responses, append them to the prompt
        if recent_responses:
            recent_responses_text = "\n".join([f"- {resp}" for resp in recent_responses])
            modified_prompt += f"\n\nNOTE: These are my 10 most recent messages. Please avoid repeating similar phrasings, topics, or using cringe slang:\n{recent_responses_text}\n\nMake your response clearly distinct from these previous messages."
    except Exception as e:
        print(f"Error fetching recent bot responses: {e}")

    params = {
        "model": model,
        "input": modified_prompt,
        "instructions": system_instructions,
    }

    # Always add web_search tool for gpt-4.1
    if model == "gpt-4.1":
        params["tools"] = [{"type": "web_search"}]

    return params

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
            except Exception:
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


# Utility functions for parsing LLM API responses
def parse_structured_llm_response(response_text, expected_keys):
    """
    Parses a structured LLM response with key-value pairs.

    Args:
        response_text (str): The text response from the LLM
        expected_keys (list): List of keys to extract from the response

    Returns:
        dict: Dictionary with keys from expected_keys and their extracted values
    """
    results = {key: None for key in expected_keys}

    for line in response_text.split('\n'):
        line = line.strip()
        for key in expected_keys:
            # Format the key as expected in the response (e.g., "message type:")
            formatted_key = key.lower().replace('_', ' ') + ':'
            if line.lower().startswith(formatted_key):
                results[key] = line.split(':', 1)[1].strip()

    return results


def extract_json_from_text(response_text):
    """
    Extracts a JSON object or array from LLM response text.

    Args:
        response_text (str): The text response from the LLM

    Returns:
        tuple: (parsed_json, success)
            - parsed_json: The parsed JSON object or None if parsing failed
            - success: Boolean indicating if parsing was successful
    """
    try:
        # Find the first occurrence of [ or { and the last occurrence of ] or }
        start_idx_array = response_text.find('[')
        start_idx_object = response_text.find('{')

        # Determine which type of JSON structure we're dealing with
        if start_idx_array != -1 and (start_idx_object == -1 or start_idx_array < start_idx_object):
            start_idx = start_idx_array
            end_idx = response_text.rfind(']')
        elif start_idx_object != -1:
            start_idx = start_idx_object
            end_idx = response_text.rfind('}')
        else:
            return None, False

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx+1]
            parsed_json = json.loads(json_str)
            return parsed_json, True
        else:
            return None, False
    except Exception as e:
        print(f"Error extracting JSON from text: {e}")
        return None, False

def upsert_identity(user_info, author_handle, wave_id=None):
    """
    Updates an existing identity or creates a new one if not exists.
    Uses the extracted field values to update all fields with latest data.
    Handles database session operations internally.
    """
    # First check if identity already exists
    existing_identity = Identity.query.filter_by(handle=author_handle).first()

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

    # For new identities only - don't use wave_id for the identity ID
    # Create a deterministic ID based on the handle instead
    author_id = hashlib.md5(author_handle.encode()).hexdigest()

    if existing_identity:
        try:
            # Update existing identity with new data
            for key, value in author_data.items():
                if value is not None:  # Only update non-None values
                    setattr(existing_identity, key, value)

            # Don't commit here - let the caller commit when ready
            return existing_identity
        except Exception as e:
            db.session.rollback()
            print(f"Error updating identity {author_handle}: {e}")
            return existing_identity
    else:
        # Create new identity - include the handle and id
        try:
            new_identity_data = author_data.copy()
            new_identity_data['handle'] = author_handle
            new_identity_data['id'] = author_id

            # Create and add the new identity
            new_identity = Identity(**new_identity_data)
            db.session.add(new_identity)

            # Flush to ensure it's saved but don't commit yet
            db.session.flush()

            print(f"Added new identity: {author_handle}")
            return new_identity
        except Exception as e:
            db.session.rollback()
            print(f"Error creating identity {author_handle}: {e}")

            # Even if creation failed, try to get the identity as it might exist
            return Identity.query.filter_by(handle=author_handle).first()

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

# Retry decorator for API calls
def retry_api_call(max_retries=3, backoff_factor=1.5, retry_on_exceptions=(requests.exceptions.RequestException,)):
    """
    Retry decorator for API calls that might fail temporarily.

    Args:
        max_retries: Maximum number of retries
        backoff_factor: Backoff factor for exponential delay between retries
        retry_on_exceptions: Tuple of exceptions that should trigger a retry
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except retry_on_exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        # Only log final failure
                        raise

                    # Calculate backoff delay with exponential backoff
                    delay = backoff_factor ** retries
                    print(f"Retrying API call ({retries}/{max_retries})")
                    time.sleep(delay)
        return wrapper
    return decorator

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
        print("Authentication successful. JWT token received.")
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
        "limit": str(limit),  # Convert to string to match fetch_all_drops_for_wave
        "wave_id": wave_id,
        "include_replies": "true",  # Use string "true" to match fetch_all_drops_for_wave
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
                existing_drop = db.session.get(Drop, drop_id)
                if existing_drop:

                    # Is the serial number higher but the drop exists?
                    if drop_serial > existing_drop.serial_no:
                        print(f"Found drop with ID {drop_id} but higher serial: API={drop_serial}, DB={existing_drop.serial_no}")
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
                if reply_to_id and not db.session.get(Drop, reply_to_id):
                    # Check if we've already created this stub in the current session
                    existing_stub = next((obj for obj in db.session.new
                                        if isinstance(obj, Drop) and obj.id == reply_to_id), None)
                    if not existing_stub:
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
                    else:
                        print(f"Skipping duplicate stub creation for ID: {reply_to_id}")

            # Handle nested replies (replies to replies)
            nested_parent_drop = reply_to.get("drop", None)
            if nested_parent_drop:
                nested_reply_to_id = nested_parent_drop.get("id")
                # If reply_to_id is missing but nested exists, use nested ID
                if reply_to_id is None and nested_reply_to_id:
                    reply_to_id = nested_reply_to_id
                with db.session.no_autoflush:
                    if nested_reply_to_id and not db.session.get(Drop, nested_reply_to_id):
                        # Check if we've already created this stub in the current session
                        existing_stub = next((obj for obj in db.session.new
                                            if isinstance(obj, Drop) and obj.id == nested_reply_to_id), None)
                        if not existing_stub:
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
                        else:
                            print(f"Skipping duplicate nested stub creation for ID: {nested_reply_to_id}")
            # --- End processing reply logic ---

            # 6. Process basic drop fields
            content_parts = drop_data.get("parts", [])
            content = "\n".join(
                str(part.get("content", "")) for part in drop_data.get("parts", [])
            )
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

            # Handle identity creation/updating if needed
            with db.session.no_autoflush:
                existing_identity = Identity.query.filter_by(handle=author_handle).first()
                if existing_identity is None:
                    # If we don't have it in our local DB, fetch from API
                    user_info = fetch_user_by_handle(jwt_token, author_handle)
                    if user_info:
                        authors_processed += 1
                        author_data = upsert_identity(user_info, author_handle, wave_id)
                        # Check for dictionaries before adding
                        check_object_for_dicts(author_data, f"Identity-{author_handle}")
                    else:
                        # Fallback to a minimal record
                        minimal_user_info = {
                            'pfp': None,
                            'profile_url': f"https://6529.io/{author_handle}"
                        }
                        authors_processed += 1
                        author_data = upsert_identity(minimal_user_info, author_handle)

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
            # Check if we've already created this drop in the current session
            existing_drop_in_session = next((obj for obj in db.session.new
                                        if isinstance(obj, Drop) and obj.id == drop_id), None)
            if not existing_drop_in_session:
                # Check for dictionaries before adding
                check_object_for_dicts(new_drop, f"Drop-{drop_id}")
                db.session.add(new_drop)
                drops_added += 1
                new_drops_found = True
            else:
                print(f"Skipping duplicate drop creation for ID: {drop_id}")

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

                # Embed newly added drops into ChromaDB
                try:
                    new_drop_records = Drop.query.filter(
                        Drop.wave_id == wave_id,
                        Drop.serial_no > local_max_serial,
                        Drop.content.isnot(None),
                        Drop.content != "",
                    ).all()
                    # Filter out stubs and empty/whitespace-only content
                    embeddable = [
                        d for d in new_drop_records
                        if d.content and d.content.strip()
                        and not (d.author == "Unknown" and d.content.startswith("(Stub"))
                    ]
                    if embeddable:
                        chroma_collection.upsert(
                            ids=[d.id for d in embeddable],
                            documents=[d.content.strip() for d in embeddable],
                            metadatas=[
                                {
                                    "author": d.author or "Unknown",
                                    "wave_id": d.wave_id or "",
                                    "serial_no": d.serial_no or 0,
                                    "created_at": d.created_at.isoformat() if d.created_at else "",
                                }
                                for d in embeddable
                            ],
                        )
                        print(f"Embedded {len(embeddable)} new drops into ChromaDB")
                except Exception as e:
                    print(f"ChromaDB embedding error (non-fatal): {e}")

                # Log summary of what happened
                print(f"Drop sync summary: {total_fetched} fetched, {skipped_count} skipped, {drops_added} added, {stubs_created} stubs created, {authors_processed} identities processed")
                print(f"Serial range: {local_max_serial} -> {newest_serial}")

            except Exception as e:
                db.session.rollback()  # Rollback on failure
                print(f"Database error: {e}")
        else:
            print(f"Drop sync summary: {total_fetched} fetched, {skipped_count} skipped, 0 added (all drops already existed)")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching drops for wave {wave_id}: {e}")
        return

    return new_serials


"""
===================================================
bot section
===================================================
"""
GENERAL_RESPONSE_THRESHOLD = 50
MAX_DROPS_FOR_SUMMARY = 50

def fetch_and_process_drops(wave_id, wave_tracking):
    """
    Fetch new drops that haven't been processed for interactions yet.
    Returns (new_drops, drop_count) or (None, 0) if no new drops.
    """
    last_interaction_serial = getattr(wave_tracking, 'last_interaction_serial_no', 0)
    print(f"Last interaction serial: {last_interaction_serial}")

    new_drops = Drop.query.filter(
        Drop.wave_id == wave_id,
        Drop.serial_no > last_interaction_serial
    ).order_by(Drop.serial_no.asc()).all()

    drop_count = len(new_drops)

    if drop_count == 0:
        print("Monitoring summary: 0 new drops to process for interaction")
        return None, 0
    else:
        print(f"Monitoring summary: {drop_count} drops to check for interactions")
        return new_drops, drop_count

def handle_bot_responses(new_drops, wave_id, jwt_token, wave_tracking):
    """
    Handle mentions, direct replies, and general responses based on accumulated drops.
    Returns general_response_posted boolean.
    """
    general_response_posted = False

    try:
        mentions_count, replies_count = handle_new_drops(new_drops, wave_id, jwt_token)
        if mentions_count > 0:
            print("Skipping general response because a mention was already replied to.")
            wave_tracking.accumulated_new_drops = 0
            general_response_posted = True
        else:
            print(f"Interaction summary: {mentions_count} mentions, {replies_count} direct replies")
    except Exception as e:
        print(f"Error handling new drops: {e}")
        raise

    # Add the count of new drops to the accumulator
    drop_count = len(new_drops)
    wave_tracking.accumulated_new_drops += drop_count
    print(f"Accumulated new drops: {wave_tracking.accumulated_new_drops}/{GENERAL_RESPONSE_THRESHOLD}")

    # Only generate responses if the accumulated count exceeds threshold
    if wave_tracking.accumulated_new_drops >= GENERAL_RESPONSE_THRESHOLD:
        wave_tracking.accumulated_new_drops = 0

        # Get drops that aren't from the bot itself
        non_bot_drops = [
            drop for drop in new_drops
            if drop.author.lower() != BOT_HANDLE.lower() and not is_bot_mentioned(drop)
        ]

        if non_bot_drops:
            # Sort by serial number (descending) to get the most recent drops
            sorted_drops = sorted(non_bot_drops, key=lambda d: d.serial_no, reverse=True)

            # Get recent conversation context using our helper function
            recent_context = get_recent_conversation_context(wave_id, limit=20)

            # Use LLM to intelligently select which drops to respond to
            # Still use new_drops for selection logic but use full context for response
            drops_to_respond_to = select_drops_to_respond_to(sorted_drops, new_drops)

            responses_count = 0

            # Generate response for only the first drop to avoid multiple responses
            if drops_to_respond_to:
                drop_to_respond_to = drops_to_respond_to[0]  # Only take the first one
                print(f"Generating a response to drop {drop_to_respond_to.serial_no} by {drop_to_respond_to.author}...")

                # Create a context that includes the specific drop being responded to
                # and the recent conversation (last 20 drops)
                context_text = f"\nYou are specifically replying to this message: {drop_to_respond_to.author}: {drop_to_respond_to.content}\n\nRecent conversation context (last 20 messages):\n{recent_context}"

                try:
                    bot_response = generate_general_response(context_text, wave_id=wave_id)

                    if bot_response is not None:
                        # Post the response as a reply to this specific drop using retry mechanism
                        try:
                            post_response_with_retry(
                                wave_id=wave_id,
                                content=bot_response,
                                jwt_token=jwt_token,
                                reply_to_id=drop_to_respond_to.id,
                                reply_to_part_id=0
                            )
                            responses_count = 1
                            print(f"Response posted as a reply to drop {drop_to_respond_to.serial_no}!")
                        except Exception as e:
                            print(f"Error posting response to drop {drop_to_respond_to.serial_no} after retries: {e}")
                    else:
                        print(f"Response generation failed for drop {drop_to_respond_to.serial_no}")
                except Exception as e:
                    print(f"Error generating response for drop {drop_to_respond_to.serial_no}: {e}")

            general_response_posted = responses_count > 0
            print(f"Posted {responses_count} individual response to drop" + ("s" if responses_count != 1 else ""))

    return general_response_posted

def extract_time_hours_from_text(text):
    """
    Extract time duration in hours from user text like 'last 3 hours', 'past 6h', etc.
    Returns hours as integer, or None if no time found.
    """
    text = text.lower()

    # Look for patterns like "last X hours", "past X hours", "X hours ago"
    hour_patterns = [
        r'(?:last|past)\s*(\d+)\s*(?:hour|hr)s?',
        r'(\d+)\s*(?:hour|hr)s?\s*(?:ago|back)',
        r'(\d+)\s*h(?:\b|$)',  # "3h" or "6h "
        r'(\d+)\s*(?:hour|hr)s?'  # Just "3 hours"
    ]

    for pattern in hour_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                hours = int(match.group(1))
                return hours
            except (ValueError, IndexError):
                continue

    return None

def get_recent_conversation_context(wave_id, limit=20):
    """
    Fetch the most recent drops from a wave to provide conversation context.
    Returns a formatted string with the recent conversation.
    """
    recent_drops = Drop.query.filter_by(wave_id=wave_id).order_by(
        Drop.serial_no.desc()
    ).limit(limit).all()

    # Reverse to show chronological order (oldest first)
    recent_drops.reverse()

    if not recent_drops:
        return ""

    context_lines = []
    for drop in recent_drops:
        if drop.content:  # Only include drops with content
            context_lines.append(f"{drop.author}: {drop.content}")

    return "\n".join(context_lines)

def retrieve_relevant_history(query_text, wave_id, n_results=20, exclude_serial_nos=None, context_window=5):
    """
    Query ChromaDB for drops semantically similar to query_text, then expand
    each hit by fetching surrounding drops (±context_window) from SQLite to
    capture full conversational flow. Returns formatted string or empty on error.
    """
    try:
        if not query_text or not query_text.strip():
            return ""

        # Strip @mentions and common request phrasing so semantic search
        # focuses on the actual topic, not the "hey @Gray can you..." wrapper
        clean_query = re.sub(r'@\[?\w+\]?', '', query_text)
        clean_query = clean_query.strip()
        if not clean_query:
            clean_query = query_text  # fallback to original if stripping emptied it
        print(f"RAG clean query: {clean_query[:80]}...")

        where_filter = {"wave_id": wave_id} if wave_id else None

        results = chroma_collection.query(
            query_texts=[clean_query],
            n_results=n_results + (len(exclude_serial_nos) if exclude_serial_nos else 0),
            where=where_filter,
            include=["documents", "metadatas"],
        )

        if not results or not results["documents"] or not results["documents"][0]:
            print(f"RAG query returned no results for: {query_text[:80]}...")
            return ""

        # Collect the anchor serial numbers from RAG hits
        exclude = set(exclude_serial_nos) if exclude_serial_nos else set()
        anchor_serials = []
        for meta in results["metadatas"][0]:
            serial = meta.get("serial_no", 0)
            if serial and serial not in exclude:
                anchor_serials.append(serial)
            if len(anchor_serials) >= n_results:
                break

        if not anchor_serials:
            return ""

        print(f"RAG found {len(anchor_serials)} anchor drops for query: {query_text[:80]}...")

        # Expand each anchor by ±context_window, fetch from SQLite
        all_serials = set()
        for serial in anchor_serials:
            for offset in range(-context_window, context_window + 1):
                s = serial + offset
                if s > 0 and s not in exclude:
                    all_serials.add(s)

        # Fetch all expanded drops from SQLite in one query
        expanded_drops = Drop.query.filter(
            Drop.serial_no.in_(all_serials),
            Drop.content.isnot(None),
            Drop.content != "",
        ).order_by(Drop.serial_no.asc()).all()

        # Filter out stubs
        expanded_drops = [
            d for d in expanded_drops
            if not (d.author == "Unknown" and d.content and d.content.startswith("(Stub"))
        ]

        # Format as lines
        lines = []
        for d in expanded_drops:
            if d.created_at:
                try:
                    timestamp = d.created_at.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    timestamp = "unknown time"
            else:
                timestamp = "unknown time"
            lines.append(f"[{timestamp}] {d.author}: {d.content}")

        result_text = "\n".join(lines)
        print(f"RAG expanded to {len(lines)} drops (from {len(anchor_serials)} anchors, ±{context_window} window)")
        for line in lines[:10]:
            print(f"  RAG> {line[:120]}")
        if len(lines) > 10:
            print(f"  RAG> ... and {len(lines) - 10} more")
        return result_text
    except Exception as e:
        print(f"ChromaDB retrieval error (non-fatal): {e}")
        return ""


def retrieve_author_drops(author_handle, sample_size=100, min_drops=20):
    """
    Query SQLite for a target author's drops, evenly sampled across their full
    history to get a representative personality sample. Returns a formatted
    string of their messages with timestamps, or None if fewer than min_drops.
    """
    try:
        # Case-insensitive lookup: find the canonical handle from the DB
        canonical = db.session.query(Drop.author).filter(
            db.func.lower(Drop.author) == author_handle.lower()
        ).first()
        if canonical:
            author_handle = canonical[0]  # use the DB's casing

        # Get total count for this author
        total = Drop.query.filter(
            Drop.author == author_handle,
            Drop.content.isnot(None),
            Drop.content != "",
        ).count()

        if total < min_drops:
            print(f"Personality analysis: {author_handle} has only {total} drops (need {min_drops})")
            return None

        # Fetch all drop IDs ordered by time so we can evenly sample
        all_drops = Drop.query.filter(
            Drop.author == author_handle,
            Drop.content.isnot(None),
            Drop.content != "",
        ).order_by(Drop.created_at.asc()).all()

        # Evenly sample across the full history
        if len(all_drops) <= sample_size:
            sampled = all_drops
        else:
            step = len(all_drops) / sample_size
            sampled = [all_drops[int(i * step)] for i in range(sample_size)]

        # Format as timestamped lines
        lines = []
        for d in sampled:
            if d.created_at:
                try:
                    timestamp = d.created_at.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    timestamp = "unknown time"
            else:
                timestamp = "unknown time"
            lines.append(f"[{timestamp}] {d.author}: {d.content}")

        print(f"Personality analysis: fetched {len(lines)} drops for {author_handle} (from {total} total)")
        return "\n".join(lines)
    except Exception as e:
        print(f"Error retrieving author drops for {author_handle}: {e}")
        return None


def generate_personality_analysis(author_handle, sampled_messages):
    """
    Send sampled messages to GPT-4.1 with a personality analysis prompt.
    Returns the analysis text in Gray's voice.
    """
    prompt = f"""You are analyzing the chat history of a community member named "{author_handle}" from a dive bar chat called "{WAVE_NAME}".

Below is a representative sample of ~100 of their messages spread across their full history:

{sampled_messages}

Based on these messages, write a personality profile that includes:
1. **Communication style & tone** -- how do they talk? Are they verbose, terse, sarcastic, earnest, chaotic?
2. **Key interests & recurring topics** -- what do they keep coming back to?
3. **Myers-Briggs guess** -- give your best MBTI type guess with brief reasoning for each letter
4. **The Gray summary** -- a fun, punchy 2-3 sentence summary of this person written in the voice of a witty dive bar bartender who's been watching them for months. Not clinical, not mean -- affectionate roasting is welcome.

Additional context about known community members that may help:
{COMMUNITY_MEMBER_INFO}

Keep the whole response under 300 words. Use markdown formatting sparingly (bold for MBTI type and section headers only)."""

    try:
        params = {
            "model": "gpt-4.1",
            "input": prompt,
            "instructions": f"You are {BOT_HANDLE}, a witty dive bar chatbot analyzing a regular's personality. Be insightful but fun -- this is a parlor trick, not a therapy session."
        }
        response = client.responses.create(**params)
        return response.output_text.strip()
    except Exception as e:
        print(f"Error generating personality analysis for {author_handle}: {e}")
        return None


def embed_all_existing_drops(wave_id):
    """
    Bulk-embed all existing drops from SQLite into ChromaDB.
    Skips stubs and drops already in ChromaDB. Batches in groups of 5000.
    """
    print("Running bulk embedding of existing drops into ChromaDB...")
    start_time = time.time()

    # Get all non-stub drops with content
    all_drops = Drop.query.filter(
        Drop.wave_id == wave_id,
        Drop.content.isnot(None),
        Drop.content != "",
    ).order_by(Drop.serial_no.asc()).all()

    # Filter out stubs and drops with empty/whitespace-only content
    drops_to_embed = [
        d for d in all_drops
        if d.content and d.content.strip()
        and not (d.author == "Unknown" and d.content.startswith("(Stub"))
    ]

    if not drops_to_embed:
        print("No drops to embed.")
        return

    # Check which IDs are already in ChromaDB (batch the check too)
    all_ids = [d.id for d in drops_to_embed]
    existing_ids = set()
    for i in range(0, len(all_ids), 5000):
        batch_ids = all_ids[i:i+5000]
        try:
            result = chroma_collection.get(ids=batch_ids)
            existing_ids.update(result["ids"])
        except Exception:
            pass

    # Filter to only new drops
    new_drops = [d for d in drops_to_embed if d.id not in existing_ids]

    if not new_drops:
        print(f"All {len(drops_to_embed)} drops already in ChromaDB. Skipping.")
        return

    print(f"Embedding {len(new_drops)} new drops ({len(existing_ids)} already in ChromaDB)...")

    # Batch embed in groups of 2000 (API limit friendly)
    batch_size = 2000
    for i in range(0, len(new_drops), batch_size):
        batch = new_drops[i:i+batch_size]
        batch_ids = [d.id for d in batch]
        batch_documents = [d.content.strip() for d in batch]
        batch_metadatas = [
            {
                "author": d.author or "Unknown",
                "wave_id": d.wave_id or "",
                "serial_no": d.serial_no or 0,
                "created_at": d.created_at.isoformat() if d.created_at else "",
            }
            for d in batch
        ]

        try:
            chroma_collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )
            elapsed = time.time() - start_time
            print(f"  Embedded batch {i//batch_size + 1} ({len(batch)} drops, {elapsed:.1f}s elapsed)")
        except Exception as e:
            print(f"  Error embedding batch {i//batch_size + 1}: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"Bulk embedding complete: {len(new_drops)} drops in {elapsed:.1f}s. ChromaDB total: {chroma_collection.count()}")


def chunk_conversation(drops, max_chunk_size=3000):
    """
    Split large conversations into smaller chunks for processing.
    Returns list of formatted conversation chunks.
    """
    chunks = []
    current_chunk = []
    current_size = 0

    for drop in drops:
        timestamp = drop.created_at.strftime("%H:%M") if drop.created_at else "??:??"
        message_line = f"[{timestamp}] {drop.author}: {drop.content}"
        message_size = len(message_line)

        # If adding this message would exceed chunk size, start new chunk
        if current_size + message_size > max_chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(message_line)
        current_size += message_size

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def summarize_conversation_by_hours(wave_id, requested_hours, requester_name="someone"):
    """
    Summarize conversation from the last X hours, with max limit of 12 hours.
    Uses intelligent chunking for large conversations with performance optimizations.
    Returns a tuple: (summary_text, actual_hours_used, was_limited)
    """
    start_time = time.time()
    max_processing_time = 45  # Maximum 45 seconds for summarization
    # Enforce 12-hour maximum
    actual_hours = min(requested_hours, 12)
    was_limited = requested_hours > 12

    # Query drops from the specified time period
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=actual_hours)

    drops = Drop.query.filter(
        Drop.wave_id == wave_id,
        Drop.created_at >= cutoff_time,
        Drop.content.isnot(None),  # Only drops with content
        Drop.content != ""  # Non-empty content
    ).order_by(Drop.created_at.asc()).all()

    if not drops:
        return f"No messages found in the last {actual_hours} hour{'s' if actual_hours != 1 else ''}.", actual_hours, was_limited

    # For small conversations (< 500 messages), summarize directly
    if len(drops) <= 500:
        conversation_text = []
        for drop in drops:
            timestamp = drop.created_at.strftime("%H:%M") if drop.created_at else "??:??"
            conversation_text.append(f"[{timestamp}] {drop.author}: {drop.content}")

        full_conversation = "\n".join(conversation_text)

        # Create summarization prompt with more generous length allowance
        prompt = f"""Please provide a comprehensive summary of this conversation from the last {actual_hours} hour{'s' if actual_hours != 1 else ''}. Include:
- Key topics and themes discussed
- Important decisions, announcements, or news shared
- Notable questions and answers
- Interesting interactions or debates
- Any significant developments or changes

Feel free to be thorough (3-4 sentences) since this covers {actual_hours} hours of activity with {len(drops)} messages.

Conversation:
{full_conversation}"""

        try:
            # Use a more specialized summarization call without system prompt interference
            params = {
                "model": "gpt-4.1",
                "input": prompt,
                "instructions": "You are a helpful assistant that creates comprehensive but concise summaries of conversations. Focus on the most important and interesting parts."
            }
            response = client.responses.create(**params)
            summary = response.output_text.strip()

        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"I had trouble summarizing the last {actual_hours} hours, but there were {len(drops)} messages during that time.", actual_hours, was_limited

    else:
        # For large conversations, use optimized chunked summarization
        print(f"Large conversation detected ({len(drops)} messages), using fast chunked summarization")

        # Use larger chunks and limit number of chunks for performance
        conversation_chunks = chunk_conversation(drops, max_chunk_size=8000)  # Larger chunks
        max_chunks = 6  # Limit to 6 chunks maximum for performance

        if len(conversation_chunks) > max_chunks:
            # Take evenly spaced chunks across the time period for representative sample
            step = len(conversation_chunks) // max_chunks
            selected_chunks = [conversation_chunks[i * step] for i in range(max_chunks)]
            print(f"Sampling {max_chunks} chunks from {len(conversation_chunks)} for performance")
            conversation_chunks = selected_chunks

        chunk_summaries = []

        for i, chunk in enumerate(conversation_chunks):
            # Check timeout
            if time.time() - start_time > max_processing_time:
                print(f"Summarization timeout reached after {max_processing_time}s, stopping at chunk {i+1}")
                break

            # Shorter, more focused prompt for speed
            chunk_prompt = f"""Briefly summarize this conversation chunk ({i+1}/{len(conversation_chunks)}) - focus only on the most important topics and events:

{chunk}"""

            try:
                params = {
                    "model": "gpt-4.1",
                    "input": chunk_prompt,
                    "instructions": "Create a brief summary focusing only on key topics and important events. Be concise."
                }
                response = client.responses.create(**params)
                chunk_summaries.append(response.output_text.strip())
                print(f"Chunk {i+1}/{len(conversation_chunks)} processed in {time.time() - start_time:.1f}s")
            except Exception as e:
                print(f"Error summarizing chunk {i+1}: {e}")
                # Skip failed chunks rather than adding placeholder
                continue

        if not chunk_summaries:
            return f"I had trouble processing the {len(drops)} messages from the last {actual_hours} hours, but it was quite busy!", actual_hours, was_limited

        # Simple combination instead of additional LLM call for speed
        if len(chunk_summaries) == 1:
            summary = chunk_summaries[0]
        else:
            # Just join the summaries with transitions
            summary = " ".join([f"Early: {chunk_summaries[0]}"] +
                              ([f"Mid: {chunk_summaries[len(chunk_summaries)//2]}"] if len(chunk_summaries) > 2 else []) +
                              [f"Recent: {chunk_summaries[-1]}"] if len(chunk_summaries) > 1 else [])

            # Add context about the scale
            summary = f"Busy {actual_hours} hours with {len(drops)} messages. {summary}"

    # Add context about time limitation if needed
    if was_limited:
        summary = f"(12 hours is my max, but here's what happened:) {summary}"

    # Add timing info for debugging
    total_time = time.time() - start_time
    print(f"Summarization completed in {total_time:.1f}s for {len(drops)} messages")

    return summary, actual_hours, was_limited

def update_wave_tracking(wave_tracking, new_drops, drop_count, general_response_posted):
    """
    Update wave tracking with the latest processed serial numbers and commit to database.
    """
    # Update last interaction serial to the highest serial number from new_drops
    if new_drops:
        highest_serial_no = max(drop.serial_no for drop in new_drops)
        last_interaction_serial = getattr(wave_tracking, 'last_interaction_serial_no', 0)
        print(f"Interaction serial range: {last_interaction_serial} -> {highest_serial_no}")
        wave_tracking.last_interaction_serial_no = highest_serial_no

    try:
        # Database operation
        db.session.commit()
        print(f"Monitoring completed: {drop_count} drops processed, general response: {'yes' if general_response_posted else 'no'}")
    except Exception as e:
        db.session.rollback()  # Rollback on failure
        print(f"Database error: {e}")
        raise

def monitor_memes_chat(jwt_token=None, wave=None):
    """
    Refactored monitor function that orchestrates the three main operations:
    1. Fetch and process new drops
    2. Handle bot responses (mentions, replies, general responses)
    3. Update wave tracking
    """
    if wave is None:
        wave = Wave.query.filter_by(name=WAVE_NAME).first()
        if not wave:
            print(f"Wave '{WAVE_NAME}' not found.")
            return

    tracking = WaveTracking.query.filter_by(wave_id=wave.id).first()
    if tracking and tracking.oracle_active:
        print("Oracle protocol active -- skipping bot interaction.")
        return

    print("monitor_memes_chat() ran at", datetime.now())
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

    # Step 1: Fetch and process new drops
    new_drops, drop_count = fetch_and_process_drops(wave_id, wave_tracking)
    if new_drops is None:
        return  # No new drops to process

    # Step 2: Handle bot responses
    try:
        general_response_posted = handle_bot_responses(new_drops, wave_id, jwt_token, wave_tracking)
    except Exception as e:
        print(f"Error in handle_bot_responses: {e}")
        return

    # Step 3: Update wave tracking
    update_wave_tracking(wave_tracking, new_drops, drop_count, general_response_posted)

def trace_reply_chain(drop):
    chain = []
    current = drop
    while current.reply_to_id:
        parent = db.session.get(Drop, current.reply_to_id)
        if not parent:
            break
        chain.append((parent.serial_no, parent.author))
        current = parent
    print(f"Reply chain: {chain}")

def handle_new_drops(new_drops, wave_id, jwt_token):
    drops_text = "\n".join([f"{drop.author}: {drop.content}" for drop in new_drops])
    print(f"Processing {len(new_drops)} new drops... {drops_text}")

    mentioned_drops = []
    replied_drops = []

    for drop in new_drops:
        is_explicit_mention = is_bot_mentioned(drop)

        if drop.author.lower() == BOT_HANDLE.lower():
            print(f"Skipping drop {drop.serial_no} -- authored by the bot.")
            continue

        if is_reply_to_bot(drop):
            print(f"Drop {drop.serial_no} is a reply to the bot.")
            print("    Reply chain:", trace_reply_chain(drop))
            replied_drops.append(drop)

        if is_explicit_mention:
            mentioned_drops.append(drop)

    mentions_count = 0
    replies_count = len(replied_drops)

    # Process mentions
    if mentioned_drops:
        print(f"Processing {len(mentioned_drops)} mention(s)")
        # Only reply to unhandled mentions
        unhandled_mentions = [drop for drop in mentioned_drops if not getattr(drop, 'bot_replied_to', False)]
        # Sort by created_at desc, then serial_no desc as fallback
        unhandled_mentions.sort(key=lambda d: (d.created_at or datetime.min, d.serial_no), reverse=True)

        print(f"Processing {len(unhandled_mentions)} new mentions")
        for drop in unhandled_mentions:
            print(f"Generating reply to mention in drop {drop.serial_no}...")
            try:
                reply_result = reply_to_mention(drop, jwt_token)
                if reply_result:
                    # Only mark as replied to after successful API call
                    drop.bot_replied_to = True
                    db.session.commit()
                    print(f"Drop {drop.serial_no} marked as replied to after successful API call")
                    mentions_count += 1
                else:
                    print(f"Failed to reply to drop {drop.serial_no}, not marking as replied to")
            except Exception as e:
                db.session.rollback()
                print(f"Error replying to mention or updating DB for drop {drop.serial_no}: {e}")

        # All mentions processed
        print(f"Processed all {len(unhandled_mentions)} unhandled mentions")
        for drop in mentioned_drops:
            if getattr(drop, 'bot_replied_to', False):
                print(f"Already replied to mention in drop {drop.serial_no}, skipping.")

    # Process direct replies to bot's messages
    if replies_count > 0:
        print(f"Processing {replies_count} direct reply/replies")
        # Only reply to unhandled direct replies that aren't also mentions (to avoid duplicate responses)
        unhandled_replies = [drop for drop in replied_drops
                            if not getattr(drop, 'bot_replied_to', False) and drop not in mentioned_drops]
        # Sort by created_at desc, then serial_no desc as fallback
        unhandled_replies.sort(key=lambda d: (d.created_at or datetime.min, d.serial_no), reverse=True)
        # Reply to ALL unhandled direct replies
        for drop in unhandled_replies:
            print(f"Generating reply to direct reply in drop {drop.serial_no}...")
            try:
                # reply_to_mention now returns True on success
                if reply_to_mention(drop, jwt_token):
                    # Only mark as replied to after successful API call
                    drop.bot_replied_to = True
                    db.session.commit()
                    print(f"Drop {drop.serial_no} marked as replied to after successful API call")
            except Exception as e:
                db.session.rollback()
                print(f"Error replying to direct reply or updating DB for drop {drop.serial_no}: {e}")
        print(f"Processed all {len(unhandled_replies)} unhandled direct replies")
        for drop in replied_drops:
            if getattr(drop, 'bot_replied_to', False):
                print(f"Already replied to direct reply in drop {drop.serial_no}, skipping.")

    return mentions_count, replies_count

def reply_to_mention(drop, jwt_token):
    """
    Generates and posts a response directly to a drop mentioning the bot.
    Returns True if the response was successfully posted or if the drop doesn't exist.
    Returns False if there was an error that should prevent marking the drop as processed.
    """
    # Check if this is a summarization request
    content_lower = drop.content.lower()
    if any(word in content_lower for word in ['summarize', 'summary', 'recap', 'what happened', 'catch me up']):
        # Extract time duration from the request
        requested_hours = extract_time_hours_from_text(drop.content)

        if requested_hours:
            # Generate time-based summary
            summary, actual_hours, was_limited = summarize_conversation_by_hours(drop.wave_id, requested_hours, drop.author)
            bot_response = summary
        else:
            # Default to 3 hours if no specific time mentioned
            summary, actual_hours, was_limited = summarize_conversation_by_hours(drop.wave_id, 3, drop.author)
            bot_response = f"Here's what's been happening (last 3 hours): {summary}"
    elif any(phrase in content_lower for phrase in [
        'personality', 'myers-briggs', 'mbti', 'what am i like',
        'what type am i', 'analyze me', 'describe me', 'what type is',
        'personality type', 'what kind of person',
    ]):
        # Personality analysis / Myers-Briggs parlor trick
        # Determine target author: "analyze me" → drop.author, "analyze maybe" → parse handle
        self_keywords = ['analyze me', 'describe me', 'my personality', 'my type',
                         'what am i like', 'what type am i', 'my mbti', 'my myers']
        is_self_request = any(kw in content_lower for kw in self_keywords)

        if is_self_request:
            target_author = drop.author
        else:
            # Strip the bot @mention so the LLM doesn't get confused
            cleaned_content = re.sub(rf'@\[?{re.escape(BOT_HANDLE)}\]?', '', drop.content, flags=re.IGNORECASE).strip()
            extract_prompt = f"""Extract the person's handle/username whose personality is being asked about from this message.
The message is from a chat where people have handles like "maybe", "david", "ricodemus", "BrynnAlise", "Zigmarillion", etc.
IMPORTANT: The bot "{BOT_HANDLE}" is being asked to analyze someone -- do NOT return "{BOT_HANDLE}" as the target.
Only return the bare handle (lowercase, no @), nothing else. If you can't determine a specific person, return "UNKNOWN".

Message: {cleaned_content}"""
            try:
                extract_resp = client.responses.create(
                    model="gpt-4.1",
                    input=extract_prompt,
                    instructions=f"Return only the handle of the person being analyzed, nothing else. Never return '{BOT_HANDLE}'."
                )
                parsed_handle = extract_resp.output_text.strip().lower().strip('@[]')
                if parsed_handle and parsed_handle != "unknown":
                    target_author = parsed_handle
                else:
                    target_author = drop.author  # fallback to requester
            except Exception as e:
                print(f"Error extracting personality target handle: {e}")
                target_author = drop.author

        print(f"Personality analysis requested by {drop.author} for target: {target_author}")

        # Retrieve sampled drops for the target author
        sampled_messages = retrieve_author_drops(target_author)

        if sampled_messages is None:
            bot_response = f"I don't have enough bar history on {target_author} to do a proper read yet -- need at least 20 messages to work with."
        else:
            analysis = generate_personality_analysis(target_author, sampled_messages)
            if analysis:
                bot_response = analysis
            else:
                bot_response = f"I tried to read {target_author}'s personality but my crystal ball glitched. Try again in a sec."
    else:
        # Regular mention response
        # Get recent conversation context for better responses
        recent_context = get_recent_conversation_context(drop.wave_id, limit=20)

        # Get semantically relevant past conversations via RAG
        recent_drops = Drop.query.filter_by(wave_id=drop.wave_id).order_by(
            Drop.serial_no.desc()
        ).limit(20).all()
        exclude_serials = [d.serial_no for d in recent_drops if d.serial_no]
        relevant_history = retrieve_relevant_history(
            drop.content, drop.wave_id, n_results=20, exclude_serial_nos=exclude_serials
        )

        rag_section = ""
        if relevant_history:
            rag_section = f"""

        RELEVANT PAST CONVERSATIONS (from the bar's history -- use as background knowledge, not as things happening now):
        {relevant_history}
        """

        prompt = f"""
        You were mentioned in the following post:
        {drop.content}. The drop author is {drop.author}.
        Respond directly to the comment in a friendly way keeping it brief (1-2 sentences).
        If the message contains a question that would benefit from recent or factual information, use web search
        to inform your response. Always keep your answers brief, conversational, and in your usual style.
        Don't include citations or attribution to sources in your responses, unless asked directly.

        Recent conversation context (last 20 messages):
        {recent_context}
        {rag_section}
        If you're responding to the following author, these pieces of information about them may help craft a better answer:
        {COMMUNITY_MEMBER_INFO}
        """

        # Generate the response using the Responses API with model-specific parameters
        params = create_openai_params("gpt-4.1", prompt)
        response = client.responses.create(**params)
        bot_response = response.output_text  # Simplified access to response text

    # Try to post as a direct reply
    try:
        post_response_with_retry(
            wave_id=drop.wave_id,
            content=bot_response,
            jwt_token=jwt_token,
            reply_to_id=drop.id,
            reply_to_part_id=0
        )
        print(f"Successfully replied to mention in drop {drop.serial_no}.")
        return True
    except requests.exceptions.HTTPError as e:
        # Handle case where drop doesn't exist in the API
        error_detail = ""
        try:
            error_detail = e.response.json()
        except Exception:
            error_detail = e.response.text if hasattr(e.response, 'text') else ""

        # If the error is about an invalid drop ID, consider it successfully handled
        if "doesn't exist" in str(error_detail) or "Invalid reply" in str(error_detail):
            print(f"Drop {drop.serial_no} doesn't exist in API, marking as handled.")
            return True
        else:
            # For other errors, propagate the exception
            raise e

def get_last_50_drops(wave_id):
    """
    Fetches the last 50 drops from the given wave, ordered by creation time.
    """
    drops = Drop.query.filter(Drop.wave_id == wave_id).order_by(Drop.created_at.desc()).limit(50).all()
    drops.reverse()  # Ensure they're in chronological order
    return drops

def select_drops_to_respond_to(non_bot_drops, all_recent_drops):
    """
    Uses the LLM to intelligently select 1-2 most relevant drops to respond to.

    Args:
        non_bot_drops: List of drops not from the bot, candidates for response
        all_recent_drops: All recent drops for conversation context

    Returns:
        List of drop objects selected by the LLM for response
    """
    # Prepare detailed data about each drop for LLM analysis
    drop_details = []
    for i, drop in enumerate(non_bot_drops[:10]):  # Limit to 10 candidates for LLM context
        reply_to = None
        if drop.reply_to_id:
            for recent_drop in all_recent_drops:
                if recent_drop.id == drop.reply_to_id:
                    reply_to = f"Reply to '{recent_drop.author}: {recent_drop.content[:50]}...' (ID: {recent_drop.id})"
                    break

        drop_details.append({
            "index": i,
            "id": drop.id,
            "serial_no": drop.serial_no,
            "author": drop.author,
            "content": drop.content,
            "reply_to": reply_to or "Not a reply"
        })

    # Create conversation context for LLM
    conversation_context = "\n".join([f"{drop.author}: {drop.content}" for drop in all_recent_drops])

    # Create prompt for the LLM
    prompt_text = f"""
    You are a social media assistant analyzing a conversation to determine which messages should be responded to.

    Here's the recent conversation for context:
    {conversation_context}. The author is followed by the colon in each case.

    Below are candidate messages that could be responded to. Your task is to select 1-2 messages that are most worthy of a response:

    {json.dumps(drop_details, indent=2)}

    Select 1-2 messages that are most worthy of a response based on the following criteria:
    1. Relevance to ongoing discussion
    2. Questions or statements that seem to invite a response
    3. Avoid selecting multiple messages from the same author
    4. Prioritize more recent messages if relevant
    5. Consider conversation flow and context
    6. Prioritize messages that contain questions about factual information, current events, or topics that might benefit from web search

    If you're responding to the following people, these pieces of information about them may help craft a better answer:
    {COMMUNITY_MEMBER_INFO}

    Response format: Return a JSON array. It should contain ONLY the "id" values of the 1-2 selected messages. Example: ["message-id-1", "message-id-2"] or ["message-id-1"]
    """

    try:
        # Call the LLM to make the selection
        params = create_openai_params("gpt-4.1", prompt_text)
        response = client.responses.create(**params)
        response_text = response.output_text.strip()

        # Extract the JSON array from the response using our utility function
        selected_ids, success = extract_json_from_text(response_text)

        if success and isinstance(selected_ids, list):
            # Find the drop objects that match the selected IDs
            selected_drops = [drop for drop in non_bot_drops if drop.id in selected_ids]

            print(f"LLM selected {len(selected_drops)} drop[s] to respond to: {[drop.id for drop in selected_drops]}")
            return selected_drops
        else:
            print("Could not parse LLM response as JSON array")
            print(f"LLM response: {response_text}")
            # Fallback to the first drop if parsing fails
            return [non_bot_drops[0]] if non_bot_drops else []
    except Exception as e:
        print(f"Error selecting drops to respond to: {e}")
        # Fallback to the first drop if an error occurs
        return [non_bot_drops[0]] if non_bot_drops else []

def is_reply_to_bot(drop):
    """
    Walk up the reply chain and detect if any ancestor was authored by the bot.
    If a stub is encountered, attempt to hydrate it from the API.
    """
    seen_ids = set()
    current = drop

    while current.reply_to_id and current.reply_to_id not in seen_ids:
        seen_ids.add(current.reply_to_id)
        parent = db.session.get(Drop, current.reply_to_id)

        if drop.author.lower() == BOT_HANDLE.lower():
            return False

        if not parent:
            print(f"Parent drop {current.reply_to_id} not found.")
            return False

        print(f"Checking ancestor drop {parent.serial_no} by {parent.author} (drop id: {parent.id})")

        # If it's a stub with Unknown author, hydrate it
        if parent.author == "Unknown":
            print(f"Attempting to hydrate stub drop {parent.id}")
            jwt_token = authenticate()  # Fetch fresh token if needed
            hydrated = hydrate_drop_from_api(parent.id, jwt_token)
            if hydrated:
                parent = db.session.get(Drop, parent.id)
                print(f"Hydrated drop {parent.id} now shows author: {parent.author}")
            else:
                print(f"Failed to hydrate drop {parent.id}")
                return False

        if parent.author and parent.author.lower() == BOT_HANDLE.lower():
            print(f"Found ancestor drop by bot at serial {parent.serial_no}")
            return True

        current = parent

    print(f"No bot-authored ancestor found for drop {drop.serial_no}")
    return False

def hydrate_drop_from_api(drop_id, jwt_token):
    """Fetch full drop from API and update existing stub in DB."""
    try:
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/json"
        }
        url = f"{BASE_URL}/drops/{drop_id}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        drop_data = response.json()

        # Extract details
        author = drop_data.get("author", {}).get("handle", "Unknown")
        content_parts = drop_data.get("parts", [])
        content = "\n".join((part.get("content") or "") for part in content_parts).strip()

        drop = db.session.get(Drop, drop_id)
        if drop:
            drop.author = author
            drop.content = content
            drop.serial_no = drop_data.get("serial_no", 0)
            drop.created_at = extract_field_value(drop_data, "created_at")
            db.session.commit()
            return True
    except Exception as e:
        print(f"Error hydrating drop {drop_id}: {e}")
    return False

def generate_general_response(drops_text, wave_id=None):
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

    # Retrieve relevant past conversations via RAG
    rag_section = ""
    if wave_id:
        relevant_history = retrieve_relevant_history(drops_text, wave_id, n_results=20)
        if relevant_history:
            rag_section = f"""

    RELEVANT PAST CONVERSATIONS (from the bar's history -- use as background knowledge, not as things happening now):
    {relevant_history}
    """

    prompt = f"""
    Here's a recent conversation from the Memes-Chat wave:

    {drops_text}

    Craft a single brief, insightful response that contributes to the conversation.
    Do NOT prefix your response with your name or any labels.
    Just write the response directly as if you're speaking in the chat.

    Never use the words vibe or existential.

    If the conversation would benefit from recent or factual information, use web search
    to inform your response. Always keep your answers brief, conversational, and in the
    style of the chat - no citations or attributions needed.
    {rag_section}
    If you're responding to the following people, these pieces of information about them may help craft a better answer:
    {COMMUNITY_MEMBER_INFO}
    """

    params = create_openai_params("gpt-4.1", prompt)
    response = client.responses.create(**params)
    bot_response = response.output_text  # Simplified access to response text

    return bot_response

def fetch_user_by_handle(jwt_token, handle):
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


@retry_api_call(max_retries=3, backoff_factor=1.5)
def post_response_with_retry(wave_id, content, jwt_token, reply_to_id=None, reply_to_part_id=0):
    """Post a response to the wave with retry capability"""
    # Construct payload with only the required fields
    payload = {
        "wave_id": wave_id,
        "drop_type": "CHAT",
        "parts": [
            {
                "content": content
            }
        ]
    }

    # Add reply_to information if provided
    if reply_to_id:
        payload["reply_to"] = {
            "drop_id": reply_to_id,
            "drop_part_id": reply_to_part_id
        }
        print(f"Posting reply to drop {reply_to_id}")
    else:
        print("Posting new message")

    url = f"{BASE_URL}/drops"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Try to get more detailed error information
        error_detail = ""
        try:
            error_detail = e.response.json()
        except Exception:
            error_detail = e.response.text if hasattr(e.response, 'text') else "No error details available"

        print(f"HTTP Error: {e}")
        print(f"Error details: {error_detail}")
        raise

def post_general_response(wave_id, content, jwt_token):
    """Post a general response to the wave (legacy function, uses post_response_with_retry)"""
    return post_response_with_retry(wave_id, content, jwt_token)

def is_bot_mentioned(drop):
    """
    Checks if your bot is mentioned in the given drop.
    Handles both @bot and @[bot] formats.
    """
    if not drop.content:
        return False

    # Handle both formats: @bot and @[bot]
    mention_pattern = re.compile(rf"(?<!\w)@(\[)?{re.escape(BOT_HANDLE)}(\])?(?!\w)", re.IGNORECASE)
    return bool(mention_pattern.search(drop.content))

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
        Create a brief, engaging response. If the message contains questions about current facts, events,
        or information that might benefit from up-to-date knowledge, use web search to inform your response.
        Keep your answer conversational and in the style of the community chat.
        Don't include citations or attribution to sources in your responses, unless asked directly.
        """

        # OpenAI Responses API call with gpt-4.1
        params = create_openai_params("gpt-4.1", prompt)
        response = client.responses.create(**params)

        # Get the response text
        bot_response = response.output_text

        # Create the drop payload
        drop_payload = {
            "wave_id": wave_id,
            "drop_type": "CHAT",
            "parts": [{"content": bot_response}],
            "reply_to": {
                "drop_id": drop.id,
                "drop_part_id": 0
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


def sync_and_monitor_job():
    with app.app_context():
        wave = Wave.query.filter_by(name=WAVE_NAME).first()
        if not wave:
            print(f"Wave '{WAVE_NAME}' not found in database.")
            return

        tracking = WaveTracking.query.filter_by(wave_id=wave.id).first()
        if tracking and tracking.oracle_active:
            print("Oracle protocol active -- skipping bot interaction.")
            return

        print("\n" + "="*80)
        print(f"SYNC AND MONITOR JOB RUNNING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        jwt_token = authenticate()
        if not jwt_token:
            print("Authentication failed. Unable to sync or monitor.")
            return

        # Step 1: Poll for new drops (sync local DB with API)
        print("\nPOLLING API FOR NEW DROPS")
        fetch_new_drops_for_wave(wave.id, jwt_token)

        # Step 2: Monitor for interactions
        print("\nMONITORING FOR INTERACTIONS")
        monitor_memes_chat(jwt_token, wave)

        print("="*80)


# Schedule combined job - polls API and then monitors for interactions
# Run every 1 minute for higher responsiveness
scheduler.add_job(
    func=sync_and_monitor_job,
    trigger=IntervalTrigger(minutes=1, start_date=datetime.now().replace(second=0, microsecond=0)),
    id="sync_and_monitor",
    name="Sync and Monitor Job",
    replace_existing=True
)


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
            existing_drop = db.session.get(Drop, drop_id)
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

        # Process identity if needed
        with db.session.no_autoflush:
            existing_identity = Identity.query.filter_by(handle=author_handle).first()
            if existing_identity is None and author_handle != "Unknown":
                user_info = fetch_user_by_handle(jwt_token, author_handle)
                if user_info:
                    authors_processed += 1
                    author_data = upsert_identity(user_info, author_handle, wave_id)
                    check_object_for_dicts(author_data, f"Identity-{author_handle}")
                else:
                    # Fallback to minimal record
                    minimal_user_info = {
                        'pfp': None,
                        'profile_url': f"https://6529.io/{author_handle}"
                    }
                    authors_processed += 1
                    author_data = upsert_identity(minimal_user_info, author_handle)

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

        # Check if we've already created this drop in the current session
        existing_drop_in_session = next((obj for obj in db.session.new
                                    if isinstance(obj, Drop) and obj.id == drop_id), None)
        if not existing_drop_in_session:
            # Check for dictionaries before adding
            check_object_for_dicts(new_drop, f"Drop-{drop_id}")
            db.session.add(new_drop)
            added_count += 1
        else:
            print(f"Skipping duplicate drop creation for ID: {drop_id}")
            skipped_count += 1

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
    """Post a 'gm' style greeting to the wave to announce bot presence using the LLM."""
    prompt = f"""
    Generate a brief, casual 'gm' (good morning) style greeting to announce your presence in a chat.
    Make it friendly, slightly memetic, and under 100 characters.
    This is your first message in the channel, so introduce yourself subtly.
    The name of the channel is {WAVE_NAME}.
    Don't bother mentioning that you now have web search capabilities to help provide up-to-date information. That's expected of everyone chatting anyway.
    Here's some examples, but you should get creative:
    "gmeme, memers. Gray reporting for duty.",
    "gm to all you dive bar denizens!" (eg: for bar-themed waves),
    "gm, wave. bot online."
    or simply: "gm" or "gmeme"
    """

    params = create_openai_params("gpt-4.1", prompt)
    response = client.responses.create(**params)

    bot_response = response.output_text.strip()
    print(f"Generated gm message: {bot_response}")
    post_general_response(wave_id, bot_response, jwt_token)


def check_for_unhandled_interactions(wave_id, jwt_token):
    """Check for recent unhandled mentions and direct replies and process up to 2 of each."""
    print("Checking for any unhandled interactions from previous sessions...")

    # Query for recent drops to check for mentions and replies
    recent_drops = Drop.query.filter_by(wave_id=wave_id).order_by(Drop.created_at.desc()).limit(100).all()
    print(f"Found {len(recent_drops)} recent drops to check")

    # Filter for unhandled mentions
    unhandled_mentions = []
    for drop in recent_drops:
        is_explicit_mention = is_bot_mentioned(drop)
        if is_explicit_mention and not getattr(drop, 'bot_replied_to', False):
            unhandled_mentions.append(drop)
            print(f"  Added drop {drop.serial_no} as unhandled mention")

    # Filter for unhandled direct replies to bot's messages
    unhandled_replies = []
    for drop in recent_drops:
        if drop.reply_to_id:
            parent_drop = db.session.get(Drop, drop.reply_to_id)
            if parent_drop and parent_drop.author.lower() == BOT_HANDLE.lower() and not getattr(drop, 'bot_replied_to', False):
                # Only add if it's not already in the mentions list (to avoid duplicate responses)
                if drop not in unhandled_mentions:
                    unhandled_replies.append(drop)
                    print(f"  Added drop {drop.serial_no} as unhandled direct reply")

    print(f"Found {len(unhandled_mentions)} unhandled mentions and {len(unhandled_replies)} unhandled direct replies to process")

    # Sort by created_at desc, then serial_no desc as fallback
    unhandled_mentions.sort(key=lambda d: (d.created_at or datetime.min, d.serial_no), reverse=True)
    unhandled_replies.sort(key=lambda d: (d.created_at or datetime.min, d.serial_no), reverse=True)

    # Process up to 1 most recent unhandled mention
    mentions_count = 0
    if unhandled_mentions:
        mentions_to_process = unhandled_mentions[:1]
        print(f"Processing {len(mentions_to_process)} unhandled mentions from previous sessions")
        for drop in mentions_to_process:
            print(f"Generating reply to previously unhandled mention in drop {drop.serial_no}...")
            try:
                reply_result = reply_to_mention(drop, jwt_token)
                if reply_result:
                    # Only mark as replied to after successful API call
                    drop.bot_replied_to = True
                    db.session.commit()
                    print(f"Drop {drop.serial_no} marked as replied to after successful API call")
                    mentions_count += 1
                else:
                    print(f"Failed to reply to drop {drop.serial_no}, not marking as replied to")
            except Exception as e:
                db.session.rollback()
                print(f"Error replying to mention or updating DB for drop {drop.serial_no}: {e}")

        # Log if we skipped any mentions
        if len(unhandled_mentions) > 2:
            print(f"Skipped {len(unhandled_mentions) - 2} older unhandled mentions to avoid spamming.")
    else:
        print("No unhandled mentions found from previous sessions.")

    # Process up to 1 most recent unhandled direct reply
    replies_to_process = unhandled_replies[:1]
    if replies_to_process:
        print(f"Processing {len(replies_to_process)} unhandled direct replies from previous sessions")
        for drop in replies_to_process:
            print(f"Generating reply to previously unhandled direct reply in drop {drop.serial_no}...")
            try:
                if reply_to_mention(drop, jwt_token):
                    drop.bot_replied_to = True
                    db.session.commit()
                    print(f"Drop {drop.serial_no} marked as replied to after successful API call")
            except Exception as e:
                db.session.rollback()
                print(f"Error replying to direct reply or updating DB for drop {drop.serial_no}: {e}")
    else:
        print("No unhandled direct replies found from previous sessions.")

    return mentions_count + len(replies_to_process)


GAP_RETRY_LIMIT = 1000  # how many serials we probe downward before giving up

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
            drops_data = resp.json()
            print(f"API returned {len(drops_data)} drops")
            return drops_data
        except HTTPError as err:
            # For "less than" queries, we handle gaps by stepping down
            if err.response.status_code == 404 and "serial_no_less_than" in params:
                tries += 1
                if tries > GAP_RETRY_LIMIT:
                    print(f"Gap retry limit hit at serial {params['serial_no_less_than']}. Stopping page.")
                    return []
                # step down one serial and retry
                params["serial_no_less_than"] -= 1
                continue
            # For other errors, including "greater than" queries, just log and return empty
            else:
                print(f"API error fetching drops: {err}")
                return []
        except Exception as e:
            print(f"Unexpected error in safe_get_drops: {e}")
            return []


def activity_check_job():
    """
    Job that runs every 2 hours to check for inactivity and post a message if:
    1. No new activity since last check
    2. Last message is from someone other than the bot
    """
    global LAST_GM_DATE

    print("\n" + "="*80)
    print(f"ACTIVITY CHECK JOB RUNNING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with app.app_context():
        jwt_token = authenticate()
        if not jwt_token:
            print("Authentication failed. Unable to check activity.")
            return

        # Get the wave
        wave = Wave.query.filter_by(name=WAVE_NAME).first()
        if not wave:
            print(f"Wave '{WAVE_NAME}' not found in database.")
            return

        # Get the last check timestamp from WaveTracking
        wave_tracking = WaveTracking.query.filter_by(wave_id=wave.id).first()
        if not wave_tracking:
            print("Wave tracking not found.")
            return

        # Get the 20 most recent drops for context
        recent_drops = Drop.query.filter_by(wave_id=wave.id).order_by(Drop.created_at.desc()).limit(20).all()
        if not recent_drops:
            print("No drops found in the wave.")
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
                print(f"No new activity since last check and last message is from {latest_drop.author}. Analyzing conversation...")

                # Format recent conversation for the LLM
                conversation_history = "\n".join([
                    f"{drop.author}: {drop.content}"
                    for drop in recent_drops
                ])

                # Retrieve relevant past conversations via RAG
                activity_rag_history = retrieve_relevant_history(conversation_history, wave.id, n_results=20)
                activity_rag_section = ""
                if activity_rag_history:
                    activity_rag_section = f"""
                RELEVANT PAST CONVERSATIONS (from the bar's history -- use as background knowledge, not as things happening now):
                {activity_rag_history}
                """

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
                You are a bot named {BOT_HANDLE} in a chat wave called "{WAVE_NAME}".
                The conversation has been inactive for a while, and you need to post a message to keep it active.

                CONTEXT INFORMATION:
                - Current date and time: {current_time.strftime("%Y-%m-%d %H:%M")} ({time_of_day})
                - Last "gm" message sent: {last_gm_time_str}
                - Time since last message: {time_since_last_message_hours:.1f} hours
                - Last message was from: {latest_drop.author}

                RECENT CONVERSATION HISTORY (most recent 20 messages):
                {conversation_history}
                {activity_rag_section}
                Based on this context, determine the most appropriate type of message to send:
                1. A direct clever reply to the last message
                2. A random thought related to the conversation themes
                3. A once-a-day greeting, like "gm wavers", or "gmeme, memers", or "gm to all you dive bar denizens"... make sure it's relevant to the Wave {WAVE_NAME}!

                Choose the most natural and context-appropriate option. If the last message asks a question or seems to invite a response, prefer a direct reply.
                If the conversation has been dormant for a long time, a greeting or new topic might be better.
                Only use a greeting like "gm" if it hasn't been used today. you can gm at any time of day.

                Never use the words vibe or existential.

                IMPORTANT: You also have web search capabilities! If the conversation involves questions about current events, facts,
                or topics that would benefit from up-to-date information, references to recent events or new releases, use web search to inform your response. Don't mention that
                you're using web search - just incorporate the information naturally into your message.

                If you're responding to the following people, these pieces of information about them may help craft a better answer:
                {COMMUNITY_MEMBER_INFO}

                FORMAT YOUR RESPONSE LIKE THIS:
                Message type: [REPLY, THOUGHT, or GREETING]
                Message: [Your actual message text]
                Reasoning: [Brief explanation of why you chose this type]
                """

                params = create_openai_params("gpt-4.1", prompt)
                response = client.responses.create(**params)

                llm_response = response.output_text

                # Parse the LLM response using our utility function
                expected_keys = ["message_type", "message", "reasoning"]
                parsed_response = parse_structured_llm_response(llm_response, expected_keys)

                # Extract the values from the parsed response
                message_type = parsed_response["message_type"].upper() if parsed_response["message_type"] else None
                message_content = parsed_response["message"]
                reasoning = parsed_response["reasoning"]

                if not message_type or not message_content:
                    print("Failed to parse LLM response. Not sending any message.")
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
                    print(f"Posted {message_type} message: {message_content}")
                except Exception as e:
                    print(f"Error posting message: {e}")
            else:
                print(f"Last message is from the bot ({BOT_HANDLE}). Skipping activity generation.")
        else:
            print(f"There has been activity since the last check. No need to generate a message.")

        # Update the last activity check time
        wave_tracking.last_activity_check = datetime.now()
        db.session.commit()

    print("="*80)

# Schedule the activity check job to run every 2 hours
scheduler.add_job(
    func=activity_check_job,
    trigger=IntervalTrigger(hours=2, start_date=(datetime.now().replace(minute=11, second=0, microsecond=0))),
    id="activity_check",
    name="Activity Check Job",
    replace_existing=True
)


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

        # Force exact name match
        exact_match = next((w for w in waves if w["name"].strip().lower() == wave_name.strip().lower()), None)
        if not exact_match:
            print(f"No exact match found for wave name: '{wave_name}'")
            return None

        wave_data = exact_match
        print(f"Found exact wave: {wave_data['name']}")

        existing_wave = db.session.get(Wave, wave_data["id"])
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


# Check if running migrations or any flask db command
running_migrations = len(sys.argv) >= 2 and sys.argv[0].endswith('flask') and sys.argv[1] == 'db'

# Register shutdown handler for scheduler
atexit.register(lambda: scheduler.shutdown() if scheduler.running else None)

# Initialize app and load data (but skip during migrations)
with app.app_context():
    if running_migrations:
        print("Database migration detected - skipping initialization and data loading")
    else:
        # Start the scheduler for all non-migration runs, unless explicitly skipped
        skip_scheduler = os.environ.get('SKIP_SCHEDULER', '').lower() == 'true'
        if not scheduler.running and not skip_scheduler:
            scheduler.start()
            print("Scheduler started - background jobs are now running")
        elif skip_scheduler:
            print("Scheduler start skipped due to SKIP_SCHEDULER environment variable")

        # Only load data when not in migration mode
        jwt_token = authenticate()
        if jwt_token:
            wave = setup_chosen_wave(jwt_token, WAVE_NAME)
            if wave:
                drop_count = Drop.query.filter_by(wave_id=wave.id).count()
                fetch_new_drops_for_wave(wave.id, jwt_token)

                # Check if ChromaDB needs a bulk embedding catch-up
                chroma_count = chroma_collection.count()
                db_count = Drop.query.filter(
                    Drop.wave_id == wave.id,
                    Drop.content.isnot(None),
                    Drop.content != "",
                ).count()
                gap = db_count - chroma_count
                print(f"ChromaDB has {chroma_count} drops, DB has {db_count} embeddable drops (gap: {gap})")
                if gap > 100:
                    print(f"ChromaDB is {gap} drops behind -- running bulk embedding...")
                    embed_all_existing_drops(wave.id)

                if drop_count == 0:
                    print("First time loading drops for this wave. Posting a gm message.")
                    post_gm_message(wave.id, jwt_token)
                else:
                    # Check for any unhandled mentions or direct replies that we might have missed
                    check_for_unhandled_interactions(wave.id, jwt_token)
                print("Initial data loaded. Scheduled jobs will handle ongoing updates.")


# ---- keep the script alive so APScheduler jobs can run ----
def main():
    """Main entry point when running as a script"""
    print(f"{BOT_HANDLE} bot is up. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down...")

if __name__ == "__main__":
    main()
