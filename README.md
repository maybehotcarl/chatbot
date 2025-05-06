# 6529 Chatbot

A Flask-based chatbot for interacting with the 6529.io API and participating in wave chats.

## Setup

1. Clone this repository

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following variables:

   ```env
   PRIVATE_KEY=your_ethereum_private_key
   OPENAI_API_KEY=your_openai_api_key
   ```

5. Initialize the database:

   ```bash
   python reset_db.py --force
   ```

## Running the Bot

To start the bot in interactive mode:

   ```bash
   python app.py
   ```

This will:

1. Create the SQLite database if it doesn't exist
2. Connect to the 6529.io API
3. Set up the chosen wave channel (can take a while for large waves, start in `botsville` for testing)
4. Start background jobs for fetching new messages and responding to interactions

## Features

- Responds to direct mentions with AI-generated responses
- Replies to messages that reply to the bot
- Periodically checks for inactivity and sends contextual messages
- Tracks all chat activity in a local SQLite database

## Customization

- Bot personality can be adjusted in the `SYSTEM_PROMPT` variable
- Response temperature can be adjusted with the `TEMPERATURE` variable
- Timing of various jobs can be adjusted in the scheduler setup

## Architecture

- Flask web application with SQLAlchemy ORM
- OpenAI API for generating responses
- Background jobs using APScheduler
- SQLite database for local storage
- 6529.io API for interacting with the wave chat

## Development

For development purposes, you can use the Flask CLI:

   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   flask run
   ```

Database migrations can be managed with:

   ```bash
   flask db migrate -m "Description of changes"
   flask db upgrade
   ```

## Troubleshooting

- If you encounter dictionary serialization errors, check the `extract_field_value` function which handles nested JSON structures.
- For timestamp errors, ensure all date/time values are properly converted to Python datetime objects.
- Authentication issues are often related to the private key format or permissions.

## Contributors

- **Original Author:** [@maybehotcarl](https://github.com/maybehotcarl)
- **AI Engineer:** [@brookr](https://github.com/brookr)
- **Other contributors:** Punk 8164, and many tolerant dive bar denizens
- **AI Assistance:** Significant portions of this project were created and/or refined using AI code tools (including from OpenAI and Anthropic), which contributed to code, documentation, and architectural decisions.
