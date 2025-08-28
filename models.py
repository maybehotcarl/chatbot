# =======================
#       MODELS
# =======================

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone, timedelta
from sqlalchemy.sql import func
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, Boolean
# Initialize SQLAlchemy
db = SQLAlchemy()
# Import your models so that they are registered with SQLAlchemy.
# import models

class Wave(db.Model):
    __tablename__ = 'wave'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    author = Column(String)
    serial_no = Column(Integer)

    def __repr__(self):
        return f"<Wave {self.name}>"

# Drop model
class Drop(db.Model):
    __tablename__ = 'drop'
    id = Column(String, primary_key=True)
    wave_id = Column(String, ForeignKey('wave.id'))
    author = Column(String)
    content = Column(Text)
    serial_no = Column(Integer)
    created_at = Column(DateTime)
    reply_to_id = Column(String, ForeignKey('drop.id'), nullable=True)
    bot_replied_to = Column(Boolean, default=False)  # Track if bot has replied to this drop (mention or direct reply)
    wave = db.relationship('Wave', backref=db.backref('drops', lazy=True))
    parent = db.relationship('Drop', remote_side=[id], backref='replies', foreign_keys=[reply_to_id])

    def __repr__(self):
        return f"<Drop {self.id} by {self.author}>"

# Identity model (replaces Author model)
class Identity(db.Model):
    __tablename__ = 'identities'
    id = Column(String, primary_key=True)  # API provided id
    handle = Column(String, unique=True, index=True, nullable=False)
    normalized_handle = Column(String, index=True)
    pfp = Column(String)
    cic = Column(Integer)
    rep = Column(Integer)
    level = Column(Integer)
    tdh = Column(Integer)
    display = Column(String)
    primary_wallet = Column(String)
    pfp_url = Column(String, nullable=True)
    profile_url = Column(String, nullable=True)

    def __repr__(self):
        return f"<Identity {self.handle}>"

class WaveTracking(db.Model):
    __tablename__ = 'wave_tracking'
    wave_id = Column(String, ForeignKey('wave.id'), primary_key=True)
    last_processed_serial_no = Column(Integer, default=0)
    last_interaction_serial_no = Column(Integer, default=0)  # Tracks last interaction check
    accumulated_new_drops = Column(Integer, default=0)
    last_activity_check = Column(DateTime)
    oracle_active = Column(Boolean, default=False)  # 👈 NEW FIELD
    oracle_start_time = Column(DateTime, nullable=True)
    wave = db.relationship("Wave", backref=db.backref("tracking", lazy=True))

    def __repr__(self):
        return f"<WaveTracking {self.wave_id}>"
