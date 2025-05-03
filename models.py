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
    bot_replied_to_mention = Column(Boolean, default=False)  # Track if bot replied to mention
    # replies = db.relationship('Drop', backref=db.backref('parent', remote_side=[id]), lazy='dynamic')
    wave = db.relationship('Wave', backref=db.backref('drops', lazy=True))
    parent = db.relationship('Drop', remote_side=[id], backref='replies', foreign_keys=[reply_to_id])

    def __repr__(self):
        return f"<Drop {self.id} by {self.author}>"

# Author model
class Author(db.Model):
    __tablename__ = 'authors'
    id = Column(String, primary_key=True)
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
        return f"<Author {self.handle}>"

class WaveTracking(db.Model):
    __tablename__ = 'wave_tracking'
    wave_id = Column(String, ForeignKey('wave.id'), primary_key=True)
    last_processed_serial_no = Column(Integer, default=0)
    last_interaction_serial_no = Column(Integer, default=0)  # Tracks last interaction check
    accumulated_new_drops = Column(Integer, default=0)
    last_activity_check = Column(DateTime)
    wave = db.relationship("Wave", backref=db.backref("tracking", lazy=True))

    def __repr__(self):
        return f"<WaveTracking {self.wave_id}>"

class Identity(db.Model):
    __tablename__ = 'identities'
    id = db.Column(db.String, primary_key=True)  # API provided id
    handle = db.Column(db.String, unique=True, index=True, nullable=False)
    normalised_handle = db.Column(db.String, index=True)
    pfp = db.Column(db.String)
    primary_wallet = db.Column(db.String)
    rep = db.Column(db.Integer)
    cic = db.Column(db.Integer)
    level = db.Column(db.Integer)
    tdh = db.Column(db.Integer)
    display = db.Column(db.String)

    def __repr__(self):
        return f"<Identity {self.handle}>"
    