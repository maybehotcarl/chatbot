# =======================
#       MODELS
# =======================

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone, timedelta
# Initialize SQLAlchemy
db = SQLAlchemy()
# Import your models so that they are registered with SQLAlchemy.
# import models

class Wave(db.Model):
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String)
    author = db.Column(db.String)
    serial_no = db.Column(db.Integer)

    def __repr__(self):
        return f"<Wave {self.name}>"

# Drop model
class Drop(db.Model):
    id = db.Column(db.String, primary_key=True)
    wave_id = db.Column(db.String, db.ForeignKey('wave.id'))
    author = db.Column(db.String)
    content = db.Column(db.Text)
    serial_no = db.Column(db.Integer)
    created_at = db.Column(db.DateTime)
    reply_to_id = db.Column(db.String, db.ForeignKey('drop.id'), nullable=True)
    # replies = db.relationship('Drop', backref=db.backref('parent', remote_side=[id]), lazy='dynamic')
    wave = db.relationship('Wave', backref=db.backref('drops', lazy=True))
    parent = db.relationship('Drop', remote_side=[id], backref='replies', foreign_keys=[reply_to_id])

    def __repr__(self):
        return f"<Drop {self.id}>"

# Author model
class Author(db.Model):
    __tablename__ = 'authors'
    id = db.Column(db.String)
    handle = db.Column(db.String, unique=True, index=True, nullable=False, primary_key=True)
    normalized_handle = db.Column(db.String, index=True, nullable=True)
    pfp = db.Column(db.String)
    cic = db.Column(db.Integer)
    rep = db.Column(db.Integer)
    level = db.Column(db.Integer)
    tdh = db.Column(db.Integer)
    display = db.Column(db.String)
    primary_wallet = db.Column(db.String)
    pfp_url = db.Column(db.String, nullable=True)
    profile_url = db.Column(db.String, nullable=True)

    def __repr__(self):
        return f"<Author {self.handle}>"

class WaveTracking(db.Model):
    __tablename__ = "wave_tracking"
    id = db.Column(db.Integer, primary_key=True)
    wave_id = db.Column(db.String, db.ForeignKey('wave.id'), nullable=False)
    last_processed_serial_no = db.Column(db.Integer, default=0, nullable=False)
    accumulated_new_drops = db.Column(db.Integer, default=0, nullable=False)
    wave = db.relationship("Wave", backref=db.backref("tracking", lazy=True))

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
    