from datetime import datetime
from universal import db
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ds_name = db.Column(db.String(40),unique=True,nullable=False)
    date_uploaded = db.Column(db.DateTime,nullable=False,default=datetime.now())
    data = db.Coloumn(db.LargeBinary,nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'),nullable=False)
    def __repr__(self):
        return f"User('{self.ds_name}', '{self.date_uploaded}','{self.data}','{self.user_id})"