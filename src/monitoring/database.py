import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# Use SQLite by default, can be overridden by env variable for Azure PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///predictions.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Features
    Age = Column(Float)
    Gender = Column(String(50))
    Tenure = Column(Float)
    Usage_Frequency = Column(Float)
    Support_Calls = Column(Float)
    Payment_Delay = Column(Float)
    Subscription_Type = Column(String(50))
    Contract_Length = Column(String(50))
    Total_Spend = Column(Float)
    Last_Interaction = Column(Float)
    
    # Model info
    model_version = Column(String(50), nullable=True)
    
    # Outputs
    prediction = Column(Integer)
    probability = Column(Float, nullable=True)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
