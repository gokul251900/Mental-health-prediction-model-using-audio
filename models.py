from sqlalchemy import Column, Integer, String, Text
from database import Base

class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    main_prediction = Column(String)

    predicted_disorders = Column(Text)

    probabilities = Column(Text)