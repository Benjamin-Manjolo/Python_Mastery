from fastapi import Header, HTTPException
from app.database import SessionLocal
from app.auth import SECRET_KEY, ALGORITHM, jwt


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(x_token: str = Header()):
    try:
        payload = jwt.decode(x_token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["user_id"]
    except:
        raise HTTPException(status_code=401, detail="Invalid token")