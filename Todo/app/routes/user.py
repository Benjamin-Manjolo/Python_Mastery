from fastapi import APIRouter ,Depends, HTTPException
from sqlalchemy.orm import Session
from app import models,schemas,auth
from app.deps import get_db


router = APIRouter(prefix="/users",tags=["Users"])

@router.post("/register")
def register (user:schemas.UserCreate,db:Session = Depends(get_db)):
    hashed = auth.hash_password(user.password)

    db_user = models.User(email=user.email.email,password=hashed)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return{"message":"User registered successfully"}

@router.post("/login")
def login(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()

    if not db_user or not auth.verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth.create_token({"user_id": db_user.id})
    return {"access_token": token}