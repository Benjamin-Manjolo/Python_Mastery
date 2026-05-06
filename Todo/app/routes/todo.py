from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app import models, schemas
from app.deps import get_db, get_current_user

router = APIRouter(prefix="/todos", tags=["Todos"])


# ✅ CREATE
@router.post("/", response_model=schemas.TodoOut)
def create(
    todo: schemas.TodoCreate,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    new_todo = models.Todo(
        title=todo.title,
        owner_id=user_id
    )
    db.add(new_todo)
    db.commit()
    db.refresh(new_todo)
    return new_todo


# ✅ READ ALL
@router.get("/", response_model=list[schemas.TodoOut])
def get_all(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    return db.query(models.Todo).filter(
        models.Todo.owner_id == user_id
    ).all()


# ✅ READ ONE
@router.get("/{todo_id}", response_model=schemas.TodoOut)
def get_one(
    todo_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    todo = db.query(models.Todo).filter(
        models.Todo.id == todo_id,
        models.Todo.owner_id == user_id
    ).first()

    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")

    return todo


# ✅ UPDATE
@router.put("/{todo_id}", response_model=schemas.TodoOut)
def update(
    todo_id: int,
    updated: schemas.TodoCreate,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    todo = db.query(models.Todo).filter(
        models.Todo.id == todo_id,
        models.Todo.owner_id == user_id
    ).first()

    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")

    todo.title = updated.title
    db.commit()
    db.refresh(todo)

    return todo


# ✅ DELETE
@router.delete("/{todo_id}")
def delete(
    todo_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    todo = db.query(models.Todo).filter(
        models.Todo.id == todo_id,
        models.Todo.owner_id == user_id
    ).first()

    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")

    db.delete(todo)
    db.commit()

    return {"message": "Todo deleted successfully"}