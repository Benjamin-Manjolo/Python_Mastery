from fastapi import FastAPI
from app.routes import todo

app = FastAPI()

app.include_router(todo.router)

@app.get("/")
def root():
    return {"message":"welcome to my todo's"}