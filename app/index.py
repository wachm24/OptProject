# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# from routes.note import note
# import os
# from routes.folder import folder

# app = FastAPI()

# # Serve static files only if folder is not empty
# if os.path.isdir("static") and os.listdir("static"):
#     app.mount("/static", StaticFiles(directory="static"), name="static")

# # Include routes
# app.include_router(note)
# app.include_router(folder)


# # Health check endpoint
# @app.get("/ping")
# async def ping():
#     return {"message": "Server is running!"}
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from routes.note import note
from routes.folder import folder
from routes.auth import auth
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ["SESSION_SECRET"],
)

if os.path.isdir("static") and os.listdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth)
app.include_router(note)
app.include_router(folder)

@app.get("/ping")
async def ping():
    return {"message": "Server is running!"}