from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from datetime import datetime, timezone, timedelta
import os

from config.db import notes_collection, db

users_collection = db["users"]

auth = APIRouter()
templates = Jinja2Templates(directory="templates")

IST = timezone(timedelta(hours=5, minutes=30))

config = Config(environ=os.environ)
oauth = OAuth(config)

oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=os.environ["GOOGLE_CLIENT_ID"],
    client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
    client_kwargs={"scope": "openid email profile"},
)


@auth.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/")
    return templates.TemplateResponse("login.html", {"request": request})


@auth.get("/auth/google")
async def login_via_google(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@auth.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")
    if not user_info:
        return RedirectResponse("/login")

    # Save/update user info in Mongo
    users_collection.update_one(
        {"_id": user_info["sub"]},
        {
            "$set": {
                "email": user_info["email"],
                "name": user_info.get("name", ""),
                "last_login": datetime.now(IST),
                "note_count": notes_collection.count_documents({"user_id": user_info["sub"]}),
            },
            "$setOnInsert": {
                "created_at": datetime.now(IST),
            }
        },
        upsert=True
    )

    request.session["user"] = {
        "id": user_info["sub"],
        "email": user_info["email"],
        "name": user_info.get("name", ""),
    }
    return RedirectResponse("/")


@auth.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")