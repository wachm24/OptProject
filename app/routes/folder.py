from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from bson import ObjectId
from bson.errors import InvalidId
from datetime import datetime, timezone, timedelta
from config.db import notes_collection, folders_collection
from schemas.note import noteEntity
from routes.deps import get_current_user

folder = APIRouter()
templates = Jinja2Templates(directory="templates")

IST = timezone(timedelta(hours=5, minutes=30))


def require_user(request: Request):
    user = get_current_user(request)
    if not user:
        return None, RedirectResponse("/login", status_code=303)
    return user, None


# GET - lista wszystkich folderów
@folder.get("/folders", response_class=HTMLResponse)
async def get_folders(request: Request):
    user, redirect = require_user(request)
    if redirect:
        return redirect

    docs = folders_collection.find({"user_id": user["id"]})
    all_folders = []
    for f in docs:
        note_count = notes_collection.count_documents({
            "folder_id": str(f["_id"]),
            "user_id": user["id"],
        })
        all_folders.append({
            "_id": str(f["_id"]),
            "name": f.get("name", ""),
            "note_count": note_count,
        })
    return templates.TemplateResponse("folders.html", {
        "request": request,
        "folders": all_folders,
        "user": user,
    })


# POST - utwórz nowy folder
@folder.post("/folders")
async def create_folder(request: Request, name: str = Form(...)):
    user, redirect = require_user(request)
    if redirect:
        return redirect

    created_at = datetime.now(IST)
    folders_collection.insert_one({
        "name": name,
        "created_at": created_at,
        "user_id": user["id"],
    })
    return RedirectResponse("/folders", status_code=303)


# GET - widok pojedynczego folderu
@folder.get("/folders/{folder_id}", response_class=HTMLResponse)
async def get_folder(request: Request, folder_id: str):
    user, redirect = require_user(request)
    if redirect:
        return redirect

    try:
        obj_id = ObjectId(folder_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid folder ID")

    folder_doc = folders_collection.find_one({"_id": obj_id, "user_id": user["id"]})
    if not folder_doc:
        raise HTTPException(status_code=404, detail="Folder not found")

    docs = notes_collection.find({"folder_id": folder_id, "user_id": user["id"]})
    folder_notes = [noteEntity(item) for item in docs]

    folder_data = {"_id": folder_id, "name": folder_doc.get("name", "")}
    return templates.TemplateResponse(
        "folder_detail.html",
        {"request": request, "folder": folder_data, "notes": folder_notes, "user": user}
    )


# POST - dodaj notatkę do folderu
@folder.post("/folders/{folder_id}/add")
async def add_note_to_folder(
    request: Request,
    folder_id: str,
    title: str = Form(...),
    desc: str = Form(...),
    important: bool = Form(False),
):
    user, redirect = require_user(request)
    if redirect:
        return redirect

    try:
        ObjectId(folder_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid folder ID")

    created_at = datetime.now(IST)
    notes_collection.insert_one({
        "title": title,
        "desc": desc,
        "important": important,
        "folder_id": folder_id,
        "created_at": created_at,
        "updated_at": created_at,
        "user_id": user["id"],
    })
    return RedirectResponse(f"/folders/{folder_id}", status_code=303)


# POST - usuń folder
@folder.post("/folders/{folder_id}/delete")
async def delete_folder(request: Request, folder_id: str):
    user, redirect = require_user(request)
    if redirect:
        return redirect

    try:
        obj_id = ObjectId(folder_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid folder ID")

    folders_collection.delete_one({"_id": obj_id, "user_id": user["id"]})
    notes_collection.update_many(
        {"folder_id": folder_id, "user_id": user["id"]},
        {"$set": {"folder_id": None}}
    )
    return RedirectResponse("/folders", status_code=303)