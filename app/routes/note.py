from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from bson import ObjectId
from bson.errors import InvalidId
from datetime import datetime, timezone, timedelta
from typing import Optional
from config.db import notes_collection, folders_collection
from schemas.note import noteEntity, notesEntity
from routes.deps import get_current_user

note = APIRouter()
templates = Jinja2Templates(directory="templates")

IST = timezone(timedelta(hours=5, minutes=30))


def require_user(request: Request):
    """Zwraca usera lub redirectuje do /login."""
    user = get_current_user(request)
    if not user:
        return None, RedirectResponse("/login", status_code=303)
    return user, None


# GET all notes (HTML)
@note.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    user, redirect = require_user(request)
    if redirect:
        return redirect

    docs = notes_collection.find({"user_id": user["id"]})
    all_notes = notesEntity(docs)

    folder_docs = folders_collection.find({"user_id": user["id"]})
    all_folders = [{"_id": str(f["_id"]), "name": f.get("name", "")} for f in folder_docs]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "newDocs": all_notes,
        "folders": all_folders,
        "user": user,
    })


# POST new note
@note.post("/")
def add_note(
    request: Request,
    title: str = Form(...),
    desc: str = Form(...),
    important: bool = Form(False),
    folder_id: Optional[str] = Form(None),
):
    user, redirect = require_user(request)
    if redirect:
        return redirect

    created_at = datetime.now(IST)
    notes_collection.insert_one({
        "title": title,
        "desc": desc,
        "important": important,
        "created_at": created_at,
        "updated_at": created_at,
        "folder_id": folder_id if folder_id else None,
        "user_id": user["id"],
    })
    return RedirectResponse("/", status_code=303)


# GET edit note page
@note.get("/edit/{note_id}", response_class=HTMLResponse)
async def edit_note_page(request: Request, note_id: str):
    user, redirect = require_user(request)
    if redirect:
        return redirect

    try:
        obj_id = ObjectId(note_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid note ID")

    # Sprawdź że notatka należy do tego usera
    note_doc = notes_collection.find_one({"_id": obj_id, "user_id": user["id"]})
    if not note_doc:
        raise HTTPException(status_code=404, detail="Note not found")

    folder_docs = folders_collection.find({"user_id": user["id"]})
    all_folders = [{"_id": str(f["_id"]), "name": f.get("name", "")} for f in folder_docs]

    return templates.TemplateResponse(
        "edit_note.html",
        {"request": request, "note": noteEntity(note_doc), "folders": all_folders, "user": user}
    )


# POST update note
@note.post("/edit/{note_id}")
async def update_note(
    request: Request,
    note_id: str,
    title: str = Form(...),
    desc: str = Form(...),
    important: bool = Form(False),
    folder_id: Optional[str] = Form(None),
):
    user, redirect = require_user(request)
    if redirect:
        return redirect

    try:
        obj_id = ObjectId(note_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid note ID")

    updated_at = datetime.now(IST)
    update_result = notes_collection.update_one(
        {"_id": obj_id, "user_id": user["id"]},
        {"$set": {
            "title": title,
            "desc": desc,
            "important": important,
            "updated_at": updated_at,
            "folder_id": folder_id if folder_id else None,
        }}
    )

    if update_result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")

    return RedirectResponse("/", status_code=303)


# DELETE note
@note.post("/delete/{note_id}")
async def delete_note(request: Request, note_id: str):
    user, redirect = require_user(request)
    if redirect:
        return redirect

    try:
        obj_id = ObjectId(note_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid note ID")

    result = notes_collection.delete_one({"_id": obj_id, "user_id": user["id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")

    return RedirectResponse("/", status_code=303)