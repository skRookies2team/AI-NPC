from fastapi import FastAPI
from app.api.endpoints import training, character, chat, update, sessions

def register_routes(app: FastAPI):
    """모든 API 라우트 등록"""

    # Training endpoints
    app.add_api_route(
        "/api/ai/train-from-s3",
        training.train_novel_from_s3,
        methods=["POST"]
    )
    app.add_api_route(
        "/api/ai/train-text",
        training.train_novel_from_text,
        methods=["POST"]
    )

    # Character endpoint
    app.add_api_route(
        "/api/ai/character",
        character.set_character,
        methods=["POST"]
    )

    # Chat endpoint
    app.add_api_route(
        "/api/ai/chat",
        chat.chat,
        methods=["POST"]
    )

    # Update endpoint
    app.add_api_route(
        "/api/ai/update",
        update.update_novel_content,
        methods=["POST"]
    )

    # Session endpoints
    app.add_api_route(
        "/api/ai/sessions",
        sessions.list_sessions,
        methods=["GET"]
    )
    app.add_api_route(
        "/api/ai/session/{session_id}",
        sessions.get_session_info,
        methods=["GET"]
    )
