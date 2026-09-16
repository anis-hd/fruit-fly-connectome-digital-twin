"""WebSocket endpoint: init/ping/control fan-out."""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .control import apply_control, state_dict


def register_websocket(app: FastAPI, rt) -> None:
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await rt.manager.connect(websocket)
        try:
            await websocket.send_json(
                {"type": "init", "state": state_dict(rt)})
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "control":
                    await apply_control(rt, data)
                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            rt.manager.disconnect(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            rt.manager.disconnect(websocket)
