import cv2
import base64
import asyncio
import websockets
import json
import time

# --- EDGE CONFIGURATION ---
CONTROL_PLANE_URL = "ws://localhost:8080/live/sensor"
NODE_IDENTITY_TOKEN = "edge-device-secret-123"
CAMERA_INDEX = 0
FPS_THROTTLE = 1  # 1 frame per second to avoid completely destroying the API server
TRIPWIRE = "Is there a person carrying a generic weapon or looking directly into the camera?"

async def edge_sensor_loop():
    print(f"[*] Booting Edge Node Sensor at {FPS_THROTTLE} FPS...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("[!] FATAL: Physical camera manifold strictly unavailable.")
        return

    headers = {"X-Node-Token": NODE_IDENTITY_TOKEN}

    try:
        async with websockets.connect(CONTROL_PLANE_URL, additional_headers=headers) as ws:
            print(f"[+] Uplink established with Control Plane -> {CONTROL_PLANE_URL}")
            
            # Authenticate stream and configure Tripwire natively
            handshake = json.dumps({"type": "init", "tripwire": TRIPWIRE})
            await ws.send(handshake)
            print("[-] Sent initial edge configuration.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[!] Frame drop. Retrying...")
                    await asyncio.sleep(1)
                    continue
                
                # Compress strictly to limit bandwidth
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                payload = json.dumps({
                    "type": "frame",
                    "frame": frame_b64
                })
                
                await ws.send(payload)
                print("[-] Pushed frame tensor to orchestrator. Awaiting inference...")
                
                # Await evaluation
                response = await ws.recv()
                data = json.loads(response)
                
                if data.get("status") == "success":
                    trigger = data["inference"].get("triggered", False)
                    reason = data["inference"].get("reason", "N/A")
                    print(f"   >>> EVAL: {'🔴 THREAT' if trigger else '🟢 SECURE'} | {reason}")
                else:
                    print(f"   >>> ERROR: {data}")
                
                time.sleep(1.0 / FPS_THROTTLE)

    except websockets.exceptions.ConnectionClosed:
        print("[!] Subspace uplink severed by Control Plane.")
    except Exception as e:
        print(f"[!] Critical Edge Node Failure: {e}")
    finally:
        cap.release()
        print("[*] Sensor teardown complete.")

if __name__ == "__main__":
    asyncio.run(edge_sensor_loop())
