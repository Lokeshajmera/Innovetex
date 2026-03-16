import base64, json, logging, os, sys, threading, time
from io import BytesIO
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except: YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except: MP_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except: OCR_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except: TRANSLATOR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("neovision")

app = Flask(__name__)
app.config["SECRET_KEY"] = "neovision2024"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

camera = None
camera_active = False
yolo_model = None
hands_detector = None
frame_count = 0
fps_counter = {"last": time.time(), "count": 0, "fps": 0}

SIGN_DICT = {
    "fist":      {"text": "Stop / No",              "emoji": "✊", "conf": 94},
    "open_hand": {"text": "Hello / Hi there",        "emoji": "✋", "conf": 96},
    "thumbs_up": {"text": "Good / Okay",             "emoji": "👍", "conf": 93},
    "peace":     {"text": "Peace / Two / Victory",   "emoji": "✌", "conf": 89},
    "pointing":  {"text": "Look / Over there",       "emoji": "👆", "conf": 88},
    "love_you":  {"text": "I love you",              "emoji": "🤟", "conf": 95},
    "pinky_up":  {"text": "I need water / Help",     "emoji": "🤙", "conf": 87},
    "ok_sign":   {"text": "Perfect / Okay",          "emoji": "👌", "conf": 90},
    "call_me":   {"text": "Call me / Phone",         "emoji": "🤙", "conf": 86},
    "prayer":    {"text": "Thank you / Please",      "emoji": "🙏", "conf": 97},
    "thumbs_down":{"text": "No / Bad",               "emoji": "👎", "conf": 91},
    "wave":      {"text": "Goodbye / See you",       "emoji": "👋", "conf": 93},
}

OFFLINE_DICT = {
    "hi": {
        "stop / no": "रुको / नहीं", "hello / hi there": "नमस्ते", "good / okay": "अच्छा / ठीक है",
        "peace / two / victory": "शांति / दो / जीत", "look / over there": "देखो / वहाँ",
        "i love you": "मैं तुमसे प्यार करता हूँ", "i need water / help": "मुझे पानी / मदद चाहिए",
        "perfect / okay": "बिल्कुल सही", "call me / phone": "मुझे कॉल करो",
        "thank you / please": "धन्यवाद / कृपया", "no / bad": "नहीं / बुरा",
        "goodbye / see you": "अलविदा / फिर मिलेंगे",
        "hello": "नमस्ते", "thank you": "धन्यवाद", "good morning": "शुभ प्रभात",
        "good night": "शुभ रात्रि", "how are you": "आप कैसे हैं", "help": "मदद",
        "water": "पानी", "food": "खाना", "hospital": "अस्पताल", "police": "पुलिस",
        "danger": "खतरा", "stop": "रुको", "yes": "हाँ", "no": "नहीं",
    },
    "mr": {
        "stop / no": "थांबा / नाही", "hello / hi there": "नमस्कार", "good / okay": "चांगले / ठीक आहे",
        "peace / two / victory": "शांती / दोन / विजय", "look / over there": "पहा / तिकडे",
        "i love you": "मी तुझ्यावर प्रेम करतो", "i need water / help": "मला पाणी / मदत हवी",
        "perfect / okay": "अगदी बरोबर", "call me / phone": "मला फोन करा",
        "thank you / please": "धन्यवाद / कृपया", "no / bad": "नाही / वाईट",
        "goodbye / see you": "निरोप / पुन्हा भेटू",
        "hello": "नमस्कार", "thank you": "धन्यवाद", "good morning": "शुभ सकाळ",
        "good night": "शुभ रात्री", "how are you": "तुम्ही कसे आहात", "help": "मदत",
        "water": "पाणी", "food": "अन्न", "hospital": "रुग्णालय", "police": "पोलीस",
        "danger": "धोका", "stop": "थांबा", "yes": "होय", "no": "नाही",
    }
}

def translate_text(text, target_lang):
    text_lower = text.lower().strip()
    result = {"original": text, "translated": None, "lang": target_lang, "method": ""}
    if TRANSLATOR_AVAILABLE:
        try:
            lang_code = "hi" if target_lang == "hi" else "mr"
            translated = GoogleTranslator(source="auto", target=lang_code).translate(text)
            result["translated"] = translated
            result["method"] = "google"
            return result
        except Exception as e:
            log.warning(f"Google Translate failed: {e}")
    d = OFFLINE_DICT.get(target_lang, {})
    for key, val in d.items():
        if key in text_lower or text_lower in key:
            result["translated"] = val
            result["method"] = "offline"
            return result
    result["translated"] = f"[{text}]"
    result["method"] = "fallback"
    return result

def preprocess_for_ocr(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = binary.shape
    scale = max(1.0, 800 / max(h, w, 1))
    scaled = cv2.resize(binary, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return scaled

def extract_text_from_image(image_b64):
    try:
        img_data = base64.b64decode(image_b64.split(",")[-1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "error": "Could not decode image"}
        processed = preprocess_for_ocr(img)
        if OCR_AVAILABLE:
            config = "--oem 3 --psm 6 -l eng"
            raw_text = pytesseract.image_to_string(processed, config=config).strip()
            data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data["conf"] if c != "-1" and int(c) > 0]
            avg_conf = sum(confs) / len(confs) if confs else 0
            return {"success": True, "text": raw_text, "confidence": round(avg_conf, 1), "method": "tesseract", "char_count": len(raw_text)}
        else:
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            char_regions = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                aspect = w / h if h > 0 else 0
                if 100 < area < 50000 and 0.1 < aspect < 10:
                    char_regions.append({"x": x, "y": y, "w": w, "h": h})
            return {"success": True, "text": f"[{len(char_regions)} characters — install pytesseract for full OCR]", "confidence": 55.0, "method": "contour", "char_count": len(char_regions)}
    except Exception as e:
        log.error(f"OCR error: {e}")
        return {"success": False, "error": str(e)}

def classify_hand_gesture(landmarks):
    lm = landmarks.landmark
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    fingers = [1 if lm[4].x < lm[3].x else 0]
    for i in range(1, 5):
        fingers.append(1 if lm[tips[i]].y < lm[pips[i]].y else 0)
    t, i, m, r, p = fingers
    def dist(a, b):
        return ((lm[a].x-lm[b].x)**2+(lm[a].y-lm[b].y)**2)**0.5
    thumb_index_close = dist(4,8) < 0.05
    if fingers==[0,0,0,0,0]: return "fist"
    if fingers==[1,1,1,1,1]: return "open_hand"
    if fingers==[1,0,0,0,0]: return "thumbs_up"
    if fingers==[0,0,0,0,0] and lm[4].y > lm[3].y: return "thumbs_down"
    if fingers==[0,1,1,0,0]: return "peace"
    if fingers==[0,1,0,0,0]: return "pointing"
    if fingers==[1,1,0,0,1]: return "love_you"
    if fingers==[0,0,0,0,1]: return "pinky_up"
    if fingers==[1,0,0,0,1]: return "call_me"
    if thumb_index_close: return "ok_sign"
    if all(f==1 for f in fingers[1:]): return "prayer" if t==0 else "wave"
    return "open_hand"

def process_frame(frame):
    global frame_count
    frame_count += 1
    detections = []
    gesture_result = None
    annotated = frame.copy()

    if YOLO_AVAILABLE and yolo_model and frame_count % 3 == 0:
        try:
            results = yolo_model(frame, conf=0.45, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = yolo_model.names[cls]
                    detections.append({"label": label, "conf": round(conf,2), "box": [x1,y1,x2,y2]})
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,245,255),2)
                    cv2.putText(annotated,f"{label} {conf:.0%}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,245,255),1)
        except Exception as e:
            log.warning(f"YOLO: {e}")

    if MP_AVAILABLE and hands_detector:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_detector.process(rgb)
            if res.multi_hand_landmarks:
                for hand_lm in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,136),thickness=2,circle_radius=3),
                        mp_drawing.DrawingSpec(color=(123,47,255),thickness=2))
                    gk = classify_hand_gesture(hand_lm)
                    sd = SIGN_DICT.get(gk, {"text":"Unknown","emoji":"?","conf":50})
                    gesture_result = {"key": gk, "text": sd["text"], "emoji": sd["emoji"], "conf": sd["conf"]}
                    cv2.putText(annotated, f"{sd['text']}", (10,36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,136), 2)
        except Exception as e:
            log.warning(f"MP: {e}")

    now = time.time()
    fps_counter["count"] += 1
    if now - fps_counter["last"] >= 1.0:
        fps_counter["fps"] = fps_counter["count"]
        fps_counter["count"] = 0
        fps_counter["last"] = now
    cv2.putText(annotated, f"FPS:{fps_counter['fps']}", (frame.shape[1]-75,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,245,255), 1)
    return annotated, detections, gesture_result

def camera_thread():
    global camera, camera_active, hands_detector
    hands = None
    if MP_AVAILABLE:
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.65, min_tracking_confidence=0.55)
    hands_detector = hands
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        log.error("Camera not found")
        socketio.emit("camera_error", {"msg": "Camera not found"})
        return
    camera = cap
    log.info("Camera started 1280x720")
    socketio.emit("camera_ready", {"width": 1280, "height": 720})
    last_gesture_emit = 0
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        annotated, detections, gesture = process_frame(frame)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 82]
        _, buf = cv2.imencode(".jpg", annotated, encode_params)
        frame_b64 = base64.b64encode(buf).decode("utf-8")
        socketio.emit("frame", {"img": frame_b64, "fps": fps_counter["fps"], "detections": detections})
        if gesture and time.time() - last_gesture_emit > 0.8:
            socketio.emit("gesture", gesture)
            last_gesture_emit = time.time()
        time.sleep(1/30)
    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({"yolo": YOLO_AVAILABLE, "mediapipe": MP_AVAILABLE, "ocr": OCR_AVAILABLE, "translator": TRANSLATOR_AVAILABLE, "camera": camera_active, "fps": fps_counter["fps"]})

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json()
    text = data.get("text","").strip()
    lang = data.get("lang","hi")
    if not text: return jsonify({"error":"No text"}),400
    return jsonify(translate_text(text, lang))

@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    data = request.get_json()
    image_b64 = data.get("image","")
    target_lang = data.get("lang","hi")
    if not image_b64: return jsonify({"error":"No image"}),400
    ocr_result = extract_text_from_image(image_b64)
    if ocr_result["success"] and ocr_result.get("text"):
        ocr_result["translation"] = translate_text(ocr_result["text"], target_lang)
    return jsonify(ocr_result)

@socketio.on("start_camera")
def on_start_camera():
    global camera_active, yolo_model
    if camera_active:
        emit("camera_status",{"active":True,"msg":"Already running"}); return
    if YOLO_AVAILABLE and yolo_model is None:
        try:
            log.info("Loading YOLOv8n...")
            yolo_model = YOLO("yolov8n.pt")
        except Exception as e:
            log.warning(f"YOLO load: {e}")
    camera_active = True
    threading.Thread(target=camera_thread, daemon=True).start()
    emit("camera_status",{"active":True,"msg":"Camera started"})

@socketio.on("stop_camera")
def on_stop_camera():
    global camera_active
    camera_active = False
    emit("camera_status",{"active":False,"msg":"Stopped"})

@socketio.on("translate_sign")
def on_translate_sign(data):
    result = translate_text(data.get("text",""), data.get("lang","hi"))
    emit("translation_result", result)

@socketio.on("ocr_frame")
def on_ocr_frame(data):
    result = extract_text_from_image(data.get("image",""))
    if result["success"] and result.get("text"):
        result["translation"] = translate_text(result["text"], data.get("lang","hi"))
    emit("ocr_result", result)

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║   NeoVision Smart Glasses — Backend Server           ║
║   YOLO v8 + OpenCV + MediaPipe + OCR + Translator    ║
╚══════════════════════════════════════════════════════╝
   → http://localhost:5000
""")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
