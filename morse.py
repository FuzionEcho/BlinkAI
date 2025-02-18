from time import perf_counter
#import numpy as np
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
#import time
from flask import Flask, render_template, Response, jsonify
import requests

# ------------------------------
# Mistral AI Configuration
# ------------------------------
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "Your API KEY"

# ------------------------------
# Global Morse Decoder State
# ------------------------------

# Binary tree representation of Morse code.
# The index (trace-1) corresponds to a letter based on the sequence of dots (0) and dashes (1).
letters = ['','E','T','I','A','N','M','S','U','R','W','D','K','G','O','H','V','F','','L','','P','J','B','X','C','Y','Z','Q','','']

def choose_letter(dots):
    if len(dots) > 4:
        return 'TOO BIG'
    trace = 1
    for dot in dots:
        if dot == 0:
            trace = 2 * trace
        else:
            trace = (2 * trace) + 1
    return letters[int(trace - 1)]

# Global variables to keep track of blink detection and Morse decoding.
start_flag = False  # When True, start processing blinks
is_paused = False  # When True, pause processing while maintaining state
is_processing = False  # For AI processing state
letter = []         # List to hold current letter's Morse signals (0 for dot, 1 for dash)
word = ''           # Decoded word (all letters concatenated)
morse = ''          # String representation of the current Morse sequence
conversation_history = []  # To store chat history
startTime = 0
notBlinking = 0
blinking = 0
counter = 0

# For smoothing the ratio calculation.
dList = []     # Vertical distances (for averaging)
horList = []   # Horizontal distances (for averaging)
r_list = []    # Ratio values
avg_d = []     # Running average of ratio

# ------------------------------
# Initialize Video and FaceMesh Detector
# ------------------------------
def initialize_camera():
    """Try different camera indices until finding a working one."""
    for index in range(10):  # Try first 10 camera indices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully opened camera at index {index}")
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
            cap.release()
    raise RuntimeError("No working camera found")

try:
    video = initialize_camera()
except RuntimeError as e:
    print(f"Error: {e}")
    exit(1)

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50])  # Optional live plot
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]  # For drawing landmarks

# ------------------------------
# Flask Application Setup
# ------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to start blink processing.
@app.route('/start', methods=['POST'])
def start_detection():
    global start_flag
    start_flag = True
    return jsonify({'status': 'started'})

# Endpoint to reset the current Morse message.
@app.route('/reset', methods=['POST'])
def reset_detection():
    global letter, word, morse, start_flag, is_paused, conversation_history
    letter = []
    word = ''
    morse = ''
    start_flag = False
    is_paused = False
    conversation_history = []
    return jsonify({'status': 'reset'})

# Endpoint to toggle pause state
@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({'status': 'paused' if is_paused else 'resumed'})

# Updated endpoint to get current message state including conversation
@app.route('/get_current_message')
def get_current_message():
    global word, morse, conversation_history
    return jsonify({
        'word': word, 
        'morse': morse,
        'conversation': conversation_history
    })

# New endpoint for AI processing
@app.route('/process_message', methods=['POST'])
def process_message():
    global word, is_processing, conversation_history
    
    if not word or is_processing:
        return jsonify({
            'status': 'error',
            'message': 'No message to process or already processing'
        })
    
    try:
        is_processing = True
        
        # Add user message to conversation history
        conversation_history.append({
            'role': 'user',
            'content': word
        })
        
        # Prepare the request for Mistral AI
        headers = {
            'Authorization': f'Bearer {MISTRAL_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'mistral-large-latest',
            'messages': conversation_history,
            'max_tokens': 1024,
            'temperature': 0.7
        }
        
        # Get AI response
        response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        ai_response = response.json()['choices'][0]['message']['content']
        
        # Add AI response to conversation history
        conversation_history.append({
            'role': 'assistant',
            'content': ai_response
        })
        
        # Reset the current word after processing
        word = ''
        
        return jsonify({
            'status': 'success',
            'response': ai_response
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            'status': 'error',
            'message': f'API Error: {str(e)}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
    finally:
        is_processing = False

# ------------------------------
# Video Streaming Generator
# ------------------------------
def gen_frames():
    global start_flag, letter, word, morse, startTime, notBlinking, blinking, counter
    global dList, horList, r_list, avg_d, is_paused
    while True:
        success, img = video.read()
        if not success:
            break

        img, faces = detector.findFaceMesh(img, draw=False)
        if faces:
            face = faces[0]
            # Get landmarks for left and right eyes.
            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            rightUp = face[386]
            rightDown = face[374]
            rightLeft = face[398]
            rightRight = face[359]

            # Calculate distances.
            distanceVert, _ = detector.findDistance(leftUp, leftDown)
            distanceHor, _ = detector.findDistance(leftLeft, leftRight)
            distanceVertR, _ = detector.findDistance(rightUp, rightDown)
            distanceHorR, _ = detector.findDistance(rightLeft, rightRight)

            # Only process blinks if not paused
            if not is_paused:
                # Average horizontal distance.
                horList.append((distanceHor + distanceHorR) / 2)
                if len(horList) > 1:
                    horList.pop(0)
                distanceHor = sum(horList) / len(horList)

                # Average vertical distance.
                dList.append((distanceVert + distanceVertR) / 2)
                if len(dList) > 2:
                    dList.pop(0)
                distanceVert = sum(dList) / len(dList)

                # Compute ratio (vertical/horizontal).
                ratio = (distanceVert / distanceHor) * 100
                r_list.append(ratio)
                if len(r_list) > 2:
                    r_list.pop(0)
                ratio = sum(r_list) / len(r_list)

                # Running average for ratio.
                avg_d.append(ratio)
                if len(avg_d) > 200:
                    avg_d.pop(0)
                avg_ratio = sum(avg_d) / len(avg_d)

                if start_flag:
                    # Blink detection: if the ratio drops significantly below the average.
                    if avg_ratio - ratio > 3:
                        if blinking == 0:  # Blink start.
                            startTime = perf_counter()
                            blinking = 1
                            counter += 1
                    else:
                        if blinking == 1:  # Blink end.
                            howLong = perf_counter() - startTime
                            notBlinking = perf_counter()
                            if howLong < 0.28:  # Short blink -> dot.
                                letter.append(0)
                                morse += '.'
                            elif howLong < 2:  # Longer blink -> dash.
                                letter.append(1)
                                morse += '_'
                        # If no blink for at least 1 second, decode the letter.
                        if perf_counter() - notBlinking >= 1:
                            if len(letter) < 5:  # Only decode if there are up to 4 signals.
                                decoded = choose_letter(letter)
                                word += decoded
                            letter = []
                            morse = ''
                        blinking = 0

            # Add pause indicator if paused
            if is_paused:
                cvzone.putTextRect(img, "", (0, 0), 
                                 colorR=(0, 0, 0), scale=0, thickness=0)

            # Always show current word and morse code
            
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
