import os
import cv2
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from werkzeug.utils import secure_filename

from services.face_input import process_face_file, process_face_input

from services.liveness import (
    process_liveness,
    process_blink_frame,
    create_blink_detector
)

from services.face_matching import (
    match_face_files,
    compare_identity_faces
)
from services.risk_scoring import (
    calculate_risk_from_identity_result
)

app = Flask(__name__)


# ============================================================
# FOLDERS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads",
    "module3"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "faces"
)


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# NORMAL PASSPORT / AADHAAR PROCESSING
# ============================================================

def process_uploaded_file(
    file,
    input_type
):

    if file is None:

        return {
            "success": False,
            "message": "No image uploaded."
        }


    if file.filename == "":

        return {
            "success": False,
            "message": "No image selected."
        }


    filename = secure_filename(
        file.filename
    )


    filename = (
        input_type
        + "_"
        + filename
    )


    original_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(
        original_path
    )


    result = process_face_file(
        original_path
    )


    if not result.get("success"):

        return result


    face_image = result.get(
        "face_image"
    )


    if face_image is None:

        return {
            "success": False,
            "message":
                "Face extraction failed."
        }


    face_filename = (
        input_type
        + "_face.jpg"
    )


    face_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        face_filename
    )


    cv2.imwrite(
        face_path,
        face_image
    )


    result["face_filename"] = (
        face_filename
    )


    result["face_path"] = (
        face_path
    )


    return result


# ============================================================
# PASSPORT UPLOAD
# ============================================================

@app.route(
    "/upload/passport",
    methods=["POST"]
)
def upload_passport():

    file = request.files.get(
        "passport"
    )


    result = process_uploaded_file(
        file,
        "passport"
    )


    if not result.get("success"):

        return jsonify(
            result
        ), 400


    return jsonify({

        "success": True,

        "type": "passport",

        "image_type":
            result.get(
                "image_type"
            ),

        "face_found":
            result.get(
                "face_found"
            ),

        "face_count":
            result.get(
                "face_count",
                1
            ),

        "face_quality_score":
            result.get(
                "face_quality_score"
            ),

        "quality_status":
            result.get(
                "quality_status"
            ),

        "blur_score":
            result.get(
                "blur_score"
            ),

        "face_box":
            result.get(
                "face_box"
            ),

        "face_filename":
            result.get(
                "face_filename"
            ),

        "message":
            result.get(
                "message"
            )
    })


# ============================================================
# AADHAAR UPLOAD
# ============================================================

@app.route(
    "/upload/aadhaar",
    methods=["POST"]
)
def upload_aadhaar():

    file = request.files.get(
        "aadhaar"
    )


    result = process_uploaded_file(
        file,
        "aadhaar"
    )


    if not result.get("success"):

        return jsonify(
            result
        ), 400


    return jsonify({

        "success": True,

        "type": "aadhaar",

        "image_type":
            result.get(
                "image_type"
            ),

        "face_found":
            result.get(
                "face_found"
            ),

        "face_count":
            result.get(
                "face_count",
                1
            ),

        "face_quality_score":
            result.get(
                "face_quality_score"
            ),

        "quality_status":
            result.get(
                "quality_status"
            ),

        "blur_score":
            result.get(
                "blur_score"
            ),

        "face_box":
            result.get(
                "face_box"
            ),

        "face_filename":
            result.get(
                "face_filename"
            ),

        "message":
            result.get(
                "message"
            )
    })


# ============================================================
# OLD SINGLE-FRAME LIVE ENDPOINT
# ============================================================

@app.route(
    "/upload/live",
    methods=["POST"]
)
def upload_live():

    file = request.files.get(
        "live_face"
    )


    result = process_uploaded_file(
        file,
        "live"
    )


    if not result.get("success"):

        return jsonify(
            result
        ), 400


    face_image = result.get(
        "face_image"
    )


    liveness_result = process_liveness(
        face_image
    )


    return jsonify({

        "success": True,

        "type": "live",

        "image_type": "live",

        "face_found":
            result.get(
                "face_found"
            ),

        "face_count":
            result.get(
                "face_count",
                1
            ),

        "face_quality_score":
            result.get(
                "face_quality_score"
            ),

        "quality_status":
            result.get(
                "quality_status"
            ),

        "blur_score":
            result.get(
                "blur_score"
            ),

        "face_box":
            result.get(
                "face_box"
            ),

        "face_filename":
            result.get(
                "face_filename"
            ),

        "liveness_score":
            liveness_result.get(
                "liveness_score"
            ),

        "liveness_decision":
            liveness_result.get(
                "liveness_decision"
            ),

        "challenge_required":
            liveness_result.get(
                "challenge_required"
            ),

        "challenge":
            liveness_result.get(
                "challenge"
            ),

        "liveness_details":
            liveness_result.get(
                "details"
            ),

        "message":
            result.get(
                "message"
            )
    })


# ============================================================
# BLINK DETECTOR
# ============================================================

blink_detector = None


# ============================================================
# START BLINK VERIFICATION
# ============================================================

@app.route(
    "/blink/start",
    methods=["POST"]
)
def blink_start():

    global blink_detector


    blink_detector = (
        create_blink_detector()
    )


    return jsonify({

        "success": True,

        "challenge":
            "BLINK",

        "instruction":
            "Please blink naturally once.",

        "message":
            "Blink verification started."
    })


# ============================================================
# PROCESS BLINK FRAME
# ============================================================

@app.route(
    "/blink/frame",
    methods=["POST"]
)
def blink_frame():

    global blink_detector


    if blink_detector is None:

        return jsonify({

            "success": False,

            "message":
                "Blink verification has not been started."

        }), 400


    file = request.files.get(
        "frame"
    )


    if file is None:

        return jsonify({

            "success": False,

            "message":
                "No camera frame received."

        }), 400


    image_bytes = file.read()


    if not image_bytes:

        return jsonify({

            "success": False,

            "message":
                "Empty camera frame."

        }), 400


    image_array = cv2.imdecode(

        np.frombuffer(
            image_bytes,
            dtype=np.uint8
        ),

        cv2.IMREAD_COLOR
    )


    if image_array is None:

        return jsonify({

            "success": False,

            "message":
                "Unable to decode camera frame."

        }), 400


    result = process_blink_frame(
        image_array,
        blink_detector
    )

    # Save the live face only after the blink challenge succeeds.
    # This becomes the live face used for SFace identity matching.
    if result.get("success") and result.get("blink_detected"):

        live_result = process_face_input(
            image_array
        )

        if live_result.get("success") and live_result.get("face_image") is not None:

            live_face_path = os.path.join(
                app.config["OUTPUT_FOLDER"],
                "live_face.jpg"
            )

            cv2.imwrite(
                live_face_path,
                live_result["face_image"]
            )

            result["live_face_captured"] = True
            result["live_face_filename"] = "live_face.jpg"
            result["live_face_quality_score"] = live_result.get(
                "face_quality_score"
            )
            result["live_face_quality_status"] = live_result.get(
                "quality_status"
            )
            result["message"] = (
                "Blink detected successfully. "
                "Live face captured and liveness verification passed."
            )

        else:

            result["live_face_captured"] = False
            result["message"] = (
                "Blink detected, but live face capture failed. "
                "Please keep your face clearly visible."
            )

    else:

        result["live_face_captured"] = False


    return jsonify(
        result
    )


# ============================================================
# RESET BLINK DETECTOR
# ============================================================

@app.route(
    "/blink/reset",
    methods=["POST"]
)
def blink_reset():

    global blink_detector


    blink_detector = None


    return jsonify({

        "success": True,

        "message":
            "Blink verification reset."
    })


# ============================================================
# PASSPORT ↔ AADHAAR MATCHING
# ============================================================

@app.route(
    "/match/passport-aadhaar",
    methods=["POST"]
)
def match_passport_aadhaar():

    passport_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        "passport_face.jpg"
    )


    aadhaar_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        "aadhaar_face.jpg"
    )


    if not os.path.exists(
        passport_path
    ):

        return jsonify({

            "success": False,

            "decision": "ERROR",

            "message":
                "Passport face has not been uploaded yet."

        }), 400


    if not os.path.exists(
        aadhaar_path
    ):

        return jsonify({

            "success": False,

            "decision": "ERROR",

            "message":
                "Aadhaar face has not been uploaded yet."

        }), 400


    result = match_face_files(
        passport_path,
        aadhaar_path
    )


    return jsonify({

        "success":
            result.get(
                "success",
                False
            ),

        "passport_aadhaar":
            True,

        "similarity":
            result.get(
                "similarity",
                0.0
            ),

        "threshold":
            result.get(
                "threshold"
            ),

        "decision":
            result.get(
                "decision",
                "ERROR"
            ),

        "message":
            result.get(
                "message"
            )
    })


# ============================================================
# FULL IDENTITY MATCHING
# ============================================================

@app.route(
    "/match/identity",
    methods=["POST"]
)
def match_identity():

    passport_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        "passport_face.jpg"
    )

    aadhaar_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        "aadhaar_face.jpg"
    )

    live_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        "live_face.jpg"
    )

    if not os.path.exists(passport_path):
        return jsonify({
            "success": False,
            "decision": "ERROR",
            "message": "Passport face has not been uploaded yet."
        }), 400

    if not os.path.exists(aadhaar_path):
        return jsonify({
            "success": False,
            "decision": "ERROR",
            "message": "Aadhaar face has not been uploaded yet."
        }), 400

    if not os.path.exists(live_path):
        return jsonify({
            "success": False,
            "decision": "ERROR",
            "message": "Live face has not been captured yet."
        }), 400

    passport_face = cv2.imread(passport_path)
    aadhaar_face = cv2.imread(aadhaar_path)
    live_face = cv2.imread(live_path)

    result = compare_identity_faces(
        passport_face,
        aadhaar_face,
        live_face
    )
        # --------------------------------------------------------
    # RISK SCORING
    # --------------------------------------------------------

    risk_result = calculate_risk_from_identity_result(
        identity_result=result,
        passport_quality=process_face_input(
            passport_face
        ).get("face_quality_score", 0.0),
        aadhaar_quality=process_face_input(
            aadhaar_face
        ).get("face_quality_score", 0.0),
        live_quality=process_face_input(
            live_face
        ).get("face_quality_score", 0.0),
        liveness_decision="REAL"
    )

    return jsonify({
        "success": result.get("success", False),

        "passport_aadhaar": result.get(
            "passport_aadhaar"
        ),

        "passport_live": result.get(
            "passport_live"
        ),

        "aadhaar_live": result.get(
            "aadhaar_live"
        ),

        "overall_decision": result.get(
            "overall_decision",
            "ERROR"
        ),

        # Risk scoring result
        "risk_score": risk_result.get(
            "risk_score"
        ),

        "risk_level": risk_result.get(
            "risk_level"
        ),

        "liveness_status": risk_result.get(
            "liveness_status"
        ),

        "face_match_confidence": risk_result.get(
            "face_match_confidence"
        ),

        "document_face_quality": risk_result.get(
            "document_face_quality"
        ),

        "minimum_face_similarity": risk_result.get(
            "minimum_face_similarity"
        ),

        "risk_reasons": risk_result.get(
            "reasons",
            []
        ),

        "risk_signals": risk_result.get(
            "signals",
            {}
        )
    })


# ============================================================
# EXTRACTED FACE FILES
# ============================================================

@app.route(
    "/faces/<filename>"
)
def get_face(filename):

    return send_from_directory(
        app.config["OUTPUT_FOLDER"],
        filename
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "running",

        "module":
            "TRUST-ID Module 3"
    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5001,

        debug=True
    )