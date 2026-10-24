from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import re
import base64
import io
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

CAMPUS_DB_MAP = {
    "Graphic Era Deemed to be University (Dehradun)": "db_geu_deemed.db",
    "Graphic Era Hill University (Dehradun Campus)": "db_gehu_dehradun.db",
    "Graphic Era Hill University (Bhimtal Campus)": "db_gehu_bhimtal.db",
    "Graphic Era Hill University (Haldwani Campus)": "db_gehu_haldwani.db",
    "Graphic Era Medical & Hospital Campus": "db_geims_hospital.db"
}

def get_db_name(campus_name):
    return CAMPUS_DB_MAP.get(campus_name, "db_geu_deemed.db")

def init_all_campus_dbs():
    for campus, db_file in CAMPUS_DB_MAP.items():
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                role TEXT DEFAULT 'Student',
                portal_type TEXT DEFAULT 'student',
                phone TEXT NOT NULL,
                campus TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                title TEXT,
                category TEXT,
                location TEXT,
                date TEXT,
                img TEXT,
                desc TEXT,
                secret TEXT,
                reporter TEXT,
                reporter_role TEXT DEFAULT 'Student',
                target_portal TEXT DEFAULT 'student',
                handover_location TEXT DEFAULT 'With Finder',
                contact TEXT,
                claimed INTEGER DEFAULT 0
            )
        ''')
        # Safe updates for older databases
        for col, default in [("role", "'Student'"), ("portal_type", "'student'")]:
            try: c.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT DEFAULT {default}")
            except Exception: pass
        for col, default in [("reporter_role", "'Student'"), ("target_portal", "'student'"), ("handover_location", "'With Finder'")]:
            try: c.execute(f"ALTER TABLE items ADD COLUMN {col} TEXT DEFAULT {default}")
            except Exception: pass
        conn.commit()
        conn.close()

init_all_campus_dbs()

@app.route('/')
def serve_home():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(os.getcwd(), filename)

def predict_target_portal(title, desc, location, category, user_selected_target):
    if user_selected_target in ['student', 'outsider', 'both'] and user_selected_target != 'auto':
        return user_selected_target
    combined = f"{title} {desc} {location} {category}".lower()
    outsider_keywords = ['opd', 'patient', 'hospital', 'medical', 'prescription', 'report', 'xray', 'parent', 'visitor', 'reception', 'parking', 'gate', 'aadhar', 'license', 'spectacles', 'ward']
    student_keywords = ['roll', 'semester', 'btech', 'bca', 'hostel', 'lab', 'classroom', 'library', 'assignment', 'notebook', 'uniform']
    is_outsider = any(k in combined for k in outsider_keywords)
    is_student = any(k in combined for k in student_keywords)
    if is_outsider and not is_student: return 'outsider'
    elif is_student and not is_outsider: return 'student'
    return 'both'

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    name = data.get('name', '').strip()
    role = data.get('role', 'Student').strip()
    portal_type = data.get('portal_type', 'student').strip()
    phone = data.get('phone', '').strip()
    campus = data.get('campus', '').strip()

    if not name or not campus:
        return jsonify({"status": "error", "message": "Name and Campus are required!"}), 400
    if not re.fullmatch(r'\d{10}', phone):
        return jsonify({"status": "error", "message": "Phone number must be exactly 10 digits!"}), 400

    db_file = get_db_name(campus)
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("INSERT INTO users (name, role, portal_type, phone, campus) VALUES (?, ?, ?, ?, ?)", (name, role, portal_type, phone, campus))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "user": {"name": name, "role": role, "portal_type": portal_type, "phone": phone, "campus": campus, "db": db_file}})

def detect_image_color_and_tags(base64_str, filename=""):
    try:
        if "," in base64_str: base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        cropped = img.crop((w * 0.2, h * 0.2, w * 0.8, h * 0.8)).resize((1, 1))
        r, g, b = cropped.getpixel((0, 0))

        if r < 60 and g < 60 and b < 60: color = "Black"
        elif r > 200 and g > 200 and b > 200: color = "White / Silver"
        elif r > 150 and g < 80 and b < 80: color = "Red"
        elif b > 130 and r < 100: color = "Blue"
        elif g > 130 and r < 100: color = "Green"
        elif r > 120 and g > 70 and b < 60: color = "Brown / Tan"
        else: color = "Multi-colored"

        fname = filename.lower()
        if any(k in fname for k in ['medical', 'report', 'prescription', 'opd']):
            category = "Medical Files / Reports"
            item_type = f"{color} Medical File"
        elif any(k in fname for k in ['id', 'card', 'aadhar']):
            category = "ID Cards & Documents"
            item_type = f"{color} ID Card"
        elif any(k in fname for k in ['wallet', 'bag', 'purse']):
            category = "Wallets & Bags"
            item_type = f"{color} Wallet / Bag"
        else:
            category = "Electronics"
            item_type = f"{color} Electronic Device"

        desc = f"AI Scan: Dominant {color.lower()} color profile (RGB: {r},{g},{b})."
        return {"title": item_type, "category": category, "desc": desc, "color": color}
    except Exception:
        return {"title": "Scanned Item", "category": "Electronics", "desc": "AI Vision scanned image.", "color": "Detected"}

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    return jsonify(detect_image_color_and_tags(request.json.get('img', ''), request.json.get('filename', '')))

def compute_nlp_similarity(text1, text2):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception: return 0.0

def compute_image_similarity(b64_img1, b64_img2):
    try:
        if not b64_img1.startswith('data:image') or not b64_img2.startswith('data:image'): return 0.80
        bytes1 = base64.b64decode(b64_img1.split(",")[1])
        bytes2 = base64.b64decode(b64_img2.split(",")[1])
        img1 = Image.open(io.BytesIO(bytes1)).convert('RGB').resize((64, 64))
        img2 = Image.open(io.BytesIO(bytes2)).convert('RGB').resize((64, 64))
        return float(cosine_similarity([img1.histogram()], [img2.histogram()])[0][0])
    except Exception: return 0.75

@app.route('/api/items', methods=['GET'])
def get_items():
    campus = request.args.get('campus', 'Graphic Era Deemed to be University (Dehradun)')
    portal_type = request.args.get('portal_type', 'student')
    db_file = get_db_name(campus)

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM items WHERE target_portal = ? OR target_portal = 'both' ORDER BY id DESC", (portal_type,))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()

    lost_items = [i for i in rows if i['type'] == 'lost' and not int(i.get('claimed') or 0)]
    found_items = [i for i in rows if i['type'] == 'found' and not int(i.get('claimed') or 0)]
    ai_matches, matched_ids = [], set()

    for lost in lost_items:
        for found in found_items:
            if lost['category'] == found['category']:
                lost_text = f"{lost['title']} {lost['desc']}"
                found_text = f"{found['title']} {found['desc']}"
                nlp_score = compute_nlp_similarity(lost_text, found_text)
                visual_score = compute_image_similarity(lost.get('img', ''), found.get('img', ''))

                total_score = 30
                same_location = bool(set(lost['location'].lower().split()).intersection(set(found['location'].lower().split())))
                if same_location: total_score += 20
                total_score = min(98, max(55, total_score + int(nlp_score * 25) + int(visual_score * 25)))

                common_words = list(set(w.lower() for w in lost_text.split() if len(w) > 2).intersection(set(w.lower() for w in found_text.split() if len(w) > 2)))
                matched_ids.add(lost['id'])
                matched_ids.add(found['id'])

                ai_matches.append({
                    "lost": lost, "found": found, "score": total_score,
                    "visualScore": int(visual_score * 100), "sameLocation": same_location,
                    "matchedKeywords": ", ".join(common_words) if common_words else lost['category']
                })

    for item in rows:
        if int(item.get('claimed') or 0) == 1:
            item['badge'] = 'Claimed ✅'
            item['badgeColor'] = 'bg-blue-100 text-blue-800'
        elif item['id'] in matched_ids:
            item['badge'] = 'AI Match Available'
            item['badgeColor'] = 'bg-emerald-100 text-emerald-800'
        else:
            item['badge'] = 'Awaiting Claim' if item['type'] == 'found' else 'Awaiting Match'
            item['badgeColor'] = 'bg-amber-100 text-amber-800'

    ai_matches.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({"items": rows, "matches": ai_matches, "campus": campus, "portal_type": portal_type})

@app.route('/api/items', methods=['POST'])
def add_item():
    data = request.json
    campus, contact = data.get('campus', ''), data.get('contact', '').strip()
    if not re.fullmatch(r'\d{10}', contact):
        return jsonify({"status": "error", "message": "Phone number must be exactly 10 digits!"}), 400

    target_portal = predict_target_portal(data.get('title', ''), data.get('desc', ''), data.get('location', ''), data.get('category', ''), data.get('target_portal', 'auto'))

    conn = sqlite3.connect(get_db_name(campus))
    c = conn.cursor()
    c.execute('''
        INSERT INTO items (type, title, category, location, date, img, desc, secret, reporter, reporter_role, target_portal, handover_location, contact, claimed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    ''', (
        data.get('type'), data.get('title'), data.get('category'), data.get('location'), 'Just Now',
        data.get('img'), data.get('desc'), data.get('secret', 'Any unique mark?'),
        data.get('reporter', 'Campus Member'), data.get('reporter_role', 'Student'),
        target_portal, data.get('handover_location', 'With Finder'), contact
    ))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "routed_to": target_portal})

@app.route('/api/claim/<int:item_id>', methods=['POST'])
def claim_item(item_id):
    conn = sqlite3.connect(get_db_name(request.args.get('campus', '')))
    conn.execute("UPDATE items SET claimed = 1 WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "claimed"})

@app.route('/api/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    conn = sqlite3.connect(get_db_name(request.args.get('campus', '')))
    conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted"})

if __name__ == '__main__':
    print("🚀 Dual-Portal (Student + Outsider) Multi-Campus AI Server running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)