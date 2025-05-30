import os
import uuid
import json
import logging
from functools import wraps
from flask import Flask, jsonify, request, abort, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Mock Pillow & Flask-Sockets (Simplified)
try: from PIL import Image
except ImportError:
    class Image: # Mock
        format = 'PNG'
        def thumbnail(self, size): pass
        def save(self, path, format): pass
        @staticmethod
        def open(*args, **kwargs): return Image()

try: from flask_sockets import Sockets
except ImportError:
    class Sockets: # Mock
        def __init__(self, app=None): self.app = app
        def route(self, rule):
            def decorator(f):
                @wraps(f)
                def decorated_function(*args, **kwargs): return f(*args, **kwargs)
                return decorated_function
            return decorator

app = Flask(__name__)
sockets = Sockets(app)

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['COLLAB_DOC_STORAGE_PATH'] = os.path.join(BASE_DIR, 'collab_documents')
app.config['CKBOX_FILE_STORAGE_PATH'] = os.path.join(BASE_DIR, 'ckbox_files')
app.config['CKBOX_ROOT_THUMBNAIL_DIR_NAME'] = '.thumbnails' # Name of thumbnail dir within each category
app.config['TELEMETRY_LOG_FILE'] = os.path.join(BASE_DIR, 'telemetry.log')
app.config['CKBOX_ACCESS_TOKEN'] = "TEST_ACCESS_TOKEN"
app.config['CKBOX_ALLOWED_EXTENSIONS'] = {'txt','pdf','png','jpg','jpeg','gif','json'}
app.config['CKBOX_IMAGE_EXTENSIONS'] = {'png','jpg','jpeg','gif'}
app.config['CKBOX_MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['CKBOX_THUMBNAIL_SIZE'] = (150, 150)

for path_key in ['COLLAB_DOC_STORAGE_PATH', 'CKBOX_FILE_STORAGE_PATH']:
    path_val = app.config[path_key]
    if not os.path.exists(path_val): os.makedirs(path_val); app.logger.info(f"Created directory: {path_val}")

VALID_TOKENS = {app.config['CKBOX_ACCESS_TOKEN']: {"user_id": "test_user_id", "name": "Test User"}}
COLLAB_SESSIONS = {}

telemetry_logger = logging.getLogger('telemetry')
if not telemetry_logger.handlers: # Check if handlers are already added
    telemetry_logger.setLevel(logging.INFO)
    th = logging.FileHandler(app.config['TELEMETRY_LOG_FILE'])
    th.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    telemetry_logger.addHandler(th)
    telemetry_logger.propagate = False

# --- Endpoints ---
@app.route('/ckeditor-license', methods=['GET'])
def ckeditor_license(): return jsonify({"licenseKey": "YOUR_PREMIUM_LICENSE_KEY_FOR_TESTING"})

def require_auth_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '): token = auth_header.split(' ')[-1]
        else: token = request.args.get('token') # Also check query param for token (e.g. for WebSockets)

        if not token: abort(401, description="Missing Authorization Token/Header.")
        # In a real app, you'd validate the token structure and expiry here
        if token not in VALID_TOKENS: abort(403, description="Invalid or expired token.")
        # g.user_info = VALID_TOKENS[token] # Optional: store user info in Flask's g object
        return f(*args, **kwargs)
    return decorated_function

@app.route('/auth/token', methods=['POST']) # Changed from /ckbox/auth/generate-token
def generate_auth_token():
    user_id = "user_" + str(uuid.uuid4())
    user_name = request.form.get("name", "User_" + user_id[:4])
    # This is highly simplified: in reality, you'd generate a new, unique token here
    # and manage its lifecycle. For this demo, we reuse the static one and update its associated user.
    VALID_TOKENS[app.config['CKBOX_ACCESS_TOKEN']]['user_id'] = user_id
    VALID_TOKENS[app.config['CKBOX_ACCESS_TOKEN']]['name'] = user_name
    return jsonify({"token_type":"Bearer","access_token":app.config['CKBOX_ACCESS_TOKEN'],"expires_in":3600,"userId":user_id,"userName":user_name})

def allowed_file(fn): return '.' in fn and fn.rsplit('.',1)[1].lower() in app.config['CKBOX_ALLOWED_EXTENSIONS']
def is_image(fn): return '.' in fn and fn.rsplit('.',1)[1].lower() in app.config['CKBOX_IMAGE_EXTENSIONS']

# --- CKBox Category Helpers ---
def get_category_base_path(category_id=None):
    """Returns the absolute base path for a given categoryId. If None, returns root CKBox path."""
    if category_id and category_id.lower() != 'root' and category_id != '':
        # Prevent path traversal: ensure category_id is a simple name
        safe_category_id = secure_filename(category_id)
        if safe_category_id != category_id: # Check if secure_filename altered it (e.g. removed ../)
             app.logger.warning(f"Potentially unsafe categoryId '{category_id}' was sanitized to '{safe_category_id}'.")
             # Depending on policy, you might reject, or use the sanitized version.
             # For now, using sanitized.
        return os.path.join(app.config['CKBOX_FILE_STORAGE_PATH'], safe_category_id)
    return app.config['CKBOX_FILE_STORAGE_PATH'] # Root storage path

def get_thumbnail_dir_path(category_id=None):
    """Returns the absolute path to the thumbnail directory for a given categoryId."""
    category_base = get_category_base_path(category_id)
    return os.path.join(category_base, app.config['CKBOX_ROOT_THUMBNAIL_DIR_NAME'])

# --- CKBox API Endpoints ---
@app.route('/ckbox/categories', methods=['GET'])
@require_auth_token
def ckbox_list_categories():
    categories = []
    base_storage_path = app.config['CKBOX_FILE_STORAGE_PATH']
    try:
        for item_name in os.listdir(base_storage_path):
            # A category is a directory directly under the base_storage_path
            # And it's not the thumbnail directory itself, nor hidden
            if os.path.isdir(os.path.join(base_storage_path, item_name)) and \
               not item_name.startswith('.') and \
               item_name != app.config['CKBOX_ROOT_THUMBNAIL_DIR_NAME']:
                categories.append({
                    "id": item_name,
                    "name": item_name.replace('_', ' ').replace('-', ' ').title(), # Basic pretty name
                    # "url": url_for('ckbox_list_files', category=item_name) # Optional: direct URL to list files in this cat
                })
        return jsonify({"data": categories}), 200
    except Exception as e:
        app.logger.error(f"Error listing categories: {e}")
        return jsonify({"error": "Could not list categories"}), 500

@app.route('/ckbox/upload', methods=['POST'])
@require_auth_token
def ckbox_upload():
    if 'upload' not in request.files: return jsonify({"error": {"message": "No file part"}}), 400
    file = request.files['upload']; original_fn = secure_filename(file.filename)
    if not original_fn: return jsonify({"error": {"message": "No selected file or filename is invalid"}}), 400

    category_id_form = request.form.get('categoryId')
    current_category_id_str = None # This will be part of the URL path
    if category_id_form and category_id_form.lower() != 'root' and category_id_form != '':
        current_category_id_str = secure_filename(category_id_form)

    storage_directory = get_category_base_path(current_category_id_str)
    thumbnail_directory = get_thumbnail_dir_path(current_category_id_str)

    if not os.path.exists(storage_directory): os.makedirs(storage_directory)

    if file and allowed_file(original_fn):
        file_id = str(uuid.uuid4()); extension = original_fn.rsplit('.',1)[1].lower()
        new_filename = f"{file_id}.{extension}"
        file_abs_path = os.path.join(storage_directory, new_filename)
        thumbnail_url_response = None

        try:
            file.save(file_abs_path)

            # Construct file URL path part (category/filename or just filename for root)
            file_url_path_part = f"{current_category_id_str}/{new_filename}" if current_category_id_str else new_filename

            if is_image(new_filename) and Image is not None and hasattr(Image, 'open'):
                if not os.path.exists(thumbnail_directory): os.makedirs(thumbnail_directory)
                try:
                    img = Image.open(file_abs_path)
                    img.thumbnail(app.config['CKBOX_THUMBNAIL_SIZE'])
                    thumb_format = (img.format or 'png').lower()
                    thumb_filename = f"{file_id}.{thumb_format if thumb_format in ['jpeg','png','gif'] else 'png'}"
                    img.save(os.path.join(thumbnail_directory, thumb_filename), format=img.format or 'PNG')

                    thumb_url_path_parts = [app.config['CKBOX_ROOT_THUMBNAIL_DIR_NAME'], thumb_filename]
                    if current_category_id_str: thumb_url_path_parts.insert(0, current_category_id_str)
                    thumbnail_url_response = url_for('serve_ckbox_asset', filepath="/".join(thumb_url_path_parts), _external=True)
                except Exception as e_thumb:
                    app.logger.error(f"Thumbnail generation error: {e_thumb}")

            response_data = {
                "id": file_id, "name": original_fn,
                "url": url_for('serve_ckbox_asset', filepath=file_url_path_part, _external=True),
                "categoryId": current_category_id_str if current_category_id_str else "root", # CKBox might expect 'root' or null
                "size": os.path.getsize(file_abs_path), "lastModified": os.path.getmtime(file_abs_path)
            }
            if thumbnail_url_response: response_data["thumbnailUrl"] = thumbnail_url_response
            return jsonify(response_data), 201
        except Exception as e:
            app.logger.error(f"CKBox Upload Error: {e}")
            return jsonify({"error": {"message": f"Could not save file: {str(e)}"}}), 500
    else:
        return jsonify({"error": {"message": "File type not allowed or invalid file"}}), 400

@app.route('/ckbox/files', methods=['GET'])
@require_auth_token
def ckbox_list_files():
    category_id_req = request.args.get('category')
    current_category_id_str = None
    if category_id_req and category_id_req.lower() != 'root' and category_id_req != '':
        current_category_id_str = secure_filename(category_id_req)

    storage_directory = get_category_base_path(current_category_id_str)
    thumbnail_directory = get_thumbnail_dir_path(current_category_id_str)
    thumb_dir_name = app.config['CKBOX_ROOT_THUMBNAIL_DIR_NAME']
    files_data_out = []

    if not os.path.exists(storage_directory):
        # If category dir doesn't exist, return empty list for that category
        return jsonify({"data": files_data_out, "total": 0, "categoryId": current_category_id_str if current_category_id_str else "root"}), 200

    for item_name in sorted(os.listdir(storage_directory)):
        if item_name == thumb_dir_name or item_name.startswith('.'): continue # Skip thumbnail dir and hidden files

        item_abs_path = os.path.join(storage_directory, item_name)
        if os.path.isfile(item_abs_path):
            file_id_str, _ = os.path.splitext(item_name)
            file_url_path_part = f"{current_category_id_str}/{item_name}" if current_category_id_str else item_name
            thumbnail_url_response = None

            if is_image(item_name) and os.path.exists(thumbnail_directory):
                for ext_check in ['png','jpg','jpeg','gif']: # Check for various thumbnail extensions
                    potential_thumb_name = f"{file_id_str}.{ext_check}"
                    if os.path.exists(os.path.join(thumbnail_directory, potential_thumb_name)):
                        thumb_url_path_parts = [thumb_dir_name, potential_thumb_name]
                        if current_category_id_str: thumb_url_path_parts.insert(0, current_category_id_str)
                        thumbnail_url_response = url_for('serve_ckbox_asset', filepath="/".join(thumb_url_path_parts), _external=True)
                        break

            files_data_out.append({
                "id": file_id_str, "name": item_name, # Using full item_name as name, original_fn would need metadata storage
                "categoryId": current_category_id_str if current_category_id_str else "root",
                "url": url_for('serve_ckbox_asset', filepath=file_url_path_part, _external=True),
                "size": os.path.getsize(item_abs_path), "lastModified": os.path.getmtime(item_abs_path),
                "thumbnailUrl": thumbnail_url_response
            })
    return jsonify({"data": files_data_out, "total": len(files_data_out), "categoryId": current_category_id_str if current_category_id_str else "root"}), 200

@app.route('/ckbox/files/<string:file_id_to_delete>', methods=['DELETE'])
@require_auth_token
def ckbox_delete_file(file_id_to_delete):
    base_storage_path = app.config['CKBOX_FILE_STORAGE_PATH']
    thumbnail_dir_name_const = app.config['CKBOX_ROOT_THUMBNAIL_DIR_NAME']

    # Determine category from request if provided, else search all
    category_id_req = request.args.get('categoryId') # CKBox might send categoryId for delete
    search_paths = []
    if category_id_req and category_id_req.lower() != 'root' and category_id_req != '':
        search_paths.append(get_category_base_path(secure_filename(category_id_req)))
    else: # Search root and all first-level category directories
        search_paths.append(base_storage_path) # Root
        for cat_name in os.listdir(base_storage_path):
            potential_cat_path = os.path.join(base_storage_path, cat_name)
            if os.path.isdir(potential_cat_path) and not cat_name.startswith('.') and cat_name != thumbnail_dir_name_const:
                search_paths.append(potential_cat_path)

    deleted_successfully = False
    for current_search_dir in search_paths:
        if not os.path.exists(current_search_dir): continue
        for item_filename in os.listdir(current_search_dir):
            if item_filename == thumbnail_dir_name_const or not os.path.isfile(os.path.join(current_search_dir, item_filename)):
                continue # Skip thumbnail directory itself or non-files

            current_file_id, _ = os.path.splitext(item_filename)
            if current_file_id == file_id_to_delete:
                try:
                    os.remove(os.path.join(current_search_dir, item_filename))
                    # Attempt to delete its thumbnail(s) from the category's .thumbnails dir
                    current_thumbnail_dir = os.path.join(current_search_dir, thumbnail_dir_name_const)
                    if os.path.exists(current_thumbnail_dir):
                        for ext_del in ['png','jpg','jpeg','gif']:
                            thumb_to_delete_path = os.path.join(current_thumbnail_dir, f"{file_id_to_delete}.{ext_del}")
                            if os.path.exists(thumb_to_delete_path): os.remove(thumb_to_delete_path)
                    deleted_successfully = True; break
                except OSError as e:
                    app.logger.error(f"CKBox Delete Error: {e}")
                    return jsonify({"error": {"message": f"Error deleting file: {str(e)}"}}), 500
        if deleted_successfully: break

    return ('', 204) if deleted_successfully else (jsonify({"error": {"message": "File not found or category hint mismatch"}}), 404)

@app.route('/ckbox_assets/<path:filepath>') # Renamed from serve_ckbox_file and serve_ckbox_thumbnail
def serve_ckbox_asset(filepath):
    # This single endpoint serves both main files and thumbnails.
    # filepath will be like "category_name/file.jpg" or "file.jpg" (for root)
    # or "category_name/.thumbnails/file.png" or ".thumbnails/file.png" (for root thumbnails)
    return send_from_directory(app.config['CKBOX_FILE_STORAGE_PATH'], filepath)

# --- Collaboration Server & Main (Placeholders, no changes from previous step) ---
def save_collab_document(d,s):pass
def broadcast_collab_message(s,m,e=None):pass
@sockets.route('/collaboration/ws/<string:doc_id>')
def collaboration_socket(ws,doc_id): return
@app.route('/telemetry',methods=['POST'])
@require_auth_token # Assuming telemetry might also want to be secured
def telemetry_receive(): data=request.get_json();telemetry_logger.info(json.dumps(data,indent=2));return jsonify({"status":"success"}),200
@app.route('/')
def hello_world(): return 'Hello CKEditor Backend!'
if __name__=='__main__':
    if not app.debug and not app.testing: # Avoid basicConfig if pytest or other runners configure logging
        logging.basicConfig(level=logging.INFO)
    app.logger.info("Flask App Starting...")
    app.run(host='0.0.0.0',port=8000,debug=True)
