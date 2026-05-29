import base64
import io
import json
import random
import requests
from datetime import datetime
from decimal import Decimal, InvalidOperation

import bcrypt
import boto3
import folium
import streamlit as st
import streamlit.components.v1 as components
try:
    from audiorecorder import audiorecorder
    _RECORDER_OK = True
except Exception:
    _RECORDER_OK = False
    audiorecorder = None
from geopy.geocoders import Nominatim
from PIL import Image
from streamlit_carousel import carousel
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from streamlit_image_select import image_select

# ── AWS clients ────────────────────────────────────────────────────────────────

def _aws_credentials():
    return {
        "aws_access_key_id": st.secrets["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": st.secrets["AWS_SECRET_ACCESS_KEY"],
    }

@st.cache_resource
def _dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=st.secrets["AWS_REGION"],
        **_aws_credentials(),
    )

@st.cache_resource
def _s3():
    return boto3.client(
        "s3",
        region_name=st.secrets["AWS_REGION"],
        **_aws_credentials(),
    )

@st.cache_resource
def _sagemaker():
    return boto3.client(
        "sagemaker-runtime",
        region_name=st.secrets["AWS_SAGEMAKER_REGION"],
        **_aws_credentials(),
    )

# ── DynamoDB tables ────────────────────────────────────────────────────────────

def table_record():
    return _dynamodb().Table("RibbitRecordings")

def frog_table():
    return _dynamodb().Table("FrogHistory")

def user_table():
    return _dynamodb().Table("UserCredentials")

def species_table():
    return _dynamodb().Table("SpeciesData")

# ── Inference ──────────────────────────────────────────────────────────────────

def call_sagemaker(audio_bytes: bytes) -> list:
    payload = json.dumps({"audio_b64": base64.b64encode(audio_bytes).decode()})
    response = _sagemaker().invoke_endpoint(
        EndpointName=st.secrets["SAGEMAKER_ENDPOINT"],
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read())
    return [(p["species"], float(p["confidence"])) for p in result["predictions"]]

# ── Translation (graceful fallback to English) ─────────────────────────────────

LANGUAGES = {"English": "en", "Español": "es", "Português": "pt", "العربية": "ar"}

def load_translation(language_code):
    try:
        import gettext
        t = gettext.translation("ribbit", localedir="locale", languages=[language_code])
        t.install()
        return t.gettext
    except FileNotFoundError:
        import gettext
        return gettext.gettext

# ── Session state init ─────────────────────────────────────────────────────────

if "navigation_stack" not in st.session_state:
    st.session_state["navigation_stack"] = []
if "language_code" not in st.session_state:
    st.session_state["language_code"] = "en"
if "_" not in st.session_state:
    st.session_state["_"] = load_translation(st.session_state["language_code"])

_ = st.session_state["_"]

# ── Sidebar ────────────────────────────────────────────────────────────────────

try:
    logo = Image.open("ribbit_logo.png")
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.title("🐸 Ribbit")

def select_language_and_navigation():
    st.sidebar.title(_("Select Language"))
    selected_language = st.sidebar.selectbox(
        _("Language"),
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.values()).index(st.session_state["language_code"]),
    )
    language_code = LANGUAGES[selected_language]
    if st.session_state["language_code"] != language_code:
        st.session_state["language_code"] = language_code
        st.session_state["_"] = load_translation(language_code)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader(_("Navigation"))
    for label, page in [
        (_("Home"), "home"),
        (_("My Recordings"), "recordings"),
        (_("My Frogs"), "frogs"),
        (_("Explore Species"), "explore"),
    ]:
        if st.sidebar.button(label, key=f"nav_{page}"):
            st.session_state["navigate_to"] = page
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader(_("Settings"))
    if st.sidebar.button(_("Privacy Settings"), key="nav_privacy"):
        st.session_state["navigate_to"] = "privacy_policy"
        st.rerun()
    if st.sidebar.button(_("FAQ"), key="nav_faq"):
        st.session_state["navigate_to"] = "faq"
        st.rerun()
    if st.sidebar.button(_("About"), key="nav_about"):
        st.session_state["navigate_to"] = "about"
        st.rerun()

# ── Auth ───────────────────────────────────────────────────────────────────────

def check_password(username, password):
    try:
        response = user_table().get_item(Key={"UserID": username})
        if "Item" in response:
            user = response["Item"]
            if bcrypt.checkpw(password.encode(), user["password"].encode()):
                return user
        return None
    except Exception as e:
        st.error(f"{_('Error checking credentials')}: {e}")
        return None

def login_form():
    st.title(_("Login to Ribbit"))
    username = st.text_input(_("Username"))
    password = st.text_input(_("Password"), type="password")
    if st.button(_("Login")):
        user_data = check_password(username, password)
        if user_data:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["user_data"] = user_data
            st.success(f"{_('Welcome')} {username}")
            st.session_state.page = (
                "home" if user_data.get("privacy_policy_accepted", False) else "privacy_policy"
            )
            st.rerun()
        else:
            st.error(_("Invalid username or password"))
    st.markdown("---")
    if st.button(_("Create Account")):
        st.session_state.page = "register"
        st.rerun()

def register_form():
    _ = st.session_state["_"]
    st.title(_("Create a New Account"))
    username = st.text_input(_("Choose a Username"))
    email = st.text_input(_("Email"))
    name = st.text_input(_("Full Name"))
    password = st.text_input(_("Password"), type="password")
    confirm_password = st.text_input(_("Confirm Password"), type="password")

    if st.button(_("Register")):
        if not all([username, email, name, password, confirm_password]):
            st.error(_("Please fill in all fields."))
            return
        if password != confirm_password:
            st.error(_("Passwords do not match."))
            return
        try:
            if "Item" in user_table().get_item(Key={"UserID": username}):
                st.error(_("Username already exists. Please choose a different username."))
                return
        except Exception as e:
            st.error(f"{_('Error checking username')}: {e}")
            return
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        try:
            user_table().put_item(Item={
                "UserID": username, "email": email, "name": name,
                "password": hashed.decode("utf-8"), "privacy_policy_accepted": False,
            })
            st.success(_("Account created successfully. Please accept the Privacy Policy."))
            st.session_state.update({
                "logged_in": True, "username": username,
                "user_data": {"UserID": username, "privacy_policy_accepted": False},
                "page": "privacy_policy",
            })
            st.rerun()
        except Exception as e:
            st.error(f"{_('Error creating account')}: {e}")

# ── S3 helpers ─────────────────────────────────────────────────────────────────

def upload_to_s3(user_id, file_name, file_data):
    try:
        _s3().put_object(
            Bucket="ribbit-recordings",
            Key=f"{user_id}/{file_name}",
            Body=file_data,
            ContentType="audio/wav",
        )
        return True
    except Exception as e:
        st.error(f"{_('Error uploading file to S3')}: {e}")
        return False

def list_recordings(user_id):
    try:
        response = _s3().list_objects_v2(Bucket="ribbit-recordings", Prefix=f"{user_id}/")
        return [obj["Key"] for obj in response.get("Contents", [])]
    except Exception as e:
        st.error(f"{_('Error fetching recordings from S3')}: {e}")
        return []

def generate_presigned_url(full_key):
    try:
        return _s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": "ribbit-recordings", "Key": full_key},
            ExpiresIn=3600,
        )
    except Exception as e:
        st.error(f"{_('Error generating pre-signed URL')}: {e}")
        return None

def get_species_audio_url(species_name):
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    try:
        response = _s3().list_objects_v2(
            Bucket="ribbit-species-sounds",
            Prefix=f"species_single_files/{species_name}/",
        )
        for obj in response.get("Contents", []):
            if any(obj["Key"].lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
                return _s3().generate_presigned_url(
                    "get_object",
                    Params={"Bucket": "ribbit-species-sounds", "Key": obj["Key"]},
                    ExpiresIn=3600,
                )
    except Exception as e:
        print(f"Error generating audio URL for {species_name}: {e}")
    return None

# ── DynamoDB helpers ───────────────────────────────────────────────────────────

def safe_decimal(value):
    try:
        return Decimal(str(value)) if value not in (None, "") else None
    except (InvalidOperation, TypeError, ValueError):
        return None

def save_metadata_to_dynamodb(user_id, recording_id, metadata):
    try:
        table_record().put_item(Item={
            "UserID": user_id,
            "RecordingID": recording_id,
            "Timestamp": metadata["timestamp"],
            "Latitude": safe_decimal(metadata.get("latitude")),
            "Longitude": safe_decimal(metadata.get("longitude")),
        })
        return True
    except Exception as e:
        st.error(f"{_('Error saving to frog history')}: {e}")
        return False

def get_frog_history(user_id):
    try:
        response = frog_table().query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("UserID").eq(user_id),
            ScanIndexForward=False,
        )
        return response.get("Items", [])
    except Exception as e:
        st.error(f"{_('Error fetching frog history from DynamoDB')}: {e}")
        return []

def get_species_data(species_id):
    try:
        return species_table().get_item(Key={"species": species_id}).get("Item")
    except Exception as e:
        st.error(f"Error fetching species data for {species_id}: {e}")
        return None

def fetch_all_species():
    try:
        return species_table().scan().get("Items", [])
    except Exception as e:
        st.error(f"Error fetching species data from DynamoDB: {e}")
        return []

def get_location_name(latitude, longitude):
    try:
        loc = Nominatim(user_agent="ribbitapp").reverse((latitude, longitude), timeout=10)
        return loc.address if loc else "Unknown Location"
    except Exception:
        return "Unknown Location"

# ── Audio helpers ──────────────────────────────────────────────────────────────

def export_audio_data(audio_segment):
    buf = io.BytesIO()
    audio_segment.export(buf, format="wav")
    return buf.getvalue()

# ── Wikipedia helpers ──────────────────────────────────────────────────────────

def _wiki_api_extract(title, lang):
    """Single MediaWiki API call; returns extract string or None."""
    try:
        resp = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "prop": "extracts",
                "explaintext": 1, "exintro": 1,
                "titles": title, "redirects": 1,
            },
            headers={"User-Agent": "RibbitApp/1.0 (educational)"},
            timeout=6,
        )
        resp.raise_for_status()
        for page in resp.json()["query"]["pages"].values():
            if "missing" not in page:
                extract = page.get("extract", "").strip()
                if extract:
                    return extract
    except Exception:
        pass
    return None

def fetch_wikipedia_content(frog_name, lang="en"):
    """Fetch intro via MediaWiki API; tries original name then sentence-case variant."""
    title = " ".join(frog_name.strip().split())
    # Sentence case: "Northern Cricket Frog" → "Northern cricket frog"
    sentence = title[0].upper() + title[1:].lower() if len(title) > 1 else title.upper()
    for t in dict.fromkeys([title, sentence]):
        result = _wiki_api_extract(t, lang)
        if result:
            return result
    return None

def extract_section(content, section_name):
    if not content:
        return None
    header = f"== {section_name} =="
    start = content.find(header)
    if start == -1:
        return None
    start += len(header)
    end = content.find("==", start)
    section = (content[start:end] if end != -1 else content[start:]).strip()
    return section or None

def clean_text(text):
    return text.replace("", "").replace("\n", " ").strip() if text else None

def fetch_wikipedia_full(frog_name, lang="en"):
    """Fetch full article text (for section extraction)."""
    try:
        resp = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={"action": "query", "format": "json", "prop": "extracts",
                    "explaintext": 1, "titles": frog_name, "redirects": 1},
            timeout=5,
        )
        pages = resp.json()["query"]["pages"]
        for page in pages.values():
            extract = page.get("extract", "").strip()
            if extract:
                return extract
    except Exception:
        pass
    return None

def get_wikipedia_details(frog_names):
    results = []
    for name in frog_names:
        intro = None
        full = None
        for lang in ["en", "es", "pt", "ar"]:
            intro = fetch_wikipedia_content(name, lang)  # exintro=1
            if intro:
                full = fetch_wikipedia_full(name, lang)
                break
        results.append({
            "name": name,
            "summary": clean_text(intro),
            "description": clean_text(extract_section(full, "Description")),
            "habitat": clean_text(extract_section(full, "Distribution and habitat")),
        })
    return results

# ── Species selection helper ───────────────────────────────────────────────────

def select_species(species_data, from_model_output=False):
    st.session_state["selected_species"] = species_data
    st.session_state["navigate_to"] = "species_details"
    st.session_state["from_model_output"] = from_model_output
    st.rerun()

# ── Prediction display helpers ─────────────────────────────────────────────────

def display_species_carousel(species_predictions, carousel_key):
    items, species_list = [], []
    for prediction in species_predictions:
        species_name = prediction.get("SpeciesName", "Unknown")
        data = get_species_data(species_name) or {}
        species_list.append(data)
        items.append({
            "title": species_name, "text": "",
            "img": data.get("image_url", "https://via.placeholder.com/200"),
        })
    if items:
        carousel(items=items, key=carousel_key)
        st.write(f"**{_('Select a species to view details')}:**")
        for data in species_list:
            name = data.get("species", "Unknown")
            if st.button(name, key=f"btn_{name}_{carousel_key}"):
                st.session_state["selected_species"] = data
                st.session_state["navigate_to"] = "species_details"
                st.rerun()

def _show_prediction_results(top_5, step_key, confirm_key):
    _ = st.session_state["_"]
    st.write(_("Top 5 Predicted Species:"))
    for species_name, confidence in top_5:
        data = get_species_data(species_name) or {}
        image_url = data.get("image_url", "https://via.placeholder.com/100")
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(image_url, width=80)
        with col2:
            st.button(species_name, key=f"btn_{species_name}_{step_key}",
                      on_click=select_species, args=(data, True))

    st.markdown(f"### {_('Is one of these your Frog?')}")
    c1, c2, c3 = st.columns(3)
    for col, label, val in [
        (c1, _("Yes"), "Yes"),
        (c2, _("No"), "No"),
        (c3, _("I don't know"), "I don't know"),
    ]:
        with col:
            if st.button(label, key=f"{confirm_key}_{val}"):
                st.session_state[confirm_key] = val
                st.session_state[step_key] = "add_notes"

def _show_add_notes(step_key, confirm_key, metadata_key, predictions_key,
                    filename_key, show_key):
    """Shared notes + save encounter UI."""
    _ = st.session_state["_"]
    metadata = st.session_state.get(metadata_key)
    top_5 = st.session_state.get(predictions_key)
    file_name = st.session_state.get(filename_key)

    if not (metadata and top_5 and file_name):
        st.error(_("Required data is missing. Please start over."))
        return

    location_name = (
        get_location_name(metadata["latitude"], metadata["longitude"])
        if metadata.get("latitude") and metadata.get("longitude")
        else "Unknown Location"
    )

    notes_key = f"notes_{step_key}"
    if notes_key not in st.session_state:
        st.session_state[notes_key] = ""
    st.text_input(_("Add any notes about your encounter (optional)"), key=notes_key)

    if st.button(_("Save Encounter"), key=f"save_{step_key}"):
        user_id = st.session_state.get("username", "guest")
        item = {
            "UserID": user_id,
            "EncounterTimestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "RecordingID": file_name,
            "SpeciesPredictions": [
                {"SpeciesName": s, "Confidence": Decimal(str(c))} for s, c in top_5
            ],
            "Latitude": safe_decimal(metadata.get("latitude")),
            "Longitude": safe_decimal(metadata.get("longitude")),
            "LocationName": location_name,
            "Notes": st.session_state[notes_key] or None,
            "UserConfirmed": st.session_state.get(confirm_key),
        }
        try:
            frog_table().put_item(Item=item)
            st.success(_("Encounter saved to your frog history."))
            for key in [confirm_key, metadata_key, predictions_key, filename_key, notes_key]:
                st.session_state.pop(key, None)
            st.session_state[step_key] = "initial"
            st.session_state[show_key] = False
            st.rerun()
        except Exception as e:
            st.error(f"{_('Error saving to frog history')}: {e}")

# ── Pages ──────────────────────────────────────────────────────────────────────

def show_home_page():
    _ = st.session_state["_"]
    user_id = st.session_state.get("username", "guest")
    user_full_name = st.session_state.get("user_data", {}).get("name", user_id)

    # Frog of the day
    species_list = fetch_all_species()
    if species_list:
        if "frog_of_the_day" not in st.session_state:
            st.session_state["frog_of_the_day"] = random.choice(species_list)
        frog = st.session_state["frog_of_the_day"]
        lang = st.session_state.get("language_code", "en")

        def localized(field, default="N/A"):
            val = frog.get(field, default)
            return val.get(lang, default) if isinstance(val, dict) else val

        st.markdown(f"<h2 style='text-align:center'>{_('Frog of the day')}</h2>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;justify-content:center'>"
            f"<img src='{frog.get('image_url', '')}' style='max-width:400px;height:auto'>"
            "</div>", unsafe_allow_html=True,
        )
        if frog.get("image_attribution"):
            st.markdown(f"<div style='text-align:center'><small>{frog['image_attribution']}</small></div>",
                        unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;font-size:18px'>{localized('common_family_name')}</p>",
                    unsafe_allow_html=True)
        audio_url = get_species_audio_url(frog["species"])
        if audio_url:
            st.audio(audio_url)
        _l, col, _r = st.columns([2, 2, 2])
        with col:
            if st.button(frog["species"], key="frog_of_the_day_btn"):
                select_species(frog)

    st.markdown(f"<h3 style='text-align:center'>{_('Welcome back')}, {user_full_name}!</h3>",
                unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align:center'>{_('Clip it, Ribbit!')}</h1>",
                unsafe_allow_html=True)

    for key in ["show_record", "show_upload", "current_step_record", "current_step_upload"]:
        if key not in st.session_state:
            st.session_state[key] = False if "show" in key else "initial"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎙️ " + _("Start Frogging!"), key="record_now"):
            st.session_state["show_record"] = not st.session_state["show_record"]
            st.session_state["show_upload"] = False
            st.session_state["current_step_record"] = "initial"
    with col2:
        if st.button("📂 " + _("Upload and Locate Yourself"), key="upload_locate"):
            st.session_state["show_upload"] = not st.session_state["show_upload"]
            st.session_state["show_record"] = False
            st.session_state["current_step_upload"] = "initial"

    st.markdown("---")

    # ── Record section ─────────────────────────────────────────────────────────
    if st.session_state["show_record"] or st.session_state["current_step_record"] != "initial":
        st.markdown(f"<h4>{_('Instructions')}:</h4>", unsafe_allow_html=True)
        st.markdown(
            f"1. **{_('Share your location')}** {_('by clicking the location icon below')}.\n"
            f"2. **{_('Record frog sounds')}** {_('by clicking the record button below')}.",
            unsafe_allow_html=True,
        )
        location = streamlit_geolocation()
        if location and location.get("latitude") is not None:
            st.write(f"{_('Location captured')}: {location['latitude']}, {location['longitude']}")
        else:
            st.write(_("No location information available until permission is granted."))

        if not _RECORDER_OK:
            st.warning(_("Live recording is not available in this environment. Please use the Upload option instead."))
            audio = None
        else:
            audio = audiorecorder(_("Click to record"), _("Click to stop recording"))

        if audio and len(audio) > 0 and "recording_uploaded" not in st.session_state:
            audio_data = export_audio_data(audio)
            st.audio(audio_data, format="audio/wav")
            file_name = "audio_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
            st.session_state["recording_uploaded"] = True

            if st.session_state["user_data"].get("privacy_policy_accepted", False):
                if upload_to_s3(user_id, file_name, audio_data):
                    st.success(f"{_('Recording uploaded')} — {file_name}")
                    latitude = location.get("latitude") if location else None
                    longitude = location.get("longitude") if location else None
                    metadata = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "latitude": latitude, "longitude": longitude,
                    }
                    save_metadata_to_dynamodb(user_id, file_name, metadata)

                    with st.spinner(_("Identifying species…")):
                        try:
                            top_5 = call_sagemaker(audio_data)
                            st.session_state.update({
                                "metadata_record": metadata,
                                "top_5_predictions_record": top_5,
                                "file_name_record": file_name,
                                "current_step_record": "confirmation",
                            })
                        except Exception as e:
                            st.error(f"Inference error: {e}")
                else:
                    st.error(_("Failed to upload recording to S3."))
            else:
                st.error(_("You have not agreed to the privacy policy. Data will not be saved."))

        elif audio is not None and len(audio) == 0:
            st.session_state.pop("recording_uploaded", None)

        if st.session_state.get("current_step_record") == "confirmation":
            top_5 = st.session_state.get("top_5_predictions_record")
            if top_5:
                _show_prediction_results(top_5, "current_step_record",
                                         "user_confirmed_record")
            else:
                st.error("No predictions available.")

        if st.session_state.get("current_step_record") == "add_notes":
            _show_add_notes("current_step_record", "user_confirmed_record",
                            "metadata_record", "top_5_predictions_record",
                            "file_name_record", "show_record")

    # ── Upload section ─────────────────────────────────────────────────────────
    if st.session_state["show_upload"] or st.session_state["current_step_upload"] != "initial":
        st.markdown(f"## {_('Upload Saved Recording')}")
        st.write(_("If you recorded a frog call in a remote location, upload it here."))

        uploaded_file = st.file_uploader(_("Choose an audio file"), type=["wav", "mp3", "m4a"])

        st.write(_("Select the location of your recording on the map."))
        geolocator = Nominatim(user_agent="RibbitApp/1.0 (nakabayashi.erica@gmail.com)")
        location_query = st.text_input(_("Enter a city, country, or location to center the map"))
        center_coords, zoom_level = [37.0902, -95.7129], 4
        if location_query:
            try:
                loc = geolocator.geocode(location_query, timeout=5)
                if loc:
                    center_coords = [loc.latitude, loc.longitude]
                    zoom_level = 12
                    st.write(f"{_('Location found')}: {loc.address}")
                else:
                    st.warning(_("Location not found. Click on the map to set your location."))
            except Exception:
                st.warning(_("Location search unavailable. Click on the map to set your location."))

        m = folium.Map(location=center_coords, zoom_start=zoom_level)
        map_data = st_folium(m, width=700, height=400)
        latitude = longitude = None
        if map_data and map_data.get("last_clicked"):
            latitude = map_data["last_clicked"]["lat"]
            longitude = map_data["last_clicked"]["lng"]
        st.write(f"{_('Selected Latitude')}: {latitude}")
        st.write(f"{_('Selected Longitude')}: {longitude}")

        if st.button(_("Upload Recording"), key="upload_recording"):
            if uploaded_file is None:
                st.error(_("Please select an audio file to upload."))
            elif not st.session_state["user_data"].get("privacy_policy_accepted", False):
                st.error(_("You have not agreed to the privacy policy. Data will not be saved."))
            else:
                uploaded_file.seek(0, 2)
                if uploaded_file.tell() / (1024 * 1024) > 2:
                    st.error(_("File too large. Please upload a file smaller than 2 MB."))
                    uploaded_file.seek(0)
                else:
                    uploaded_file.seek(0)
                    audio_data = uploaded_file.read()
                    file_name = "manual_upload_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"

                    if upload_to_s3(user_id, file_name, audio_data):
                        st.success(_("Recording uploaded."))
                        metadata = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "latitude": latitude, "longitude": longitude,
                        }
                        save_metadata_to_dynamodb(user_id, file_name, metadata)

                        with st.spinner(_("Identifying species…")):
                            try:
                                top_5 = call_sagemaker(audio_data)
                                st.session_state.update({
                                    "metadata_upload": metadata,
                                    "top_5_predictions_upload": top_5,
                                    "file_name_upload": file_name,
                                    "current_step_upload": "confirmation",
                                })
                            except Exception as e:
                                st.error(f"Inference error: {e}")
                    else:
                        st.error(_("Failed to upload recording to S3."))

        if st.session_state.get("current_step_upload") == "confirmation":
            top_5 = st.session_state.get("top_5_predictions_upload")
            if top_5:
                _show_prediction_results(top_5, "current_step_upload",
                                         "user_confirmed_upload")
            else:
                st.error(_("No predictions available."))

        if st.session_state.get("current_step_upload") == "add_notes":
            _show_add_notes("current_step_upload", "user_confirmed_upload",
                            "metadata_upload", "top_5_predictions_upload",
                            "file_name_upload", "show_upload")


def show_recordings_list():
    _ = st.session_state["_"]
    st.title(_("My Recordings"))
    user_id = st.session_state.get("username", "guest")
    recordings = list_recordings(user_id)
    if recordings:
        for key in recordings:
            file_name = key.split("/", 1)[1]
            url = generate_presigned_url(key)
            if url:
                st.write(f"**{file_name}**")
                st.audio(url)
    else:
        st.write(_("No recordings found."))


def show_frog_history():
    _ = st.session_state["_"]
    st.title(_("My Frogs"))
    user_id = st.session_state.get("username", "guest")
    history = get_frog_history(user_id)
    if history:
        for idx, encounter in enumerate(history):
            with st.container():
                st.write(f"**{_('Encountered on')}:** {encounter['EncounterTimestamp']}")
                if "LocationName" in encounter:
                    st.write(f"**{_('Location')}:** {encounter['LocationName']}")
                if encounter.get("Notes"):
                    st.write(f"**{_('Notes')}:** {encounter['Notes']}")
                if "UserConfirmed" in encounter:
                    st.write(f"**{_('User Confirmed')}:** {encounter['UserConfirmed']}")
                preds = encounter.get("SpeciesPredictions", [])
                if preds:
                    display_species_carousel(preds, carousel_key=f"carousel_{idx}")
                st.markdown("---")
    else:
        st.write(_("No frog encounters found."))


def show_explore():
    _ = st.session_state["_"]
    st.title(_("Explore Species"))
    search_query = st.text_input(_("Search for a species"), "", key="explore_search").strip().lower()

    if search_query != st.session_state.get("previous_search_query", ""):
        st.session_state["initial_load_complete"] = False
        st.session_state["previous_selection"] = None
        st.session_state["previous_search_query"] = search_query

    species_list = sorted(fetch_all_species(), key=lambda x: x.get("species", "").lower())
    filtered = [s for s in species_list if search_query in s.get("species", "").lower()]
    if not filtered:
        st.write(_("No species found matching the search criteria."))
        return

    selected_idx = image_select(
        label="",
        images=[s.get("image_url", "https://via.placeholder.com/100") for s in filtered],
        captions=[s["species"] for s in filtered],
        use_container_width=True,
        return_value="index",
        key=f"explore_{search_query}_{len(filtered)}",
    )
    if selected_idx != st.session_state.get("previous_selection"):
        st.session_state["previous_selection"] = selected_idx
        if st.session_state.get("initial_load_complete", False):
            select_species(filtered[selected_idx])
        else:
            st.session_state["initial_load_complete"] = True


def display_species_details(species):
    _ = st.session_state["_"]
    if not species:
        st.error(_("No species selected."))
        return

    st.subheader(species["species"])
    st.image(species.get("image_url", "https://via.placeholder.com/400"), use_container_width=True)
    if species.get("image_attribution"):
        st.markdown(f"<div style='text-align:center'><small>{species['image_attribution']}</small></div>",
                    unsafe_allow_html=True)
    if species.get("image_license"):
        st.markdown(f"<div style='text-align:center'><small>License: {species['image_license']}</small></div>",
                    unsafe_allow_html=True)

    audio_url = get_species_audio_url(species["species"])
    if audio_url:
        st.audio(audio_url)

    lang = st.session_state.get("language_code", "en")
    def localized(field, default="N/A"):
        val = species.get(field, default)
        # Some fields stored as JSON strings instead of DynamoDB Maps
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    val = parsed
            except Exception:
                pass
        if isinstance(val, dict):
            inner = val.get(lang, val.get("en", default))
            # Handle DynamoDB low-level {"S": "value"} nesting
            return inner.get("S", default) if isinstance(inner, dict) else inner
        return val

    for label, field in [
        (_("Common Name"), "common_name"),
        (_("Family"), "common_family_name"),
        (_("Threat Status"), "threat_status"),
        (_("Trade Information"), "trade_dictionary"),
    ]:
        st.write(f"**{label}:** {localized(field)}")

    # Try scientific name first, then localized common name as fallback
    search_names = [species["species"]]
    raw_common = species.get("common_name")
    common_name_en = (raw_common.get("en") if isinstance(raw_common, dict) else raw_common)
    if common_name_en and common_name_en not in search_names:
        search_names.append(common_name_en)

    wiki = get_wikipedia_details(search_names)
    bio = next((x for x in wiki if any(x.get(k) for k in ("summary", "description", "habitat"))), None)
    if bio:
        st.write(_("Wikipedia Summary:"))
        for label, key in [(_("Summary"), "summary"), (_("Description"), "description"),
                            (_("Habitat"), "habitat")]:
            if bio.get(key):
                st.write(f"**{label}:** {bio[key]}")
    else:
        st.write(_("No Wikipedia data available for this species."))


    if st.session_state.get("from_model_output", False):
        if st.button(_("Back to Results"), key="back_to_results"):
            st.session_state["page"] = "home"
            st.session_state["from_model_output"] = False
            st.rerun()
    else:
        if st.button(_("Back to Explore"), key="back_to_explore"):
            st.session_state["selected_species"] = None
            st.session_state.page = "explore"
            st.rerun()
        if st.button(_("Back to Homepage"), key="back_to_home"):
            st.session_state["selected_species"] = None
            st.session_state.page = "home"
            st.rerun()


def show_privacy_policy():
    _ = st.session_state["_"]
    st.title(_("Privacy Policy"))
    st.markdown(_("""
    ### Welcome to Ribbit!

    Your privacy is important to us. This Privacy Policy outlines how we collect, use, and protect your information.

    ### Information We Collect
    1. Audio recordings of frog calls that you capture and upload through the app.
    2. Location data where recordings are made, if you allow location services.
    3. User data such as your name and email address.
    """))
    agree = st.checkbox(_("I agree to the Privacy Policy"), value=False)
    if st.button(_("Save")):
        if agree:
            user_id = st.session_state.get("username")
            if user_id:
                try:
                    user_table().update_item(
                        Key={"UserID": user_id},
                        UpdateExpression="SET privacy_policy_accepted = :val",
                        ExpressionAttributeValues={":val": True},
                    )
                    st.session_state["user_data"]["privacy_policy_accepted"] = True
                    st.success(_("Privacy settings updated."))
                    st.session_state.page = "home"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating user data: {e}")
        else:
            st.error(_("You must agree to the Privacy Policy to continue."))


def show_faq():
    _ = st.session_state["_"]
    st.title(_("Frequently Asked Questions (FAQ)"))
    faqs = [
        (_("What is Ribbit?"), _("Ribbit is a web application for automatic identification of frogs and toads using AI.")),
        (_("How does the app identify frogs?"), _("We use BirdNET (Cornell Lab of Ornithology) embeddings with an ensemble classifier trained on 71 frog species.")),
        (_("Can I upload external recordings?"), _("Yes! Use the 'Upload and Locate Yourself' button on the home page.")),
        (_("Do I need phone reception to record?"), _("No — record offline and upload later when you have a connection.")),
        (_("How do I contact the Ribbit team?"), _("Reach us at julianagc@berkeley.edu")),
        (_("Can I delete my account?"), _("Contact julianagc@berkeley.edu to request account deletion.")),
    ]
    for question, answer in faqs:
        st.markdown(f"#### {question}")
        st.write(answer)
        st.markdown("---")


def show_about():
    _ = st.session_state["_"]
    st.title(_("About"))
    st.markdown(_("""
    Our team developed Ribbit as our capstone project for the Masters in Information and Data Science (MIDS) program at UC Berkeley.
    Lia Cappellari was in charge of modeling, Farouk Ghandour in charge of data engineering,
    Erica Nakabayashi in charge of ML engineering, Haissam Akhras in charge of our MVP,
    and Juliana Gómez Consuegra was the product manager and subject matter expert.
    """))

# ── Router ─────────────────────────────────────────────────────────────────────

def page_router():
    if "page" not in st.session_state:
        st.session_state["page"] = "login"
    else:
        nav = st.session_state.get("navigate_to")
        if nav and nav != st.session_state["page"]:
            st.session_state["navigation_stack"].append(st.session_state["page"])
            st.session_state["page"] = nav
            st.session_state["navigate_to"] = None

    page = st.session_state["page"]

    if page == "login":
        if st.session_state.get("logged_in"):
            st.session_state["page"] = "home"
            st.rerun()
        else:
            login_form()
    elif page == "register":
        register_form()
    elif page == "home":
        if not st.session_state.get("logged_in"):
            st.session_state["page"] = "login"
            st.rerun()
        else:
            show_home_page()
    elif page == "recordings":
        show_recordings_list()
    elif page == "frogs":
        show_frog_history()
    elif page == "explore":
        show_explore()
    elif page == "privacy_policy":
        show_privacy_policy()
    elif page == "species_details":
        display_species_details(st.session_state.get("selected_species"))
    elif page == "faq":
        show_faq()
    elif page == "about":
        show_about()


select_language_and_navigation()
page_router()
