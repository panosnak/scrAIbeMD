import streamlit as st
import os
import base64
from pathlib import Path
import atexit
from speech_to_text import speech_to_text
from create_soap import generate_soap_note

# Title of the app
st.title("Voice Recorder, Transcription, and SOAP Note Generator")

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose an option:", ["Record Audio", "Transcribe & Generate SOAP Note"])

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    save_path = Path(f"uploads/{uploaded_file.name}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to handle cleanup when the app is closed
def cleanup_uploads():
    if os.path.exists(UPLOAD_FOLDER):
        # Delete all files in the 'uploads' folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Register the cleanup function to be called on exit
atexit.register(cleanup_uploads)
# ------------------ Recording Audio Section ------------------
if option == "Record Audio":
    st.header("🎙️ Record Your Audio")

    # Custom HTML/JavaScript for recording audio in the browser
    recording_html = """
    <div>
        <button id="startRecord" onclick="startRecording()">Start Recording</button>
        <button id="stopRecord" onclick="stopRecording()" disabled>Stop Recording</button>
        <audio id="audioPlayer" controls></audio>
        <a id="downloadLink" style="display: none;">Download Recording</a>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];

        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.start();

                    document.getElementById("startRecord").disabled = true;
                    document.getElementById("stopRecord").disabled = false;

                    audioChunks = [];

                    mediaRecorder.addEventListener("dataavailable", event => {
                        audioChunks.push(event.data);
                    });
                });
        }

        function stopRecording() {
            mediaRecorder.stop();

            document.getElementById("startRecord").disabled = false;
            document.getElementById("stopRecord").disabled = true;

            mediaRecorder.addEventListener("stop", () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const audioUrl = URL.createObjectURL(audioBlob);
                const audioPlayer = document.getElementById("audioPlayer");
                audioPlayer.src = audioUrl;

                const downloadLink = document.getElementById("downloadLink");
                downloadLink.href = audioUrl;
                downloadLink.download = 'recording_' + new Date().toISOString() + '.webm';
                downloadLink.style.display = 'block';
                downloadLink.textContent = 'Download Recording';
            });
        }
    </script>
    """

    # Display the HTML/JavaScript in Streamlit
    st.components.v1.html(recording_html, height=300)

    st.write("Use the buttons above to record and download your audio.")

# ------------------ Transcribe & Generate SOAP Note Section ------------------
elif option == "Transcribe & Generate SOAP Note":
    st.header("📤 Upload and Transcribe Audio")

    # Upload audio file
    uploaded_file = st.file_uploader("Upload an audio file to transcribe", type=["mp3", "wav", "ogg", "webm"])

    if uploaded_file is not None:
        # Save the uploaded file
        saved_path = save_uploaded_file(uploaded_file)
        st.success(f"File saved to: {saved_path}")

        # Transcribe the uploaded audio using Whisper model
        st.write("🔍 **Transcribing the audio...**")

        # Perform transcription
        transcription_result = speech_to_text(str(saved_path))
        transcription_text = transcription_result
        

        # Display the transcription
        st.subheader("📝 Transcription")
        st.text_area("Transcription Result", transcription_text, height=200)

        # Generate SOAP note
        st.write("🩺 **Generating SOAP Note...**")
        soap_note = generate_soap_note(transcription_text)

        # Display the SOAP note
        st.subheader("🗒️ Generated SOAP Note")
        st.text_area("SOAP Note", soap_note, height=300)

        # Provide option to download the SOAP note
        soap_note_file = "generated_soap_note.txt"
        with open(soap_note_file, "w") as f:
            f.write(soap_note)

        with open(soap_note_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="soap_note.txt">Download SOAP Note</a>'
            st.markdown(href, unsafe_allow_html=True)
          
