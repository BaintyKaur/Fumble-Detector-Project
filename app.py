import streamlit as st
import threading
import queue
import time
import numpy as np
from collections import deque
from live import start_live_recording, detect_fillers_and_repeats
import io
import tempfile
import os

# --- Global queue for thread communication ---
results_queue = queue.Queue()

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="🎤 Fumble Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Eye-Catching UI ---
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);}
    .stButton>button {background-color: #6366f1; color: white; font-weight: bold;}
    .stButton>button:hover {background-color: #818cf8;}
    .metric-label {color: #6366f1;}
    .fumble {color: #ef4444; font-weight: bold;}
    .filler {background: #fef3c7; color: #b45309; border-radius: 6px; padding: 2px 6px;}
    .debug {background: #f0f0f0; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto;}
    .status-recording {color: #22c55e; font-weight: bold;}
    .status-stopped {color: #ef4444; font-weight: bold;}
    .upload-success {background: #dcfce7; padding: 15px; border-radius: 10px; border-left: 4px solid #22c55e;}
    .upload-error {background: #fef2f2; padding: 15px; border-radius: 10px; border-left: 4px solid #ef4444;}
    </style>
""", unsafe_allow_html=True)

# --- File Processing Functions ---
def process_audio_file(uploaded_file):
    """Process uploaded audio file and extract fumbles"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        st.info(f"Processing {uploaded_file.name}... This may take a moment.")
        
        # Convert to numpy array based on file type
        audio_data, sample_rate = load_audio_file(temp_path)
        
        if audio_data is None:
            return None
            
        # Process with Deepgram
        results = transcribe_file_with_deepgram(audio_data, sample_rate)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return results
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def load_audio_file(file_path):
    """Load audio file and convert to numpy array"""
    try:
        file_ext = file_path.split('.')[-1].lower()
        
        if file_ext == 'wav':
            import wave
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                # Convert to numpy array
                if sample_width == 1:
                    audio_data = np.frombuffer(frames, dtype=np.uint8)
                    audio_data = (audio_data.astype(np.float32) - 128) / 128
                elif sample_width == 2:
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768
                else:
                    audio_data = np.frombuffer(frames, dtype=np.int32)
                    audio_data = audio_data.astype(np.float32) / 2147483648
                
                # Convert to mono if stereo
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                return audio_data, sample_rate
                
        elif file_ext in ['mp3', 'm4a', 'mp4']:
            try:
                # Try using pydub
                from pydub import AudioSegment
                
                if file_ext == 'mp3':
                    audio = AudioSegment.from_mp3(file_path)
                else:
                    audio = AudioSegment.from_file(file_path, format='mp4')
                
                # Convert to mono and get raw data
                audio = audio.set_channels(1)
                sample_rate = audio.frame_rate
                
                # Convert to numpy array
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                audio_data = audio_data / (2**(audio.sample_width * 8 - 1))
                
                return audio_data, sample_rate
                
            except ImportError:
                st.error("pydub is required for MP3/M4A files. Please install it with: pip install pydub")
                return None, None
        else:
            st.error(f"Unsupported file format: {file_ext}")
            return None, None
            
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None, None

def transcribe_file_with_deepgram(audio_data, original_sample_rate):
    """Transcribe audio file using Deepgram"""
    try:
        from deepgram import DeepgramClient, PrerecordedOptions
        import wave
        
        # Resample to 16kHz if needed
        if original_sample_rate != 16000:
            # Simple resampling (you might want to use scipy.signal.resample for better quality)
            target_length = int(len(audio_data) * 16000 / original_sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), target_length),
                np.arange(len(audio_data)),
                audio_data
            )
        
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        # Convert to WAV bytes
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            wav_data = wav_io.getvalue()
        
        # Deepgram options
        options = PrerecordedOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            punctuate=True,
            diarize=False,
            filler_words=True,
            profanity_filter=False,
            redact=False,
            utterances=True,
            utt_split=0.8,
            dictation=False,
            numerals=False,
        )
        
        # API call
        api_key = "59f0c5a5b3fed54559f9218b5a85fcec60848b82"
        deepgram = DeepgramClient(api_key)
        
        response = deepgram.listen.rest.v("1").transcribe_file(
            {"buffer": wav_data, "mimetype": "audio/wav"},
            options
        )
        
        # Parse results
        results = {
            "transcript": "",
            "utterances": [],
            "alternatives": [],
            "audio_duration": len(audio_data) / 16000
        }
        
        if response and response.results and response.results.channels:
            channel = response.results.channels[0]
            
            if channel.alternatives:
                results["transcript"] = channel.alternatives[0].transcript.lower()
                
                for alt in channel.alternatives:
                    results["alternatives"].append({
                        "transcript": alt.transcript.lower(),
                        "confidence": getattr(alt, 'confidence', 0.0)
                    })
            
            if hasattr(response.results, 'utterances') and response.results.utterances:
                for utt in response.results.utterances:
                    results["utterances"].append({
                        "text": utt.transcript.lower(),
                        "start": utt.start,
                        "end": utt.end,
                        "confidence": getattr(utt, 'confidence', 0.0)
                    })
        
        return results
        
    except Exception as e:
        st.error(f"Deepgram transcription failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# --- State Management ---
if "recording" not in st.session_state:
    st.session_state.recording = False
if "fumble_count" not in st.session_state:
    st.session_state.fumble_count = 0
if "fumble_words" not in st.session_state:
    st.session_state.fumble_words = deque(maxlen=50)
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "debug_messages" not in st.session_state:
    st.session_state.debug_messages = deque(maxlen=20)
if "last_update" not in st.session_state:
    st.session_state.last_update = 0
if "update_counter" not in st.session_state:
    st.session_state.update_counter = 0

# --- Helper Functions ---
def add_debug(message):
    timestamp = time.strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    st.session_state.debug_messages.append(full_msg)
    print(full_msg)

def on_result(transcript, fumble_count, fumble_words):
    add_debug(f"CALLBACK: Got result - transcript='{transcript}', fumbles={fumble_count}, words={fumble_words}")
    results_queue.put({
        'transcript': transcript,
        'fumble_count': fumble_count,
        'fumble_words': fumble_words,
        'timestamp': time.time()
    })

# --- Main UI ---
st.markdown("<h1 style='text-align:center; color:#6366f1;'>🎤 Fumble Detector</h1>", unsafe_allow_html=True)

# --- Tabs for different modes ---
tab1, tab2, tab3 = st.tabs(["📁 File Upload", "🎙️ Live Recording", "🔧 Debug"])

# --- FILE UPLOAD TAB ---
with tab1:
    st.markdown("## 📁 Upload Audio File for Analysis")
    st.markdown("Upload a WAV, MP3, or M4A file to test fumble detection")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a'],
        help="Supported formats: WAV, MP3, M4A"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        process_file = st.button("🔍 Analyze File", disabled=uploaded_file is None, type="primary")
    
    with col2:
        if uploaded_file:
            st.info(f"File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
    
    if process_file and uploaded_file:
        with st.spinner("Processing audio file..."):
            results = process_audio_file(uploaded_file)
            
            if results and results["transcript"]:
                st.markdown('<div class="upload-success">', unsafe_allow_html=True)
                st.success("✅ File processed successfully!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                # Analyze transcript
                words = results["transcript"].split()
                fumbles = detect_fillers_and_repeats(words)
                filler_fumbles = [f for f in fumbles if f['type'] == 'filler']
                repeat_fumbles = [f for f in fumbles if f['type'] == 'repeat']
                
                # Analyze pauses
                pause_fumbles = []
                if results["utterances"]:
                    from live import detect_pauses_from_utterances
                    pause_fumbles = detect_pauses_from_utterances(results["utterances"], 1.0)
                
                total_fumbles = len(fumbles) + len(pause_fumbles)
                
                with col1:
                    st.metric("Total Fumbles", total_fumbles)
                    st.metric("Audio Duration", f"{results.get('audio_duration', 0):.1f}s")
                
                with col2:
                    st.metric("Filler Words", len(filler_fumbles))
                    st.metric("Repetitions", len(repeat_fumbles))
                
                with col3:
                    st.metric("Long Pauses", len(pause_fumbles))
                    if total_fumbles > 0 and results.get('audio_duration', 0) > 0:
                        fumble_rate = total_fumbles / (results['audio_duration'] / 60)
                        st.metric("Fumbles/Min", f"{fumble_rate:.1f}")
                
                # Show transcript with highlights
                st.markdown("### 📝 Transcript")
                highlighted_transcript = results["transcript"]
                
                # Highlight fillers
                filler_words = [f['word'] for f in filler_fumbles]
                for word in set(filler_words):
                    highlighted_transcript = highlighted_transcript.replace(
                        word, f"<span class='filler'>{word}</span>"
                    )
                
                st.markdown(highlighted_transcript, unsafe_allow_html=True)
                
                # Show detected fumbles
                if filler_fumbles:
                    st.markdown("### 🎯 Filler Words Found")
                    filler_counts = {}
                    for f in filler_fumbles:
                        word = f['word']
                        filler_counts[word] = filler_counts.get(word, 0) + 1
                    
                    filler_html = " ".join([
                        f"<span class='filler'>{word} ({count})</span>"
                        for word, count in filler_counts.items()
                    ])
                    st.markdown(filler_html, unsafe_allow_html=True)
                
                if repeat_fumbles:
                    st.markdown("### 🔄 Repeated Words")
                    repeat_words = [f['word'] for f in repeat_fumbles]
                    st.write(", ".join(set(repeat_words)))
                
                if pause_fumbles:
                    st.markdown("### ⏸️ Long Pauses")
                    for pause in pause_fumbles:
                        st.write(f"• {pause['gap']:.1f}s pause after: \"{pause['after']}\"")
                
                # Show alternatives if available
                if len(results["alternatives"]) > 1:
                    with st.expander("🔄 Alternative Transcriptions"):
                        for i, alt in enumerate(results["alternatives"][:3]):
                            st.write(f"**Alternative {i+1}** (confidence: {alt['confidence']:.2f})")
                            st.write(alt["transcript"])
            else:
                st.markdown('<div class="upload-error">', unsafe_allow_html=True)
                st.error("❌ Failed to process file or no speech detected")
                st.markdown('</div>', unsafe_allow_html=True)

# --- LIVE RECORDING TAB ---
with tab2:
    st.markdown("## 🎙️ Live Recording")
    
    # Status indicator
    if st.session_state.recording:
        st.markdown("<p class='status-recording'>🔴 RECORDING</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='status-stopped'>⏹️ STOPPED</p>", unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("🟢 Start Live", use_container_width=True, type="primary", disabled=st.session_state.recording):
            add_debug("START: Button clicked")
            st.session_state.recording = True
            st.session_state.stop_event.clear()
            st.session_state.fumble_count = 0
            st.session_state.fumble_words.clear()
            st.session_state.transcript = ""
            
            def recording_thread():
                add_debug("THREAD: Recording thread started")
                try:
                    start_live_recording(on_result, st.session_state.stop_event)
                    add_debug("THREAD: Recording thread completed normally")
                except Exception as e:
                    add_debug(f"THREAD: Recording thread error: {str(e)}")
            
            threading.Thread(target=recording_thread, daemon=True).start()
            st.rerun()

    with col2:
        if st.button("🔴 Stop", use_container_width=True, disabled=not st.session_state.recording):
            st.session_state.stop_event.set()
            st.session_state.recording = False
            st.rerun()

    with col3:
        if st.button("🧹 Reset", use_container_width=True):
            st.session_state.fumble_count = 0
            st.session_state.fumble_words.clear()
            st.session_state.transcript = ""
            st.session_state.debug_messages.clear()
            st.session_state.update_counter = 0
            while not results_queue.empty():
                try:
                    results_queue.get_nowait()
                except queue.Empty:
                    break
            st.rerun()

    # Process results queue
    while not results_queue.empty():
        try:
            result = results_queue.get_nowait()
            st.session_state.transcript = result['transcript']
            st.session_state.fumble_count = result['fumble_count']
            st.session_state.fumble_words.extend(result['fumble_words'])
            st.session_state.last_update = result['timestamp']
            st.session_state.update_counter += 1
        except queue.Empty:
            break

    # Live stats
    st.markdown("### 📊 Live Stats")
    stats1, stats2, stats3, stats4 = st.columns(4)
    with stats1:
        st.metric("Total Fumbles", st.session_state.fumble_count)
    with stats2:
        st.metric("Unique Fillers", len(set(st.session_state.fumble_words)))
    with stats3:
        st.metric("Queue Size", results_queue.qsize())
    with stats4:
        st.metric("Updates", st.session_state.update_counter)

    # Transcript display
    st.markdown("### 📝 Live Transcript")
    if st.session_state.transcript:
        highlighted = st.session_state.transcript
        for fw in set(st.session_state.fumble_words):
            if fw:
                highlighted = highlighted.replace(fw, f"<span class='filler'>{fw}</span>")
        st.markdown(f"**Latest:** {highlighted}", unsafe_allow_html=True)
    else:
        if st.session_state.recording:
            st.markdown("🎤 **Listening...** Speak clearly into your microphone")
        else:
            st.markdown("⏹️ **Not recording** - Click Start to begin")

    # Auto-refresh while recording
    if st.session_state.recording:
        time.sleep(0.5)
        st.rerun()

# --- DEBUG TAB ---
with tab3:
    st.markdown("## 🔧 Debug Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### System Status")
        st.write(f"Recording: {st.session_state.recording}")
        st.write(f"Queue Size: {results_queue.qsize()}")
        st.write(f"Updates: {st.session_state.update_counter}")
        
        if st.button("🔊 Test Audio"):
            from live import test_audio_input
            with st.spinner("Testing audio..."):
                if test_audio_input():
                    st.success("✅ Audio working!")
                else:
                    st.error("❌ Audio issue detected")
    
    with col2:
        st.markdown("### Manual Test")
        manual_text = st.text_area("Test text:", "um hello uh this is a test")
        if manual_text:
            fumbles = detect_fillers_and_repeats(manual_text.split())
            fillers = [f['word'] for f in fumbles if f['type'] == 'filler']
            st.write(f"Fumbles: {len(fumbles)}")
            st.write(f"Fillers: {fillers}")
    
    st.markdown("### Debug Log")
    if st.session_state.debug_messages:
        debug_text = "\n".join(reversed(list(st.session_state.debug_messages)))
        st.markdown(f"<div class='debug'>{debug_text}</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6366f1;'>"
    "✨ Powered by <b>Deepgram</b> & <b>Streamlit</b>"
    "</div>", unsafe_allow_html=True
)