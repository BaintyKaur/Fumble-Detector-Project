import numpy as np
import sounddevice as sd
import threading
import queue
import time

# Optimized settings for better responsiveness
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # Reduced for faster processing
FILLER_WORDS = {"um", "uh", "erm", "ah", "aaa", "hmm", "er", "uhm", "you", "like", "so", "actually"}
pause_threshold = 1.0  # Reduced threshold

q = queue.Queue(maxsize=50)  # Prevent memory buildup

def audio_callback(indata, frames, time, status):
    if status:
        print(f"[AUDIO WARNING] {status}")
    
    try:
        q.put_nowait(indata.copy())
    except queue.Full:
        # Drop oldest if queue is full
        try:
            q.get_nowait()
            q.put_nowait(indata.copy())
        except queue.Empty:
            pass

def detect_fillers_and_repeats(words):
    fumbles = []
    tokens = [w.lower().strip('.,!?;:') for w in words if w.strip()]
    
    # Detect filler words
    for i, token in enumerate(tokens):
        if token in FILLER_WORDS:
            fumbles.append({"type": "filler", "word": token, "position": i})
    
    # Detect word repetitions
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1] and tokens[i] not in FILLER_WORDS and len(tokens[i]) > 2:
            fumbles.append({"type": "repeat", "word": tokens[i], "position": i})
    
    return fumbles

def detect_pauses_from_utterances(utterances, threshold):
    pauses = []
    for i in range(1, len(utterances)):
        gap = utterances[i]['start'] - utterances[i-1]['end']
        if gap > threshold:
            pauses.append({"after": utterances[i-1]['text'], "gap": gap})
    return pauses

def transcribe_with_deepgram(audio_chunk):
    print(f"[DEEPGRAM] Transcribing {len(audio_chunk)} samples...")
    
    try:
        from deepgram import DeepgramClient, PrerecordedOptions
        import io
        import wave

        def numpy_to_wav_bytes(audio, sample_rate):
            # Ensure audio is in correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    # Convert to int16
                    audio_int16 = (audio * 32767).astype(np.int16)
                    wf.writeframes(audio_int16.tobytes())
                return wav_io.getvalue()

        wav_data = numpy_to_wav_bytes(audio_chunk, SAMPLE_RATE)
        
        # Optimized Deepgram options
        options = PrerecordedOptions(
            model="nova-2",
            language="en-US",
            smart_format=False,
            punctuate=True,  # Enable for better word separation
            diarize=False,
            filler_words=True,  # Let Deepgram detect fillers too
            profanity_filter=False,
            redact=False,
            utterances=True,
            utt_split=0.5,  # Shorter utterance splits
            dictation=False,  # Disable for faster processing
            numerals=False,
        )
        
        # Use your API key
        api_key = "59f0c5a5b3fed54559f9218b5a85fcec60848b82"
        deepgram = DeepgramClient(api_key)
        
        # Make the API call
        response = deepgram.listen.rest.v("1").transcribe_file(
            {"buffer": wav_data, "mimetype": "audio/wav"},
            options
        )
        
        # Parse results
        results = {
            "transcript": "",
            "utterances": [],
            "alternatives": []
        }
        
        if response and response.results and response.results.channels:
            channel = response.results.channels[0]
            
            if channel.alternatives:
                # Get primary transcript
                primary_alt = channel.alternatives[0]
                results["transcript"] = primary_alt.transcript.lower().strip()
                
                # Get all alternatives
                for alt in channel.alternatives:
                    results["alternatives"].append({
                        "transcript": alt.transcript.lower().strip(),
                        "confidence": getattr(alt, 'confidence', 0.0)
                    })
            
            # Get utterances if available
            if hasattr(response.results, 'utterances') and response.results.utterances:
                for utt in response.results.utterances:
                    results["utterances"].append({
                        "text": utt.transcript.lower().strip(),
                        "start": utt.start,
                        "end": utt.end,
                        "confidence": getattr(utt, 'confidence', 0.0)
                    })
        
        print(f"[DEEPGRAM] Result: '{results['transcript']}'")
        return results
        
    except Exception as e:
        print(f"[DEEPGRAM ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"transcript": "", "utterances": [], "alternatives": []}

def start_live_recording(on_result, stop_event):
    print("[RECORDING] Starting live recording system...")
    
    total_fumbles = 0
    audio_buffer = np.array([], dtype=np.float32)
    last_process_time = time.time()
    
    # Shorter audio chunks for more responsive processing
    MIN_CHUNK_LENGTH = SAMPLE_RATE * 2  # 2 seconds minimum
    MAX_CHUNK_LENGTH = SAMPLE_RATE * 4  # 4 seconds maximum
    
    try:
        print(f"[RECORDING] Opening audio stream: SR={SAMPLE_RATE}, BS={BLOCK_SIZE}")
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE, 
            blocksize=BLOCK_SIZE,
            dtype='float32', 
            channels=1, 
            callback=audio_callback,
            latency='low'  # Try for lower latency
        ):
            print("[RECORDING] ✅ Audio stream opened successfully!")
            print("[RECORDING] 🎤 Start speaking...")
            
            while not stop_event.is_set():
                # Process audio chunks from queue
                chunks_added = 0
                while not q.empty() and chunks_added < 10:  # Limit to prevent blocking
                    try:
                        audio_chunk = q.get_nowait()
                        audio_buffer = np.concatenate((audio_buffer, audio_chunk.flatten()))
                        chunks_added += 1
                    except queue.Empty:
                        break
                
                # Check if we should process
                current_time = time.time()
                buffer_length = len(audio_buffer)
                time_since_last = current_time - last_process_time
                
                should_process = (
                    buffer_length >= MIN_CHUNK_LENGTH and 
                    (buffer_length >= MAX_CHUNK_LENGTH or time_since_last > 3.0)
                )
                
                if should_process:
                    # Get chunk to process
                    chunk_length = min(buffer_length, MAX_CHUNK_LENGTH)
                    audio_chunk = audio_buffer[:chunk_length]
                    
                    # Check audio level
                    max_amplitude = np.max(np.abs(audio_chunk))
                    print(f"[RECORDING] Processing chunk: {chunk_length} samples, max_amp: {max_amplitude:.4f}")
                    
                    if max_amplitude > 0.005:  # Lower threshold for sensitivity
                        print("[RECORDING] 🔊 Audio detected, transcribing...")
                        
                        # Transcribe
                        results = transcribe_with_deepgram(audio_chunk)
                        
                        if results["transcript"]:
                            transcript = results["transcript"]
                            print(f"[PROCESSING] 📝 Transcript: '{transcript}'")
                            
                            # Choose best transcript (prefer ones with more fillers for detection)
                            best_transcript = transcript
                            best_filler_count = sum(1 for word in FILLER_WORDS if word in transcript.split())
                            
                            for alt in results["alternatives"][1:]:  # Skip first as it's same as primary
                                alt_transcript = alt["transcript"]
                                alt_filler_count = sum(1 for word in FILLER_WORDS if word in alt_transcript.split())
                                if alt_filler_count > best_filler_count:
                                    best_transcript = alt_transcript
                                    best_filler_count = alt_filler_count
                            
                            # Detect pauses
                            pause_fumbles = 0
                            if results["utterances"]:
                                pauses = detect_pauses_from_utterances(results["utterances"], pause_threshold)
                                pause_fumbles = len(pauses)
                                if pause_fumbles > 0:
                                    print(f"[PROCESSING] 🕐 Found {pause_fumbles} pauses")
                            
                            # Detect fillers and repeats
                            words = best_transcript.split()
                            fumbles = detect_fillers_and_repeats(words)
                            word_fumbles = len(fumbles)
                            
                            # Update total
                            total_fumbles += word_fumbles + pause_fumbles
                            
                            # Get filler words for display
                            fumble_words = [f['word'] for f in fumbles if f['type'] == 'filler']
                            
                            print(f"[PROCESSING] 📊 Found {word_fumbles} word fumbles + {pause_fumbles} pauses = {word_fumbles + pause_fumbles} total")
                            print(f"[PROCESSING] 🎯 Filler words: {fumble_words}")
                            print(f"[PROCESSING] 🔢 Running total: {total_fumbles}")
                            
                            # Send to UI
                            print(f"[CALLBACK] 📤 Sending to UI...")
                            try:
                                on_result(best_transcript, total_fumbles, fumble_words)
                                print(f"[CALLBACK] ✅ Successfully sent to UI")
                            except Exception as e:
                                print(f"[CALLBACK ERROR] ❌ Failed to send to UI: {e}")
                        else:
                            print("[PROCESSING] ❌ No transcript received from Deepgram")
                    else:
                        print(f"[RECORDING] 🔇 Audio too quiet (max: {max_amplitude:.4f})")
                    
                    # Update timing and trim buffer
                    last_process_time = current_time
                    overlap_samples = SAMPLE_RATE // 4  # 0.25s overlap
                    keep_samples = min(overlap_samples, len(audio_buffer) - chunk_length + overlap_samples)
                    audio_buffer = audio_buffer[chunk_length - keep_samples:]
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
        print("[RECORDING] 🛑 Recording stopped")
        
    except Exception as e:
        print(f"[RECORDING ERROR] ❌ {e}")
        import traceback
        traceback.print_exc()

# Test function
def test_audio_input():
    print("[TEST] 🧪 Testing audio input...")
    test_queue = queue.Queue()
    
    def test_callback(indata, frames, time, status):
        if status:
            print(f"[TEST] Audio status: {status}")
        test_queue.put(indata.copy())
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, 
            blocksize=BLOCK_SIZE,
            dtype='float32', 
            channels=1, 
            callback=test_callback
        ):
            print("[TEST] 🎤 Recording test audio for 3 seconds...")
            time.sleep(3)
        
        # Analyze captured audio
        audio_data = []
        while not test_queue.empty():
            audio_data.append(test_queue.get())
        
        if audio_data:
            combined = np.concatenate([chunk.flatten() for chunk in audio_data])
            max_amp = np.max(np.abs(combined))
            rms = np.sqrt(np.mean(combined**2))
            
            print(f"[TEST] 📊 Captured {len(combined)} samples")
            print(f"[TEST] 📊 Max amplitude: {max_amp:.4f}")
            print(f"[TEST] 📊 RMS level: {rms:.4f}")
            
            if max_amp > 0.005:
                print("[TEST] ✅ Audio input is working!")
                return True
            else:
                print("[TEST] ⚠️ Audio level too low - speak louder or check microphone")
                return False
        else:
            print("[TEST] ❌ No audio data captured")
            return False
            
    except Exception as e:
        print(f"[TEST ERROR] ❌ Audio test failed: {e}")
        return False