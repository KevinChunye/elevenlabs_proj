let ws = null;
let audioCtx = null;
let processor = null;
let source = null;
let stream = null;
let isRunning = false;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');
const partialEl = document.getElementById('partial');
const finalEl = document.getElementById('final');

function setStatus(msg) {
  statusEl.textContent = `status: ${msg}`;
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buffer);
}

function toBase64(uint8) {
  let binary = '';
  for (let i = 0; i < uint8.byteLength; i++) {
    binary += String.fromCharCode(uint8[i]);
  }
  return btoa(binary);
}

async function getRealtimeToken() {
  const res = await fetch('/api/realtime-token', { method: 'POST' });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`token error: ${txt}`);
  }
  return res.json();
}

async function start() {
  if (isRunning) return;
  isRunning = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus('requesting token');

  try {
    const tokenPayload = await getRealtimeToken();
    const token = tokenPayload.token || tokenPayload.data?.token;
    if (!token) throw new Error('no token in response');

    const wsUrl = `wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime&token=${encodeURIComponent(token)}&language_code=en`;
    ws = new WebSocket(wsUrl);

    ws.onopen = async () => {
      setStatus('websocket open, requesting mic');
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      source = audioCtx.createMediaStreamSource(stream);

      // ScriptProcessor is widely supported for MVP; swap to AudioWorklet for production.
      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      source.connect(processor);
      processor.connect(audioCtx.destination);

      processor.onaudioprocess = (event) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const pcm = floatTo16BitPCM(event.inputBuffer.getChannelData(0));
        ws.send(JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: toBase64(pcm),
        }));
      };

      setStatus('streaming');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        const partial = msg.partial_transcript || msg.text || msg.transcript || '';
        const finalText = msg.final_transcript || msg.aligned_text || '';
        if (partial) partialEl.textContent = partial;
        if (finalText) finalEl.textContent += `${finalText}\n`;
      } catch (e) {
        // Keep raw output visible for API compatibility differences.
        partialEl.textContent = String(event.data);
      }
    };

    ws.onerror = (e) => {
      setStatus('websocket error');
      console.error(e);
    };

    ws.onclose = () => {
      if (isRunning) stop();
      setStatus('closed');
    };

    // Force periodic commit/flush frames.
    const commitInterval = setInterval(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN || !isRunning) {
        clearInterval(commitInterval);
        return;
      }
      ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
    }, 900);
  } catch (err) {
    console.error(err);
    setStatus(`failed: ${err.message}`);
    stop();
  }
}

function stop() {
  isRunning = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;

  if (processor) {
    processor.disconnect();
    processor.onaudioprocess = null;
    processor = null;
  }
  if (source) {
    source.disconnect();
    source = null;
  }
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
    ws.send(JSON.stringify({ type: 'session.close' }));
    ws.close();
  }
  ws = null;
  setStatus('stopped');
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
