import os
import threading
import base64
import json
import time
import logging

import gradio as gr
import numpy as np
from pydub import AudioSegment
from websocket import WebSocketApp
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Load environment variables
API_KEY = os.environ["API_KEY"]
WS_URL = 'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview'

def start_recording_user():
    return gr.Audio(label="Input Audio", sources=["microphone"], type="numpy", recording=True)


def stop_conv():
    return gr.Audio(label="Input Audio", sources=["microphone"], type="numpy", recording=False)


class GradioClient:
    def __init__(self):
        self.audio_buffer = b""
        self.ws_app = None
        self.user_speech_stopped = False
        self.model_speech_stopped = False

    def connect_to_openai(self):
        self.ws_app = WebSocketApp(
            WS_URL,
            header=[f'Authorization: Bearer {API_KEY}', 'OpenAI-Beta: realtime=v1'],
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        def run_ws():
            self.ws_app.run_forever()

        ws_thread = threading.Thread(target=run_ws)
        ws_thread.daemon = True
        ws_thread.start()

        # Wait for connection to open
        time.sleep(1)

        #Send initial message
        self.ws_app.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "Your knowledge cutoff is 2023-10. You are a helpful assistant.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "temperature": 0.8,
                "max_response_output_tokens": "inf"
            }
        }))

    def on_error(self, ws_app, error):
        logging.info(f'Error: {error}')

    def on_close(self, ws_app, code, error):
        logging.info(f'Closing code: {code}, error:{error}')

    def on_message(self, ws_app, message):
        try:
            message = json.loads(message)
            event_type = message['type']
            logging.info(f"Type: {event_type}, Message: {message}")
            if event_type == 'response.audio.delta':
                audio_content = base64.b64decode(message['delta'])
                self.audio_buffer+=audio_content
            elif event_type == 'input_audio_buffer.speech_started':
                self.model_speech_stopped = False
            elif event_type == 'input_audio_buffer.speech_stopped':
                self.user_speech_stopped = True
            elif event_type == 'response.audio.done' or  event_type == 'response.text.done':
                self.model_speech_stopped = True
        except Exception as e:
            logging.info(f'Error processing message: {e}')

    def send_audio(self, audio_data:bytes):
        encoded_chunk = base64.b64encode(audio_data).decode()
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "audio": encoded_chunk
                }]
            }
        }
        self.ws_app.send(json.dumps(event))

    def process_input(self, audio: tuple):
        segment = AudioSegment(
            audio[1].tobytes(),
            frame_rate=audio[0],
            sample_width=audio[1].dtype.itemsize,
            channels=1,
        )
        pcm_audio = segment.set_frame_rate(24000).set_channels(1).set_sample_width(2).raw_data

        event = {
            "type": "input_audio_buffer.append",
            'audio': base64.b64encode(pcm_audio).decode()
        }
        self.ws_app.send(json.dumps(event))

        if self.user_speech_stopped:
            return gr.Audio(recording=False)
        return None

    def process_output(self):
        while not self.model_speech_stopped:
            time.sleep(0.5)
        audio = np.frombuffer(self.audio_buffer, dtype=np.int16)
        self.audio_buffer = b""
        return 24000, audio

    def audio_finished(self):
        self.user_speech_stopped = False
        self.model_speech_stopped = False
        return gr.Audio(label="Input Audio", sources=["microphone"], type="numpy", recording=True)


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(
                    label="Input Audio", sources=["microphone"], type="numpy"
                )
            with gr.Column():
                output_audio = gr.Audio(label="Output Audio", streaming=True, autoplay=True)
        client = GradioClient()
        stream = input_audio.stream(
            client.process_input,
            [input_audio],
            [input_audio],
            stream_every=1,
            time_limit=30,
        )

        input_audio.stop_recording(
            client.process_output,
            [],
            [output_audio]
        )

        output_audio.stop(
            client.audio_finished,
            [],
            [input_audio]
        )

        cancel = gr.Button("Stop Conversation", variant="stop")
        cancel.click(stop_conv, [], [input_audio])
    client.connect_to_openai()
    logging.info(f"Launching")
    demo.launch()