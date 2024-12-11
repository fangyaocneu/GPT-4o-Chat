# Real-Time Speech-to-Speech Conversation with GPT-4o

This README provides an overview of a Python application that leverages the GPT-4 language model from OpenAI to perform real-time speech-to-speech conversation with GPT-4o. The application uses the Gradio library for the user interface and the PyDub library for audio processing.

## Features

- Real-time speech input from the microphone
- Speech-to-text transcription using OpenAI's Whisper ASR model
- Text translation using GPT-4
- Text-to-speech synthesis using OpenAI's voice synthesis model
- Streaming audio output for a seamless user experience

## Prerequisites

Before running the application, ensure that you have the following:

- Python 3.7 or later
- OpenAI API key (stored in the `API_KEY` environment variable)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/fangyaocneu/GPT-4o-Chat.git
```

2. Change to the project directory:

```bash
cd GPT-4o-Chat
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Set the `API_KEY` environment variable with your OpenAI API key:

```bash
export API_KEY=your_openai_api_key
```

2. Run the application:

```bash
python demo.py
```

3. The Gradio interface will open in your default web browser. Click the "Start Recording" button to begin speaking into your microphone.

4. The application will transcribe your speech, translate it using GPT-4, and synthesize the translated speech output, which will be played back to you in real-time.

5. To stop the conversation, click the "Stop Conversation" button.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [OpenAI](https://openai.com/) for providing the GPT-4 language model and other AI services
- [Gradio](https://github.com/gradio-app/gradio) for the easy-to-use UI library
- [PyDub](https://github.com/jiaaro/pydub) for audio processing capabilities