import base64
import pyautogui
import numpy as np
from threading import Lock, Thread
import cv2
import openai
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()


class ScreenCapture:
    def __init__(self):
        self.frame = None
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()


class ScreenAssistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a screen observation assistant that helps users understand what's 
        on their laptop screen. You will receive screenshots and questions about 
        the screen content. Your job is to:
        
        1. Describe visual elements when asked
        2. Explain text content when requested
        3. Identify applications/windows visible
        4. Provide summaries of visible information
        5. Answer questions about what's displayed
        
        Be concise but thorough in your descriptions. Focus on the most relevant 
        elements based on the user's question. If text is small and unreadable, 
        say so rather than guessing.
        
        For technical content (code, documents, etc.), provide accurate explanations.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


# Initialize screen capture
screen_capture = ScreenCapture().start()

# Use Gemini for better image understanding
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")

# Alternatively, use GPT-4 with vision capabilities
# model = ChatOpenAI(model="gpt-4-vision-preview")

assistant = ScreenAssistant(model)


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, screen_capture.read(encode=True))

    except UnknownValueError:
        print("Could not understand audio. Please try again.")


recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

print("Screen observation assistant is running. Ask questions about your screen.")
print("Press Ctrl+C to quit.")

try:
    # Show screen preview (optional)
    while True:
        cv2.imshow("Screen Preview", screen_capture.read())
        if cv2.waitKey(1) in [27, ord("q")]:
            break
except KeyboardInterrupt:
    pass

screen_capture.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)