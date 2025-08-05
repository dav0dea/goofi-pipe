import json
from os import environ, path

import requests

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import FloatParam, IntParam, StringParam


class TextGeneration(Node):
    """
    This node performs text generation using various large language models (LLMs) from different providers, including OpenAI GPT, Anthropic Claude, Google Gemini, Ollama, and optionally a local model endpoint. It sends an input prompt to the selected LLM and returns the generated text response. Conversation history and system prompts can be used depending on the model type, and generated conversations can be saved to a file if enabled.

    Inputs:
    - prompt: The input text prompt, typically a user message or instruction to be completed or responded to by the language model.

    Outputs:
    - generated_text: The generated response text produced by the selected language model, based on the input prompt and optional conversation history.
    """

    @staticmethod
    def config_input_slots():
        return {"prompt": DataType.STRING}

    @staticmethod
    def config_output_slots():
        return {"generated_text": DataType.STRING}

    @staticmethod
    def config_params():
        return {
            "text_generation": {
                "api_key": StringParam("openai.key", doc="API key for the text generation service"),
                "model": StringParam("gpt-4o-mini", doc="Model ID for the text generation service"),
                "system_prompt": StringParam("", doc="System prompt for text generation"),
                "temperature": FloatParam(1.0, 0.0, 2.0, doc="Temperature for text generation"),
                "max_tokens": IntParam(20, 5, 2048, doc="Maximum number of tokens to generate"),
                "max_conversation_length": IntParam(
                    0, 0, 100, doc="Maximum length of conversation history (0 to disable, -1 for unlimited)"
                ),
                "save_conversation": StringParam("", doc="Whether to save conversation history to a JSON file"),
            }
        }

    def setup(self):
        self.provider = None
        self.client = None
        self.messages = []
        self.text_generation_api_key_changed(self.params.text_generation.api_key.value)
        self.text_generation_model_changed(self.params.text_generation.model.value)

    def generate_openai_response(self, messages):
        if self.client is None:
            self.client = self.provider.OpenAI(api_key=self.api_key)

        payload = {
            "model": self.params.text_generation.model.value,
            "messages": messages,
            "temperature": self.params.text_generation.temperature.value,
            "max_tokens": self.params.text_generation.max_tokens.value,
        }

        if len(self.params.text_generation.system_prompt.value) > 0 and len(messages) < 2:
            payload["messages"] = [{"role": "system", "content": self.params.text_generation.system_prompt.value}] + messages

        response = self.client.chat.completions.create(**payload)
        return response.choices[0].message.content

    def generate_anthropic_response(self, messages):
        if self.client is None:
            self.client = self.provider.Anthropic(api_key=self.api_key)

        system_prompt = self.params.text_generation.system_prompt.value
        response = self.client.messages.create(
            model=self.params.text_generation.model.value,
            messages=messages,
            max_tokens=self.params.text_generation.max_tokens.value,
            temperature=self.params.text_generation.temperature.value,
            system=system_prompt if len(system_prompt) > 0 else None,
        )
        return response.content[0].text

    def generate_gemini_response(self, prompt):
        if self.client is None:
            self.provider.configure(api_key=self.api_key)
            system_prompt = self.params.text_generation.system_prompt.value
            self.client = self.provider.GenerativeModel(
                model_name=self.params.text_generation.model.value,
                system_instruction=system_prompt if len(system_prompt) > 0 else "You are a helpful assistant.",
            )

        chat = self.client.start_chat()
        generation_config = self.provider.GenerationConfig(
            max_output_tokens=self.params.text_generation.max_tokens.value,
            temperature=self.params.text_generation.temperature.value,
        )

        # send the request
        response = chat.send_message(prompt, generation_config=generation_config)
        return response.text

    def generate_ollama_response(self, model, messages):
        # create an Ollama client instance
        if self.client is None:
            self.client = self.provider

        response = self.client.chat(
            model=str(model.replace("ollama-", "")),  # remove ollama- prefix
            messages=messages,
            options={
                "system": self.params.text_generation.system_prompt.value,
                "temperature": self.params.text_generation.temperature.value,
                "max_tokens": self.params.text_generation.max_tokens.value,
            },
        )
        return response["message"]["content"]

    def generate_local_response(self, prompt):
        url = "http://127.0.0.1:5000/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "mode": "chat-instruct",
            "max_tokens": self.params.text_generation.max_tokens.value,
            "temperature": self.params.text_generation.temperature.value,
            "character": "Example",
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(url, headers=headers, json=data, verify=False)
        return response.json()["choices"][0]["message"]["content"]

    def save_conversation_to_json(self):
        filename = f"conversation_{self.params.text_generation.save_conversation.value}.json"
        with open(filename, "w") as f:
            json.dump(self.messages, f, indent=4)
        print(f"Conversation saved to {filename}")

    def process(self, prompt: Data):
        if prompt is None or prompt.data is None:
            return None

        model = self.params.text_generation.model.value
        max_conversation_length = self.params.text_generation.max_conversation_length.value
        save_conversation = self.params.text_generation.save_conversation.value

        if max_conversation_length == 0:
            self.messages = [{"role": "user", "content": prompt.data}]
        else:
            if max_conversation_length > 0 and len(self.messages) >= max_conversation_length:
                self.messages = self.messages[-max_conversation_length + 1 :]
            self.messages.append({"role": "user", "content": prompt.data})

        if model.startswith("gpt-"):
            generated_text = self.generate_openai_response(self.messages)

        elif model.startswith("claude-"):
            generated_text = self.generate_anthropic_response(self.messages)

        elif model.startswith("gemini-"):
            if max_conversation_length != 0:
                print("The API call for Gemini currently does not support conversation history. Open an issue if needed.")
            generated_text = self.generate_gemini_response(prompt.data)

        elif model.startswith("local-"):
            if max_conversation_length != 0:
                print("The local model currently does not support conversation history. Open an issue if needed.")
            generated_text = self.generate_local_response(prompt.data)

        elif model.startswith("ollama-"):
            generated_text = self.generate_ollama_response(model, self.messages)
        else:
            raise ValueError(f"Unknown model: {model}")

        if max_conversation_length != 0:
            self.messages.append({"role": "assistant", "content": generated_text})

        if save_conversation:
            self.save_conversation_to_json()

        return {"generated_text": (generated_text, prompt.meta)}

    def text_generation_api_key_changed(self, value):
        self.client = None
        self.api_key = value

        model = self.params.text_generation.model.value
        if path.exists(self.api_key):
            with open(self.api_key, "r") as f:
                self.api_key = f.read().strip()
        elif model.startswith("gpt-"):
            if len(environ.get("OPENAI_API_KEY", "")) > 0:
                print("Using OPENAI_API_KEY from environment variables.")
            self.api_key = environ.get("OPENAI_API_KEY", self.api_key)
        elif model.startswith("claude-"):
            if len(environ.get("ANTHROPIC_API_KEY", "")) > 0:
                print("Using ANTHROPIC_API_KEY from environment variables.")
            self.api_key = environ.get("ANTHROPIC_API_KEY", self.api_key)
        elif model.startswith("gemini-"):
            if len(environ.get("GOOGLE_API_KEY", "")) > 0:
                print("Using GOOGLE_API_KEY from environment variables.")
            self.api_key = environ.get("GOOGLE_API_KEY", self.api_key)

        if not self.api_key and not model.startswith("local-"):
            raise ValueError(f"API key for {model} not found in environment variables or input parameters.")
        print(f"Loaded API key for {model}.")

    def text_generation_model_changed(self, model):
        self.provider = None
        self.client = None

        if model.startswith("gpt-"):
            try:
                import openai
            except ImportError:
                raise ImportError("You need to install openai: pip install openai")
            self.provider = openai

        elif model.startswith("claude-"):
            try:
                import anthropic
            except ImportError:
                raise ImportError("You need to install anthropic: pip install anthropic")
            self.provider = anthropic

        elif model.startswith("gemini-"):
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("You need to install google-generativeai: pip install google-generativeai\nWith Python>=3.10")
            self.provider = genai

        elif model.startswith("ollama-"):
            try:
                import ollama
            except ImportError:
                raise ImportError("You need to install ollama: pip install ollama")
            self.provider = ollama

        else:
            raise ValueError(f"Unknown model: {model}")
