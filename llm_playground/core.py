import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from rich.console import Console
from rich.prompt import Prompt
from typing import Optional

from transformers.generation.streamers import BaseStreamer

console = Console()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt: bool = True, **kwargs):
        super().__init__(tokenizer, **kwargs)
        # self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.prompt_printed = False
        self.collected = ""
        self.idx = 0

    def put(self, tokens):
        # print(f"{self.idx} Got value: {tokens}")
        # self.idx += 1

        # print(tokens.shape)

        if len(tokens.shape) > 1:
            return


        text = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        self.collected += text


        console.print(text, end="")


class LLMInterface:
    def __init__(self, model_name: str | None, checkpoint_path: str | None):
        self.device = get_device()
        console.print(f"[bold green]Using device: {self.device}[/bold green]")


        if model_name and checkpoint_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path, local_files_only=True)
        elif model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif checkpoint_path:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path, local_files_only=True)

        self.model.eval()
        self.model = self.model.to(self.device)

        # Default system prompt
        self.system_prompt = "You are a helpful assistant"

        # Initialize streamer
        self.streamer = CustomStreamer(self.tokenizer)

    def generate_response(self, user_message: str) -> None:
        # Prepare chat messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Apply chat template and get inputs
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        ).to(self.device)

        # Generate response with streaming
        out = self.model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=self.streamer
        )
        console.print()  # Add newline after response

        # print(f"{self.tokenizer.decode(out[0])=}")

    def interactive_session(self):
        console.print("[bold green]Starting interactive session. Type 'exit' to quit.[/bold green]")
        chat_history = []

        while True:
            user_input = Prompt.ask("\nYou")
            if user_input.lower() == 'exit':
                break

            # In interactive mode, we'll maintain the full chat history
            messages = [{"role": "system", "content": self.system_prompt}]

            # Add all previous exchanges
            for i in range(0, len(chat_history), 2):
                messages.extend([
                    {"role": "user", "content": chat_history[i]},
                    {"role": "assistant", "content": chat_history[i + 1]}
                ])

            # Add current user message
            messages.append({"role": "user", "content": user_input})

            # Generate response with full context
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True
            ).to(self.device)

            console.print("\n[bold blue]Assistant[/bold blue]:", end=" ")
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=self.streamer
            )
            console.print()  # Add newline after response

            # Get full response for chat history
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the assistant's response for history
            response = response.split("assistant")[-1].split(":", 1)[1].strip()
            chat_history.extend([user_input, response])