import click
from rich.console import Console
from .core import LLMInterface

console = Console()


@click.command()
@click.option('-m', '--model', required=True, help='Model identifier (e.g., meta-llama/Llama-3.2B-3B)')
@click.option('-c', '--checkpoint', default=None, help='Path to custom model checkpoint')
@click.option('-i', '--interactive', is_flag=True, help='Start interactive chat session')
@click.option('-s', '--system-prompt', default="You are a helpful assistant",
              help='System prompt to use for the chat')
@click.argument('prompt', required=False)
def main(model: str, checkpoint: str, interactive: bool, system_prompt: str, prompt: str):
    try:
        # Initialize model
        llm = LLMInterface(model, checkpoint)
        llm.system_prompt = system_prompt

        if interactive:
            llm.interactive_session()
        elif prompt:
            llm.generate_response(prompt)  # Response is streamed, no need to print return value
        else:
            console.print("[red]Error: Please provide either a prompt or use -i for interactive mode[/red]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise e


if __name__ == '__main__':
    main()