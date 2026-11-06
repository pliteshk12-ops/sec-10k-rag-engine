from rich.console import Console
console = Console()
def log_step(msg: str):
    console.print(f'[bold green][+] {msg}[/bold green]')

