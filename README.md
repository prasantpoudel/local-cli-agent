# 🤖 AI Agent

A sophisticated, terminal-based AI agentic framework designed for autonomous task execution with a premium text user interface. Built with Python 3.14+, this agent leverages LLMs to interact with the system through a structured tool registry and real-time event streaming.

## ✨ Features

- **🚀 Autonomous Execution**: Leverages an agentic loop to process complex prompts and determine necessary actions.
- **🛠️ Extensible Tool Registry**: Built-in support for system tools (like `read_file`) with an easy-to-extend architecture for custom tools.
- **🎨 Premium TUI**: Rich, interactive terminal interface featuring:
    - Real-time response streaming.
    - Beautifully formatted tool execution panels.
    - Syntax-highlighted code output.
    - Status indicators and progress tracking.
- **📡 Event-Driven Architecture**: Uses a robust event system (`AgentEventType`) to decouple the core logic from the user interface.
- **🧠 Context Management**: Sophisticated chat history and context handling for long-running interactions.

## 🛠️ Tech Stack

- **Luguage**: Python 3.14+
- **CLI Framework**: [Click](https://click.palletsprojects.com/)
- **UI Rendering**: [Rich](https://rich.readthedocs.io/)
- **LLM Integration**: OpenAI API (compatible with local providers like Ollama)
- **Dependency Management**: Poetry / PDM

## 🚀 Getting Started

### Prerequisites

- Python 3.14 or higher
- An OpenAI API Key (or equivalent compatible endpoint)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # OR if using poetry
   poetry install
   ```

3. **Configure environment**:
   Create a `.env` file in the root directory and add your credentials:
   ```env
   OPEN_ROUTER_API_KEY=your_api_key_here
   # Optional: BASE_URL for local LLMs
   ```

### Usage

Run the agent with a prompt directly from the terminal:

```bash
python main.py "can you analyze the file structure of this project and summarize what main.py does?"
```

## 🏗️ Project Structure

- `agent/`: Core agentic loop and event definitions.
- `client/`: LLM client implementation and response parsing.
- `tools/`: Registry and implementation of available tools.
- `ui/`: TUI components and terminal formatting logic.
- `utils/`: Common utilities for path handling and text processing.
- `context/`: History and message state management.

## 🔧 Adding New Tools

Tools are registered in the `tools/registry.py`. To add a new tool:
1. Create a new tool class Inheriting from `BaseTool`.
2. Define the tool's schema and logic.
3. Register it in the `create_default_registry` function.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
