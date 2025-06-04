# LFG Community Bot

A Discord bot that uses Retrieval-Augmented Generation (RAG) to answer questions based on the LFG support site and other knowledge sources.

## Features

- Answers questions using information from the LFG support site
- Provides sources for its answers
- Admin commands to refresh the knowledge base
- Easy to extend with additional knowledge sources

## Prerequisites

- Python 3.9 or higher
- Discord Bot Token ([Create one here](https://discord.com/developers/applications))
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd lfg-community-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment file and update it with your credentials:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file and add your Discord bot token and OpenAI API key.

4. Run the bot:
   ```bash
   python bot.py
   ```

## Usage

- `!ask <question>` - Ask a question about LFG services
- `!refresh` - Refresh the knowledge base (Admin only)

## Adding More Knowledge Sources

To add more knowledge sources, modify the `initialize_rag` method in `cogs/rag_cog.py`. You can add more URLs to the `WebBaseLoader` or use other document loaders from LangChain.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
