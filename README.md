# AutoFT-Builder

**Intelligent Fine-Tuning Dataset Generator with Semantic Deduplication**

A lightweight CLI pipeline that generates high-quality synthetic fine-tuning datasets using LLMs while automatically filtering semantically redundant samples using embedding similarity.

## Features

- **LLM-based Data Generation**: Generate synthetic training data via OpenRouter API (supports 100+ models)
- **Semantic Deduplication**: Filter near-duplicate samples using cosine similarity on embeddings
- **Local Embeddings**: Uses BAAI/bge-small-en-v1.5 model (384 dimensions, runs on CPU)
- **Persistent Storage**: SQLite database for incremental dataset building
- **JSONL Export**: Export datasets in fine-tuning-ready format
- **Batch Processing**: Generate multiple batches with configurable sizes
- **Progress Tracking**: Real-time progress bars and detailed logging
- **Robust Error Handling**: Automatic retries with exponential backoff for API errors

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install from source

```bash
# Clone the repository
git clone https://github.com/yasinsaleem/AutoFT-Builder.git
cd AutoFT-Builder

# Install dependencies (creates virtual environment automatically)
uv sync

# Or using make
make install-dev
```

### Alternative: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e ".[dev]"
```

## Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional (defaults shown)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=anthropic/claude-3-haiku
SIMILARITY_THRESHOLD=0.85
DATABASE_PATH=data/dataset.db
```

### Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key (required for generate) | - |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | Model to use for generation | `anthropic/claude-3-haiku` |
| `SIMILARITY_THRESHOLD` | Cosine similarity threshold for deduplication (0-1) | `0.85` |
| `DATABASE_PATH` | Path to SQLite database file | `data/dataset.db` |

## Usage

**Important**: All commands should be run with `uv run` to ensure the correct virtual environment is used.

### Generate Data

Generate synthetic training samples with automatic deduplication:

```bash
# Basic generation
uv run autoft generate --task "Generate question-answer pairs about Python programming"

# With custom batch size and multiple batches
uv run autoft generate \
  --task "Generate coding exercises for beginners" \
  --batch-size 20 \
  --batches 5

# Override similarity threshold
uv run autoft generate \
  --task "Generate diverse writing prompts" \
  --threshold 0.9

# Use a different database
uv run autoft generate \
  --task "Generate math problems" \
  --db custom_dataset.db

# With format specification (inline)
uv run autoft generate \
  --task "Generate JSON tool calls for a scheduling assistant" \
  --format "Output must be valid JSON with 'instruction' and 'output' fields. Include diverse examples."

# With format specification from file
uv run autoft generate \
  --task "Generate scheduling assistant data" \
  --format "$(cat format.txt)"

# Verbose mode (show debug info)
uv run autoft -v generate --task "Generate Q&A pairs" --batch-size 10

# Quiet mode (errors only)
uv run autoft -q generate --task "Generate Q&A pairs" --batch-size 10
```

### Using Format Specification

The `--format` option allows you to provide detailed output format instructions. This is especially useful for generating structured data like JSON tool calls.

**Tip**: For complex format specifications, use a file:

```bash
# Create format.txt in your project root
cat > format.txt << 'EOF'
Generate JSON tool calls with these rules:
- Output must have "instruction" and "output" fields
- Use tool names: create_event, reschedule_event, cancel_event
- Include diverse user requests
- Dates in YYYY-MM-DD format
EOF

# Use the file
uv run autoft generate \
  --task "Generate scheduling assistant requests" \
  --format "$(cat format.txt)"
```

For complex prompts with JSON examples, avoid putting them in `--task` - use `--format` instead to prevent template conflicts.

### Export Dataset

Export the dataset to a JSONL file for fine-tuning:

```bash
# Export to default location
uv run autoft export --output dataset.jsonl

# Export from specific database
uv run autoft export --db my_dataset.db --output training_data.jsonl
```

### View Statistics

Show dataset statistics:

```bash
uv run autoft stats

# Example output:
# ==================================================
# AutoFT-Builder - Statistics
# ==================================================
# Database: data/dataset.db
#
# Total samples:      150
# Database size:      245.5 KB
# Embedding dimension: 384
#
# Sample preview (first entry):
#   Instruction: What is a Python list?...
```

### Clear Dataset

Clear all data from the dataset:

```bash
uv run autoft clear          # With confirmation prompt
uv run autoft clear --force  # Skip confirmation
```

## CLI Reference

```
uv run autoft <command> [options]

Commands:
  generate    Generate synthetic training data
  export      Export dataset to JSONL file
  stats       Show dataset statistics
  clear       Clear the dataset

Global Options:
  --verbose, -v  Enable verbose/debug output
  --quiet, -q    Suppress all output except errors
  --db PATH      Path to database file (overrides DATABASE_PATH)
  -h, --help     Show help message

Generate Options:
  --task TEXT        Task description for data generation (required)
  --batch-size INT   Samples per batch (default: 10)
  --batches INT      Number of batches to generate (default: 1)
  --format TEXT      Additional format specification
  --threshold FLOAT  Similarity threshold override (0.0-1.0)

Export Options:
  --output PATH      Output JSONL file path (required)

Clear Options:
  --force            Skip confirmation prompt
```

## Example Output

When running `uv run autoft generate`, you'll see detailed progress:

```
==================================================
AutoFT-Builder - Generate Mode
==================================================
Task: Generate question-answer pairs about Python programming...
Batch size: 10 samples x 5 batches
Similarity threshold: 0.85
Database: data/dataset.db
Model: anthropic/claude-3-haiku

[1/4] Initializing LLM generator...
[2/4] Loading embedding model (BAAI/bge-small-en-v1.5)...
  Embedding model loaded successfully
[3/4] Initializing similarity filter (threshold=0.85)...
[4/4] Connecting to database...
  Database connected: 0 existing samples

Starting generation...
--------------------------------------------------
Batch 1/5: Generated 10, accepted 10, rejected 0
Batch 2/5: Generated 10, accepted 9, rejected 1
Batch 3/5: Generated 10, accepted 8, rejected 2
Batch 4/5: Generated 10, accepted 7, rejected 3
Batch 5/5: Generated 10, accepted 6, rejected 4

==================================================
Generation Complete
==================================================
Total samples generated: 50
Total samples accepted:  40
Total samples rejected:  10
Acceptance rate:         80.0%
New samples added:       40
Final dataset size:      40 samples
```

## How It Works

### Pipeline Architecture

```
┌─────────────┐    ┌───────────┐    ┌─────────────────┐    ┌──────────┐    ┌─────────────┐
│   Generate  │ →  │   Embed   │ →  │ Similarity Check│ →  │  Filter  │ →  │    Store    │
│    (LLM)    │    │  (BGE)    │    │   (Cosine)      │    │(Dedupe)  │    │  (SQLite)   │
└─────────────┘    └───────────┘    └─────────────────┘    └──────────┘    └─────────────┘
                                                                                  │
                                                                                  ↓
                                                                          ┌─────────────┐
                                                                          │   Export    │
                                                                          │  (JSONL)    │
                                                                          └─────────────┘
```

### Step-by-Step Process

1. **Generate**: LLM creates instruction-output pairs based on your task description
2. **Embed**: Each sample is embedded using BAAI/bge-small-en-v1.5 (384-dim vectors)
3. **Compare**: New embeddings are compared against existing dataset using cosine similarity
4. **Filter**: Samples with similarity > threshold are rejected as near-duplicates
5. **Store**: Unique samples are stored in SQLite with their embeddings
6. **Export**: Dataset can be exported to JSONL format for fine-tuning

### Deduplication Strategy

- **Cross-batch**: New samples are compared against all existing samples in the database
- **Within-batch**: New samples are also compared against each other within the same batch
- **Configurable threshold**: Default 0.85 (higher = stricter, fewer duplicates allowed)

## Output Format

Exported JSONL files contain one sample per line:

```json
{"instruction": "What is a Python list?", "output": "A Python list is a mutable, ordered collection that can hold items of different types..."}
{"instruction": "How do you define a function?", "output": "In Python, you define a function using the def keyword followed by the function name..."}
```

This format is compatible with most fine-tuning frameworks including:
- OpenAI fine-tuning
- Hugging Face Transformers
- LLaMA-Factory
- Axolotl

## Development

### Project Structure

```
AutoFT-Builder/
├── src/autoft/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration management
│   ├── generator/          # LLM generation module
│   ├── embedder/           # Embedding module (BGE)
│   ├── filter/             # Similarity filtering
│   ├── storage/            # SQLite database
│   └── export/             # JSONL export
├── tests/                  # Test suite (98% coverage)
├── data/                   # Default database location
└── pyproject.toml          # Project configuration
```

### Running Tests

```bash
# Run all unit tests
make test

# Run with coverage report
make coverage

# Run integration tests (requires more time)
uv run pytest -m integration

# Run tests with real models/APIs (requires API key)
uv run pytest -m real
```

### Code Quality

```bash
# Run linter
make lint

# Auto-fix linting issues
make lint-fix

# Run type checker
make typecheck

# Format code
make format

# Run all checks (lint + typecheck + test)
make check
```

## Troubleshooting

### Common Issues

**"Command not found: autoft"**
- Make sure you're using `uv run autoft` instead of just `autoft`
- Or activate the virtual environment first: `source .venv/bin/activate`

**"OPENROUTER_API_KEY environment variable is required"**
- Make sure you've created a `.env` file with your API key
- Or set the environment variable: `export OPENROUTER_API_KEY=your_key`

**"Model not found" or API errors**
- Check that your API key is valid
- Verify the model name is correct (e.g., `anthropic/claude-3-haiku`)
- Check your OpenRouter account has credits

**"Connection error" during generation**
- The pipeline automatically retries with exponential backoff
- Check your internet connection
- OpenRouter may be experiencing issues

**High memory usage with large datasets**
- The export uses streaming to handle large datasets efficiently
- For very large datasets (>100k samples), consider using batch operations

**Embedding model download**
- On first run, the BGE model (~130MB) is downloaded automatically
- Ensure you have internet access and sufficient disk space

### Performance Tips

- Use larger batch sizes (20-50) for better throughput
- The embedding model is loaded lazily and cached
- Database operations are optimized for batch inserts
- Export uses streaming for memory efficiency

## License

MIT License

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass (`make test`)
2. Code passes linting (`make lint`)
3. Types are correct (`make typecheck`)
4. Coverage remains >90%
