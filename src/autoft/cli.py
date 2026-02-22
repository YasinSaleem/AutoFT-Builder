"""Command-line interface for AutoFT-Builder."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from autoft import __version__
from autoft.config import Config
from autoft.embedder import EmbeddingModel
from autoft.export import export_from_database
from autoft.filter import SimilarityFilter
from autoft.generator import GenerationError, LLMGenerator
from autoft.storage import Database

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="autoft",
        description="AutoFT-Builder: Generate high-quality synthetic fine-tuning datasets with semantic deduplication.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic data batch",
    )
    generate_parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description for data generation",
    )
    generate_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of samples to generate per batch (default: 10)",
    )
    generate_parser.add_argument(
        "--batches",
        type=int,
        default=1,
        help="Number of batches to generate (default: 1)",
    )
    generate_parser.add_argument(
        "--format",
        type=str,
        default=None,
        help="Output format specification (optional)",
    )
    generate_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for filtering (default: from config)",
    )
    generate_parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path (default: from config)",
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export dataset to JSONL file",
    )
    export_parser.add_argument(
        "--output",
        type=str,
        default="dataset.jsonl",
        help="Output file path (default: dataset.jsonl)",
    )
    export_parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path (default: from config)",
    )

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show dataset statistics",
    )
    stats_parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path (default: from config)",
    )

    # Clear command
    clear_parser = subparsers.add_parser(
        "clear",
        help="Clear all data from the dataset",
    )
    clear_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    clear_parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path (default: from config)",
    )

    return parser


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity.

    Args:
        verbose: Enable debug-level logging
        quiet: Suppress all output except errors
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure root logger for our app
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Set our logger level
    logging.getLogger("autoft").setLevel(level)


def get_database_path(args: argparse.Namespace, config: Config) -> Path:
    """Get database path from args or config."""
    if args.db:
        return Path(args.db)
    return config.database_path


def cmd_generate(args: argparse.Namespace, config: Config) -> int:
    """Execute the generate command.

    Pipeline flow:
    1. Load existing embeddings from database
    2. Generate batch via LLM
    3. Embed each sample
    4. Filter by similarity (vs existing + vs batch)
    5. Store accepted samples
    6. Report statistics
    """
    db_path = get_database_path(args, config)
    threshold = (
        args.threshold if args.threshold is not None else config.similarity_threshold
    )

    # Header
    logger.info("=" * 50)
    logger.info("AutoFT-Builder - Generate Mode")
    logger.info("=" * 50)
    logger.info(f"Task: {args.task[:80]}{'...' if len(args.task) > 80 else ''}")
    logger.info(f"Batch size: {args.batch_size} samples x {args.batches} batches")
    logger.info(f"Similarity threshold: {threshold}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Model: {config.llm_model}")
    logger.info("")

    # Step 1: Initialize LLM generator
    logger.info("[1/4] Initializing LLM generator...")
    generator = LLMGenerator(
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
        model=config.llm_model,
    )
    logger.debug(f"  Using model: {config.llm_model}")
    logger.debug(f"  API base URL: {config.openrouter_base_url}")

    # Step 2: Initialize embedding model
    logger.info("[2/4] Loading embedding model (BAAI/bge-small-en-v1.5)...")
    embedder = EmbeddingModel()
    # Trigger model loading by encoding a dummy text
    _ = embedder.encode("warmup")
    logger.info("  Embedding model loaded successfully")

    # Step 3: Initialize similarity filter
    logger.info(f"[3/4] Initializing similarity filter (threshold={threshold})...")
    similarity_filter = SimilarityFilter(threshold=threshold)

    # Step 4: Connect to database
    logger.info("[4/4] Connecting to database...")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    total_accepted = 0
    total_rejected = 0

    with Database(db_path) as db:
        initial_count = db.get_sample_count()
        logger.info(f"  Database connected: {initial_count} existing samples")
        logger.info("")

        # Generation loop
        logger.info("Starting generation...")
        logger.info("-" * 50)

        for batch_num in tqdm(
            range(args.batches),
            desc="Generating",
            disable=logging.getLogger().level > logging.INFO,
        ):
            batch_label = f"Batch {batch_num + 1}/{args.batches}"

            # Load existing embeddings (including newly added ones)
            existing_embeddings = db.get_all_embeddings()
            existing_count = (
                len(existing_embeddings) if existing_embeddings.size > 0 else 0
            )

            # Generate samples via LLM
            logger.debug(f"{batch_label}: Calling LLM API...")
            try:
                samples = generator.generate_batch(
                    task_description=args.task,
                    batch_size=args.batch_size,
                    format_spec=args.format,
                )
            except GenerationError as e:
                logger.error(f"{batch_label}: Generation failed - {e}")
                continue

            generated_count = len(samples)
            total_generated += generated_count

            if not samples:
                logger.warning(f"{batch_label}: No samples generated")
                continue

            logger.debug(f"{batch_label}: Generated {generated_count} samples")

            # Embed samples
            logger.debug(f"{batch_label}: Computing embeddings...")
            embeddings = embedder.encode_samples_batch(
                [(s.instruction, s.output) for s in samples]
            )

            # Filter by similarity
            logger.debug(
                f"{batch_label}: Filtering against {existing_count} existing samples..."
            )
            filter_result = similarity_filter.filter_batch_detailed(
                new_embeddings=embeddings,
                existing_embeddings=existing_embeddings,
            )

            accepted_count = len(filter_result.accepted_indices)
            rejected_count = len(filter_result.rejected_indices)
            total_accepted += accepted_count
            total_rejected += rejected_count

            # Store accepted samples
            if accepted_count > 0:
                samples_to_store = [
                    (
                        samples[i].instruction,
                        samples[i].output,
                        embeddings[i],
                    )
                    for i in filter_result.accepted_indices
                ]
                db.insert_samples_batch(samples_to_store)

            # Log batch results
            logger.info(
                f"{batch_label}: Generated {generated_count}, "
                f"accepted {accepted_count}, rejected {rejected_count}"
            )

            # Log rejection info in verbose mode
            if rejected_count > 0 and logger.isEnabledFor(logging.DEBUG):
                # Show first 3 rejections
                for idx, sim in list(filter_result.similarities.items())[:3]:
                    logger.debug(f"  Rejected sample {idx}: similarity={sim:.3f}")

    # Final summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("Generation Complete")
    logger.info("=" * 50)
    logger.info(f"Total samples generated: {total_generated}")
    logger.info(f"Total samples accepted:  {total_accepted}")
    logger.info(f"Total samples rejected:  {total_rejected}")

    if total_generated > 0:
        acceptance_rate = (total_accepted / total_generated) * 100
        logger.info(f"Acceptance rate:         {acceptance_rate:.1f}%")

    with Database(db_path) as db:
        final_count = db.get_sample_count()
        new_samples = final_count - initial_count
        logger.info(f"New samples added:       {new_samples}")
        logger.info(f"Final dataset size:      {final_count} samples")

    return 0


def cmd_export(args: argparse.Namespace, config: Config) -> int:
    """Execute the export command."""
    db_path = get_database_path(args, config)
    output_path = Path(args.output)

    logger.info("=" * 50)
    logger.info("AutoFT-Builder - Export Mode")
    logger.info("=" * 50)
    logger.info(f"Database: {db_path}")
    logger.info(f"Output:   {output_path}")
    logger.info("")

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return 1

    with Database(db_path) as db:
        count = db.get_sample_count()
        if count == 0:
            logger.warning("No samples in database to export.")
            return 0

        logger.info(f"Exporting {count} samples...")
        exported = export_from_database(db, output_path)

    logger.info(f"Successfully exported {exported} samples to {output_path}")
    return 0


def cmd_stats(args: argparse.Namespace, config: Config) -> int:
    """Execute the stats command."""
    db_path = get_database_path(args, config)

    logger.info("=" * 50)
    logger.info("AutoFT-Builder - Statistics")
    logger.info("=" * 50)
    logger.info(f"Database: {db_path}")
    logger.info("")

    if not db_path.exists():
        logger.warning(f"Database not found at {db_path}")
        logger.info("No data available.")
        return 0

    with Database(db_path) as db:
        count = db.get_sample_count()

        logger.info(f"Total samples:      {count}")

        if count > 0:
            # Get database file size
            db_size = db_path.stat().st_size
            if db_size < 1024:
                size_str = f"{db_size} bytes"
            elif db_size < 1024 * 1024:
                size_str = f"{db_size / 1024:.1f} KB"
            else:
                size_str = f"{db_size / (1024 * 1024):.1f} MB"
            logger.info(f"Database size:      {size_str}")
            logger.info(f"Embedding dimension: {db.embedding_dim}")

            # Show sample preview
            sample = db.get_sample_by_id(1)
            if sample:
                logger.info("")
                logger.info("Sample preview (first entry):")
                instruction = sample["instruction"][:60]
                if len(sample["instruction"]) > 60:
                    instruction += "..."
                logger.info(f"  Instruction: {instruction}")

    return 0


def cmd_clear(args: argparse.Namespace, config: Config) -> int:
    """Execute the clear command."""
    db_path = get_database_path(args, config)

    logger.info("=" * 50)
    logger.info("AutoFT-Builder - Clear Database")
    logger.info("=" * 50)
    logger.info(f"Database: {db_path}")
    logger.info("")

    if not db_path.exists():
        logger.warning(f"Database not found at {db_path}")
        return 0

    with Database(db_path) as db:
        count = db.get_sample_count()
        if count == 0:
            logger.info("Database is already empty.")
            return 0

        logger.warning(f"This will delete {count} samples.")

        if not args.force:
            confirm = input("Are you sure you want to clear all data? [y/N]: ")
            if confirm.lower() != "y":
                logger.info("Aborted.")
                return 0

        deleted = db.clear()
        logger.info(f"Deleted {deleted} samples.")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Set up logging
    setup_logging(
        verbose=getattr(args, "verbose", False),
        quiet=getattr(args, "quiet", False),
    )

    if args.command is None:
        parser.print_help()
        return 0

    # Load config (allow missing API key for some commands)
    try:
        config = Config.from_env()
    except ValueError as e:
        # Allow stats, export, and clear without API key
        if args.command in ("stats", "export", "clear"):
            # Create minimal config for these commands
            import os
            from pathlib import Path

            config = Config(
                openrouter_api_key="",
                openrouter_base_url=os.getenv(
                    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                ),
                llm_model=os.getenv("LLM_MODEL", "anthropic/claude-3-haiku"),
                similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
                database_path=Path(os.getenv("DATABASE_PATH", "data/dataset.db")),
            )
        else:
            logger.error(f"Configuration error: {e}")
            return 1

    # Dispatch to command handlers
    if args.command == "generate":
        return cmd_generate(args, config)
    elif args.command == "export":
        return cmd_export(args, config)
    elif args.command == "stats":
        return cmd_stats(args, config)
    elif args.command == "clear":
        return cmd_clear(args, config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
