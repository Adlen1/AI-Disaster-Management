from loguru import logger
from pathlib import Path
import sys

def setup_logger(log_dir: str = "logs", level: str = "INFO"):
    """
    Configure loguru logger.
    Logs to both console and rotating file.
    Every script imports this once at the top.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger.remove()  # remove default handler
    
    # Console — clean format
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[source]}</cyan> | {message}",
        level=level,
        colorize=True
    )
    
    # File — full format with timestamps
    logger.add(
        f"{log_dir}/pipeline.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[source]} | {message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    return logger