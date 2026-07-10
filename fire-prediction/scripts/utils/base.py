"""
Base interface for all data sources.
Every source (GADM, DEM, WorldPop, ERA5, FIRMS, Sentinel) 
inherits from this and implements the same three methods.
This is what makes the pipeline fully automatable  
the orchestrator just loops over sources and calls the same methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger
import yaml


class DataSource(ABC):
    """
    Abstract base class for all data sources.
    
    Every data source must implement:
      - ingest(start_date, end_date) : download raw data
      - curate()                     : clean and validate
      - load()                       : return ready-to-use dataframe/geodataframe
    
    Static sources (DEM, WorldPop, OSM, GADM) ignore start/end dates in ingest() since they are downloaded once and reused.
    
    Dynamic sources (ERA5, Sentinel-2, FIRMS) use start/end dates to define the time window, different dates for training vs operational mode.
    """

    def __init__(self, config: dict):
        self.config   = config
        self.raw_dir  = Path(config["paths"]["raw"])
        self.cur_dir  = Path(config["paths"]["curated"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cur_dir.mkdir(parents=True, exist_ok=True)
        self.logger   = logger.bind(source=self.__class__.__name__)

    @abstractmethod
    def ingest(self, start_date: str = None, end_date: str = None) -> None:
        """Download raw data. Static sources ignore dates."""
        pass

    @abstractmethod
    def curate(self) -> None:
        """Clean, validate, reproject, clip to Algeria. Save to curated/."""
        pass

    @abstractmethod
    def load(self):
        """Return the curated data ready for integration."""
        pass

    def run(self, start_date: str = None, end_date: str = None):
        """
        Full pipeline for this source:
          ingest → curate → load
        
        Call this from the orchestrator.
        Same call signature for all sources.
        Training:    source.run("2018-01-01", "2025-12-31")
        Operational: source.run("2026-07-09", "2026-07-09")
        Static:      source.run()  # dates ignored
        """
        self.logger.info(f"Starting ingestion [{start_date} → {end_date}]")
        self.ingest(start_date, end_date)
        self.logger.info("Ingestion complete. Starting curation.")
        self.curate()
        self.logger.info("Curation complete. Loading.")
        return self.load()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)