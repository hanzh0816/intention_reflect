"""
SimulationHistory serialization and deserialization using pickle + lzma compression.
"""
import pickle
import lzma
from dataclasses import dataclass
import logging

from nuplan.planning.simulation.history.simulation_history import SimulationHistory

logger = logging.getLogger(__name__)


@dataclass
class SerializedHistory:
    """Container for serialized history data."""

    data: bytes
    format: str
    size_bytes: int
    num_samples: int


class HistorySerializer:
    """
    Serializes and deserializes SimulationHistory objects.

    Uses pickle + lzma compression for efficient storage.
    """

    def serialize(self, history: SimulationHistory) -> SerializedHistory:
        """
        Serialize SimulationHistory to compressed bytes.

        Args:
            history: SimulationHistory object

        Returns:
            SerializedHistory with compressed data
        """
        try:
            # Pickle the history object
            pickled_data = pickle.dumps(history, protocol=pickle.HIGHEST_PROTOCOL)

            # Compress with lzma (preset=3 for balanced compression)
            compressed_data = lzma.compress(pickled_data, preset=3)

            compression_ratio = len(compressed_data) / len(pickled_data) if len(pickled_data) > 0 else 1.0

            logger.debug(
                f"Serialized history: {len(pickled_data)} bytes → "
                f"{len(compressed_data)} bytes "
                f"(compression ratio: {compression_ratio:.2%})"
            )

            return SerializedHistory(
                data=compressed_data,
                format="pickle_lzma",
                size_bytes=len(compressed_data),
                num_samples=len(history),
            )

        except Exception as e:
            logger.error(f"Failed to serialize history: {e}", exc_info=True)
            raise

    def deserialize(self, serialized: SerializedHistory) -> SimulationHistory:
        """
        Deserialize compressed bytes back to SimulationHistory.

        Args:
            serialized: SerializedHistory object

        Returns:
            SimulationHistory object
        """
        try:
            if serialized.format != "pickle_lzma":
                raise ValueError(f"Unsupported format: {serialized.format}")

            # Decompress
            decompressed_data = lzma.decompress(serialized.data)

            # Unpickle
            history = pickle.loads(decompressed_data)

            logger.debug(
                f"Deserialized history: {len(history)} samples"
            )

            return history

        except Exception as e:
            logger.error(f"Failed to deserialize history: {e}", exc_info=True)
            raise

    def deserialize_from_bytes(self, blob_data: bytes) -> SimulationHistory:
        """
        Deserialize history directly from compressed bytes (from database BLOB).

        Args:
            blob_data: Compressed history bytes (lzma compressed pickle)

        Returns:
            SimulationHistory object
        """
        try:
            # Decompress
            decompressed_data = lzma.decompress(blob_data)

            # Unpickle
            history = pickle.loads(decompressed_data)

            logger.debug(
                f"Deserialized history from bytes: {len(history)} samples"
            )

            return history

        except Exception as e:
            logger.error(f"Failed to deserialize history from bytes: {e}", exc_info=True)
            raise
