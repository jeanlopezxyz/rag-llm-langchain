# src/utils/db_utils.py

import psycopg
import logging

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Manages a simple, persistent connection to PostgreSQL."""

    def __init__(self, db_params: dict):
        self.conninfo = " ".join([f"{k}='{v}'" for k, v in db_params.items()])
        self.connection = None
        self._connect()

    def _connect(self):
        """Establishes the database connection."""
        try:
            if not self.connection or self.connection.closed:
                self.connection = psycopg.connect(self.conninfo)
                logger.info("Database connection established.")
        except psycopg.OperationalError as e:
            logger.error(f"Error connecting to the database: {e}")
            raise

    def execute_query(self, query, params=None, fetch=None):
        """Execute a query, ensuring the connection is open."""
        self._connect()  # Ensure connection is alive
        try:
            with self.connection.cursor() as cur:
                cur.execute(query, params)
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                self.connection.commit()
                return None
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Query failed: {e}", exc_info=True)
            raise

    def get_connection(self):
        """Returns the managed connection, ensuring it's open."""
        self._connect()
        return self.connection

    def close(self):
        """Close the connection."""
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("Database connection closed.")