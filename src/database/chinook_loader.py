"""
Database loader for Chinook SQLite DB.
Sets up in-memory SQLite and wraps it in LangChain's SQLDatabase.
"""

import sqlite3
import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_community.utilities.sql_database import SQLDatabase


def get_engine_for_chinook_db():
    """
    Downloads the Chinook SQL script, loads it into
    an in-memory SQLite database, and returns a SQLAlchemy engine.
    """
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    
    # Fetch SQL script
    response = requests.get(url)
    sql_script = response.text #gives you the content of the file as a string 

    # Create in-memory SQLite connection
    connection = sqlite3.connect(":memory:", check_same_thread=False) #Creates a new SQLite database in RAM 
    connection.executescript(sql_script)

    # Create SQLAlchemy engine
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def get_chinook_db() -> SQLDatabase:
    """
    Returns a LangChain SQLDatabase instance
    connected to the Chinook DB.
    """
    engine = get_engine_for_chinook_db()
    return SQLDatabase(engine)
