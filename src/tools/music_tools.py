"""
Music-related tools for querying the Chinook DB.
"""

from sqlalchemy import text
from src.database.chinook_loader import get_chinook_db

db = get_chinook_db()


def get_albums_by_artist(artist: str):
    """Retrieve albums by a given artist. Returns list of album titles."""
    query = text("""
        SELECT Title
        FROM Album
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.Name LIKE :artist
    """)
    with db._engine.connect() as conn:
        result = conn.execute(query, {"artist": f"%{artist}%"}).fetchall()
    return [row[0] for row in result]   # ✅ clean list


def get_tracks_by_artist(artist: str):
    """Retrieve tracks by a given artist. Returns list of track names."""
    query = text("""
        SELECT Track.Name
        FROM Track
        JOIN Album ON Track.AlbumId = Album.AlbumId
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.Name LIKE :artist
    """)
    with db._engine.connect() as conn:
        result = conn.execute(query, {"artist": f"%{artist}%"}).fetchall()
    return [row[0] for row in result]   # ✅ clean list


def get_songs_by_genre(genre: str):
    """Retrieve songs by a given genre. Returns list of song titles."""
    query = text("""
        SELECT Track.Name
        FROM Track
        JOIN Genre ON Track.GenreId = Genre.GenreId
        WHERE Genre.Name LIKE :genre
    """)
    with db._engine.connect() as conn:
        result = conn.execute(query, {"genre": f"%{genre}%"}).fetchall()
    return [row[0] for row in result]   # ✅ clean list


def check_for_songs(song_title: str):
    """Check if a song exists by its title. Returns dict with {exists: bool}."""
    query = text("""
        SELECT COUNT(*)
        FROM Track
        WHERE Name LIKE :title
    """)
    with db._engine.connect() as conn:
        result = conn.execute(query, {"title": f"%{song_title}%"}).fetchone()
    exists = result[0] > 0 if result else False
    return {"exists": exists}   # ✅ clean dict
