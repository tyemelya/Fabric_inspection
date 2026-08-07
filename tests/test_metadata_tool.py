from app.tools.metadata import MetadataTool
import sqlite3
import pytest

def create_database(db_path):
    connection = sqlite3.connect(db_path)
    cur = connection.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fabric_defects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE,
            defect_class TEXT,
            severity TEXT,
            description TEXT
    )
    """)
    connection.commit()
    connection.close()

def add_record_to_database(db_path):
    connection = sqlite3.connect(db_path)
    cur = connection.cursor()

    cur.execute(
        """
        INSERT OR IGNORE INTO fabric_defects
            (image_path, defect_class, severity, description)
            VALUES (?, ?, ?, ?)
        """,
        (
            "images/test.jpg",
            "broken stitch",
            "medium",
            "Test defect.",
        ),
    )
    connection.commit()
    connection.close()       
    
def test_metadata_get(tmp_path):
    db_path = tmp_path / "test.db"
    create_database(db_path)
    add_record_to_database(db_path)
    
    metadata_tool = MetadataTool(db_path)
    result = metadata_tool.get(1)
    
    assert result.image_path == "images/test.jpg"
    assert result.defect_class == "broken stitch"
    assert result.severity == "medium"
    assert result.description == "Test defect."

def test_metadata_get_missing_id(tmp_path):
    db_path = tmp_path / "test.db"
    create_database(db_path)
    metadata_tool = MetadataTool(db_path)
    
    with pytest.raises(ValueError):
        metadata_tool.get(999)