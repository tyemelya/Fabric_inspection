from dataclasses import dataclass
import sqlite3

@dataclass
class Metadata:
    image_path: str
    defect_class: str
    severity: str
    description: str

class MetadataTool:
    def __init__(self, db_path="data/fabric.db"):
        self.db_path = db_path
        
    def get(self, idx: int) -> Metadata:
        with sqlite3.connect(self.db_path) as connection:
            cur = connection.cursor()
            cur.execute(
                 """
                     SELECT image_path, defect_class, severity, description 
                    FROM fabric_defects
                    WHERE id = ?
                    LIMIT 1
                """,
                (idx,)
            )
            res = cur.fetchone()
            if res is None:
                raise ValueError(f"No metadata found for id {idx}")
        
        return Metadata(
            image_path = res[0],
            defect_class = res[1],
            severity = res[2],
            description = res[3]
        )
