from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sqlite3, time

@dataclass
class Schema:
    tables: Dict[str, List[str]]
    foreign_keys: List[Tuple[str, str, str, str]]
    dialect: str = "sqlite"

@dataclass
class SQLResult:
    sql: str
    success: bool
    rows: Optional[List]
    error: Optional[str]
    execution_time_ms: float

class SQLEngine:
    def __init__(self, dialect="sqlite"):
        self.dialect = dialect
        self.conn = sqlite3.connect(":memory:")
    def execute_with_feedback(self, sql: str, max_retries=3):
        start = time.time()
        try:
            cur = self.conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return SQLResult(sql, True, rows, None, (time.time()-start)*1000)
        except Exception as e:
            return SQLResult(sql, False, None, str(e), (time.time()-start)*1000)
    def verify(self, sql: str, schema: Schema):
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, read=self.dialect)
            return {"valid": True, "parsed": str(parsed)}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    @staticmethod
    def test_interface():
        engine = SQLEngine()
        engine.conn.execute("CREATE TABLE users (id INT, name TEXT)")
        engine.conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        res = engine.execute_with_feedback("SELECT * FROM users")
        assert res.success and len(res.rows) == 1
        print("SQL interface test passed")
