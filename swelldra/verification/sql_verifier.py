import sqlglot
def verify_sql(sql: str, dialect="postgres"):
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
        tables = [t.name for t in parsed.find_all(sqlglot.exp.Table)]
        return {"valid": True, "tables": tables, "dialect": dialect, "sql": parsed.sql(dialect=dialect)}
    except Exception as e:
        return {"valid": False, "error": str(e)}
