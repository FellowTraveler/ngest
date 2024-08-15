from neo4j import GraphDatabase
from datetime import datetime

class File:
    def __init__(self, filename, full_path, size_in_bytes, created_date=None, modified_date=None, extension=None):
        self.filename = filename
        self.full_path = full_path
        self.size_in_bytes = size_in_bytes
        self.created_date = created_date or datetime.now().isoformat()
        self.modified_date = modified_date or datetime.now().isoformat()
        self.extension = extension or self.get_extension(filename)
        self.uri = "bolt://localhost:7689"
        self.user = "neo4j"
        self.password = "mynewpassword"

    @staticmethod
    def get_extension(filename):
        return filename.split('.')[-1] if '.' in filename else ''

    def generate_cypher_query(self, element_id=None):
        if element_id:
            query = f"""
            MATCH (f:File) WHERE elementId(f) = $element_id
            SET f.filename = $filename, f.full_path = $full_path, f.size_in_bytes = $size_in_bytes,
                f.created_date = $created_date, f.modified_date = $modified_date, f.extension = $extension
            RETURN f
            """
        else:
            query = """
            CREATE (f:File {filename: $filename, full_path: $full_path, size_in_bytes: $size_in_bytes,
                            created_date: $created_date, modified_date: $modified_date, extension: $extension})
            RETURN f
            """
        return query

    def execute_cypher_query(self, query, element_id=None):
        with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
            with driver.session() as session:
                params = {
                    "filename": self.filename,
                    "full_path": self.full_path,
                    "size_in_bytes": self.size_in_bytes,
                    "created_date": self.created_date,
                    "modified_date": self.modified_date,
                    "extension": self.extension
                }
                if element_id is not None:
                    params["element_id"] = element_id
                result = session.run(query, **params)
                record = result.single()
                if record:
                    return str(record["f"].element_id), True
                return None, False

    @staticmethod
    def retrieve_from_database(element_id):
        uri = "bolt://localhost:7689"
        user = "neo4j"
        password = "mynewpassword"
        
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                query = """
                MATCH (f:File) WHERE elementId(f) = $element_id
                RETURN f.filename AS filename, f.full_path AS full_path, f.size_in_bytes AS size_in_bytes,
                       f.created_date AS created_date, f.modified_date AS modified_date, f.extension AS extension
                """
                result = session.run(query, element_id=element_id)
                record = result.single()
                if record:
                    return File(record["filename"], record["full_path"], record["size_in_bytes"],
                                record["created_date"], record["modified_date"], record["extension"])
                return None
