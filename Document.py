from File import File
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

class Document(File):
    uri = "bolt://localhost:7689"
    user = "neo4j"
    password = "mynewpassword"

    def __init__(self, filename, full_path, size_in_bytes, created_date=None, modified_date=None, extension=None, content_type=None):
        super().__init__(filename, full_path, size_in_bytes, created_date, modified_date, extension)
        self.content_type = content_type

    def generate_cypher_query(self, element_id=None):
        properties = {
            "filename": self.filename,
            "full_path": self.full_path,
            "size_in_bytes": self.size_in_bytes,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "extension": self.extension,
            "content_type": self.content_type
        }
        if element_id:
            query = f"""
            MATCH (d:File) WHERE elementId(d) = $element_id
            SET d:Document
            SET d += $properties
            RETURN d
            """
        else:
            query = f"""
            CREATE (d:File:Document)
            SET d = $properties
            RETURN d
            """
        return query

    def execute_cypher_query(self, query, element_id=None):
        try:
            with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
                with driver.session() as session:
                    params = {
                        "properties": {
                            "filename": self.filename,
                            "full_path": self.full_path,
                            "size_in_bytes": self.size_in_bytes,
                            "created_date": self.created_date,
                            "modified_date": self.modified_date,
                            "extension": self.extension,
                            "content_type": self.content_type
                        }
                    }
                    if element_id is not None:
                        params["element_id"] = element_id
                    result = session.run(query, **params)
                    record = result.single()
                    if record:
                        return str(record["d"].element_id), True
                    return None, False
        except Neo4jError as e:
            print(f"An error occurred: {e}")
            return None, False

    @staticmethod
    def retrieve_from_database(element_id):
        try:
            with GraphDatabase.driver(Document.uri, auth=(Document.user, Document.password)) as driver:
                with driver.session() as session:
                    query = """
                    MATCH (d:Document) WHERE elementId(d) = $element_id
                    RETURN d
                    """
                    result = session.run(query, element_id=element_id)
                    record = result.single()
                    if record:
                        node = record["d"]
                        return Document(
                            node["filename"], node["full_path"], node["size_in_bytes"],
                            node["created_date"], node["modified_date"], node["extension"],
                            node["content_type"]
                        )
                    return None
        except Neo4jError as e:
            print(f"An error occurred: {e}")
            return None

    def add_label(self, label):
        query = f"""
        MATCH (d:Document) WHERE elementId(d) = $element_id
        SET d:{label}
        RETURN d
        """
        return self.execute_cypher_query(query, self.element_id)

    def remove_label(self, label):
        query = f"""
        MATCH (d:Document) WHERE elementId(d) = $element_id
        REMOVE d:{label}
        RETURN d
        """
        return self.execute_cypher_query(query, self.element_id)
