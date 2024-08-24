from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

class Project:
    def __init__(self, folder_name, description=None, start_date=None, end_date=None, status=None):
        self.folder_name = folder_name
        self.description = description
        self.start_date = start_date
        self.end_date = end_date
        self.status = status

    def generate_cypher_query(self, element_id=None):
        if element_id:
            query = f"""
            MATCH (p:Project) WHERE elementId(p) = $element_id
            SET p.folder_name = $folder_name, p.description = $description, p.start_date = $start_date,
                p.end_date = $end_date, p.status = $status
            RETURN p
            """
        else:
            query = """
            CREATE (p:Project {folder_name: $folder_name, description: $description, start_date: $start_date,
                               end_date: $end_date, status: $status})
            RETURN p
            """
        return query


    @staticmethod
    def retrieve_from_database(element_id):
        uri = "bolt://localhost:7689"
        user = "neo4j"
        password = "mynewpassword"
        
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                query = """
                MATCH (p:Project) WHERE elementId(p) = $element_id
                RETURN p.folder_name AS folder_name, p.description AS description, p.start_date AS start_date,
                       p.end_date AS end_date, p.status AS status
                """
                result = session.run(query, element_id=element_id)
                record = result.single()
                if record:
                    return Project(record["folder_name"], record["description"], record["start_date"],
                                   record["end_date"], record["status"])
                return None
