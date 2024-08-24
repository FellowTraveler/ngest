from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

class Project:
    def __init__(self, name, description=None, start_date=None, end_date=None, status=None):
        self.name = name
        self.description = description
        self.start_date = start_date
        self.end_date = end_date
        self.status = status
        self.uri = "bolt://localhost:7689"
        self.user = "neo4j"
        self.password = "mynewpassword"

    def generate_cypher_query(self, element_id=None):
        if element_id:
            query = f"""
            MATCH (p:Project) WHERE elementId(p) = $element_id
            SET p.name = $name, p.description = $description, p.start_date = $start_date,
                p.end_date = $end_date, p.status = $status
            RETURN p
            """
        else:
            query = """
            CREATE (p:Project {name: $name, description: $description, start_date: $start_date,
                               end_date: $end_date, status: $status})
            RETURN p
            """
        return query

    def execute_cypher_query(self, query, element_id=None):
        with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
            with driver.session() as session:
                params = {
                    "name": self.name,
                    "description": self.description,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "status": self.status
                }
                if element_id is not None:
                    params["element_id"] = element_id
                result = session.run(query, **params)
                record = result.single()
                if record:
                    return str(record["p"].element_id), True
                return None, False

    @staticmethod
    def retrieve_from_database(element_id):
        uri = "bolt://localhost:7689"
        user = "neo4j"
        password = "mynewpassword"
        
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                query = """
                MATCH (p:Project) WHERE elementId(p) = $element_id
                RETURN p.name AS name, p.description AS description, p.start_date AS start_date,
                       p.end_date AS end_date, p.status AS status
                """
                result = session.run(query, element_id=element_id)
                record = result.single()
                if record:
                    return Project(record["name"], record["description"], record["start_date"],
                                   record["end_date"], record["status"])
                return None
