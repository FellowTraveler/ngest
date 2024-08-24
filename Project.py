from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

class Project:
    def __init__(self, project_id, folder_name, description=None, start_date=None, end_date=None, status=None, created_date=None, modified_date=None):
        self.project_id = project_id
        self.folder_name = folder_name
        self.created_date = created_date or datetime.now().isoformat()
        self.modified_date = modified_date or datetime.now().isoformat()
        self.description = description
        self.start_date = start_date
        self.end_date = end_date
        self.status = status

    def generate_cypher_query(self, element_id=None):
        if element_id:
            query = f"""
            MATCH (p:Project) WHERE elementId(p) = $element_id
            SET p.folder_name = $folder_name, p.description = $description, p.start_date = $start_date,
                p.end_date = $end_date, p.status = $status, p.created_date = $created_date, p.modified_date = $modified_date
            RETURN p
            """
        else:
            query = """
            CREATE (p:Project {project_id: $project_id, folder_name: $folder_name, description: $description, start_date: $start_date,
                               end_date: $end_date, status: $status, created_date: $created_date, modified_date: $modified_date})
            RETURN p
            """
        return query


    @staticmethod
    async def retrieve_from_database(session, element_id):
        uri = "bolt://localhost:7689"
        user = "neo4j"
        password = "mynewpassword"
        
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                query = """
                MATCH (p:Project) WHERE elementId(p) = $element_id
                RETURN p.project_id AS project_id, p.folder_name AS folder_name, p.description AS description, p.start_date AS start_date,
                       p.end_date AS end_date, p.status AS status, p.created_date AS created_date, p.modified_date AS modified_date
                """
                result = session.run(query, element_id=element_id)
                record = result.single()
                if record:
                    return Project(record["project_id"], record["folder_name"], record["description"], record["start_date"],
                                   record["end_date"], record["status"], record["created_date"], record["modified_date"])
                return None