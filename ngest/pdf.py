# Copyright 2024 Chris Odom
# MIT License

from ngest.document import Document
from neo4j.exceptions import Neo4jError

class PDF(Document):
    def __init__(self, filename, full_path, size_in_bytes, project_id, created_date=None, modified_date=None, 
                 content_type="application/pdf", page_count=None, author=None, title=None):
        super().__init__(filename, full_path, size_in_bytes, project_id, created_date, modified_date, "pdf", content_type)
        self.page_count = page_count
        self.author = author
        self.title = title

    @staticmethod
    async def create_in_database(session, project_id, full_path, page_count=None, author=None, title=None):
        document = await Document.retrieve_from_database(session, project_id, full_path)
        if not document:
            return None

        file_id = f"{project_id}/{full_path}"
        query = """
        MATCH (d:Document {id: $file_id})
        SET d:PDF
        SET d.page_count = $page_count,
            d.author = $author,
            d.title = $title
        RETURN d
        """
        try:
            result = await session.run(query,
                file_id=file_id,
                page_count=page_count,
                author=author,
                title=title
            )
            record = await result.single()
            if record:
                node = record["d"]
                return PDF(
                    filename=node["filename"],
                    full_path=node["full_path"],
                    size_in_bytes=node["size_in_bytes"],
                    project_id=node["project_id"],
                    created_date=node["created_date"],
                    modified_date=node["modified_date"],
                    content_type=node["content_type"],
                    page_count=node["page_count"],
                    author=node["author"],
                    title=node["title"]
                )
            return None
        except Neo4jError as e:
            print(f"Error creating PDF in database: {e}")
            return None

    @staticmethod
    async def retrieve_from_database(session, project_id, full_path):
        document = await Document.retrieve_from_database(session, project_id, full_path)
        if not document:
            return None

        query = """
        MATCH (p:PDF {id: $id})
        RETURN p.page_count AS page_count, p.author AS author, p.title AS title
        """
        result = await session.run(query, id=document.id)
        record = await result.single()
        if record:
            return PDF(
                document.filename, document.full_path, document.size_in_bytes,
                document.project_id, document.created_date, document.modified_date, document.content_type,
                record["page_count"], record["author"], record["title"]
            )
        return None
