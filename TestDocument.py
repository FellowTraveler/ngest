import unittest
import os
from neo4j import GraphDatabase
from Document import Document

class TestDocument(unittest.TestCase):
    def setUp(self):
        self.uri = "bolt://localhost:7689"
        self.user = "neo4j"
        self.password = "mynewpassword"
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

    def tearDown(self):
        self.driver.close()

    def test_create_document(self):
        file_path = os.path.join(self.test_dir, "test_document.txt")
        with open(file_path, "w") as f:
            f.write("Test content")
        
        file_size = os.path.getsize(file_path)
        document = Document(filename="test_document.txt", full_path=file_path, size_in_bytes=file_size,
                            content_type="text/plain", source_id="test_source")
        query = document.generate_cypher_query()
        result = document.execute_cypher_query(query)
        self.assertTrue(result[1])  # Check if creation was successful
        self.assertIsNotNone(result[0])  # Check if elementId is returned

        os.remove(file_path)  # Clean up the test file

    def test_retrieve_document(self):
        # First, create a document
        file_path = os.path.join(self.test_dir, "retrieve_test_doc.txt")
        with open(file_path, "w") as f:
            f.write("Retrieve test content")
        
        file_size = os.path.getsize(file_path)
        document = Document(filename="retrieve_test_doc.txt", full_path=file_path, size_in_bytes=file_size,
                            content_type="text/plain", source_id="retrieve_source")
        query = document.generate_cypher_query()
        result = document.execute_cypher_query(query)
        element_id = result[0]

        # Now, retrieve the document
        retrieved_document = Document.retrieve_from_database(element_id)
        self.assertIsNotNone(retrieved_document)
        self.assertEqual(retrieved_document.filename, "retrieve_test_doc.txt")
        self.assertEqual(retrieved_document.full_path, file_path)
        self.assertEqual(retrieved_document.size_in_bytes, file_size)
        self.assertEqual(retrieved_document.content_type, "text/plain")
        self.assertEqual(retrieved_document.source_id, "retrieve_source")

        os.remove(file_path)  # Clean up the test file

    def test_update_document(self):
        # First, create a document
        file_path = os.path.join(self.test_dir, "update_test_doc.txt")
        with open(file_path, "w") as f:
            f.write("Initial content")
        
        file_size = os.path.getsize(file_path)
        document = Document(filename="update_test_doc.txt", full_path=file_path, size_in_bytes=file_size,
                            content_type="text/plain", source_id="initial_source")
        query = document.generate_cypher_query()
        result = document.execute_cypher_query(query)
        element_id = result[0]

        # Update the document content and properties
        with open(file_path, "w") as f:
            f.write("Updated content")
        
        updated_size = os.path.getsize(file_path)
        updated_document = Document(filename="updated_test_doc.txt", full_path=file_path, size_in_bytes=updated_size,
                                    content_type="text/plain", source_id="updated_source")
        query = updated_document.generate_cypher_query(element_id)
        result = updated_document.execute_cypher_query(query, element_id)
        self.assertTrue(result[1])  # Check if update was successful

        # Retrieve and verify the update
        retrieved_document = Document.retrieve_from_database(element_id)
        self.assertEqual(retrieved_document.filename, "updated_test_doc.txt")
        self.assertEqual(retrieved_document.full_path, file_path)
        self.assertEqual(retrieved_document.size_in_bytes, updated_size)
        self.assertEqual(retrieved_document.content_type, "text/plain")
        self.assertEqual(retrieved_document.source_id, "updated_source")

        os.remove(file_path)  # Clean up the test file

if __name__ == '__main__':
    unittest.main()
