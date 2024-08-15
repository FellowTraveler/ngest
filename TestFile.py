import unittest
import os
from neo4j import GraphDatabase
from File import File

class TestFile(unittest.TestCase):
    def setUp(self):
        self.uri = "bolt://localhost:7689"
        self.user = "neo4j"
        self.password = "mynewpassword"
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

    def tearDown(self):
        self.driver.close()

    def test_create_file(self):
        file_path = os.path.join(self.test_dir, "test_file.txt")
        with open(file_path, "w") as f:
            f.write("Test content")
        
        file_size = os.path.getsize(file_path)
        file = File(filename="test_file.txt", full_path=file_path, size_in_bytes=file_size)
        query = file.generate_cypher_query()
        result = file.execute_cypher_query(query)
        self.assertTrue(result[1])  # Check if creation was successful
        self.assertIsNotNone(result[0])  # Check if elementId is returned

        os.remove(file_path)  # Clean up the test file

    def test_retrieve_file(self):
        # First, create a file
        file_path = os.path.join(self.test_dir, "retrieve_test.txt")
        with open(file_path, "w") as f:
            f.write("Retrieve test content")
        
        file_size = os.path.getsize(file_path)
        file = File(filename="retrieve_test.txt", full_path=file_path, size_in_bytes=file_size)
        query = file.generate_cypher_query()
        result = file.execute_cypher_query(query)
        element_id = result[0]

        # Now, retrieve the file
        retrieved_file = File.retrieve_from_database(element_id)
        self.assertIsNotNone(retrieved_file)
        self.assertEqual(retrieved_file.filename, "retrieve_test.txt")
        self.assertEqual(retrieved_file.full_path, file_path)
        self.assertEqual(retrieved_file.size_in_bytes, file_size)

        os.remove(file_path)  # Clean up the test file

    def test_update_file(self):
        # First, create a file
        file_path = os.path.join(self.test_dir, "update_test.txt")
        with open(file_path, "w") as f:
            f.write("Initial content")
        
        file_size = os.path.getsize(file_path)
        file = File(filename="update_test.txt", full_path=file_path, size_in_bytes=file_size)
        query = file.generate_cypher_query()
        result = file.execute_cypher_query(query)
        element_id = result[0]

        # Update the file content
        with open(file_path, "w") as f:
            f.write("Updated content")
        
        updated_size = os.path.getsize(file_path)
        updated_file = File(filename="updated_test.txt", full_path=file_path, size_in_bytes=updated_size)
        query = updated_file.generate_cypher_query(element_id)
        result = updated_file.execute_cypher_query(query, element_id)
        self.assertTrue(result[1])  # Check if update was successful

        # Retrieve and verify the update
        retrieved_file = File.retrieve_from_database(element_id)
        self.assertEqual(retrieved_file.filename, "updated_test.txt")
        self.assertEqual(retrieved_file.full_path, file_path)
        self.assertEqual(retrieved_file.size_in_bytes, updated_size)

        os.remove(file_path)  # Clean up the test file

if __name__ == '__main__':
    unittest.main()
