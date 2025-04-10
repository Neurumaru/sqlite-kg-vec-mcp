import sqlite3

import numpy as np
import pytest

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('vector_database.db')
cursor = conn.cursor()

# Create a table for storing vectors
cursor.execute('''
CREATE TABLE IF NOT EXISTS vectors (
    id INTEGER PRIMARY KEY,
    vector BLOB
)
''')

# Function to convert numpy array to binary
def array_to_binary(array):
    return array.tobytes()

# Function to convert binary to numpy array
def binary_to_array(binary, dtype=np.float32, shape=(-1,)):
    return np.frombuffer(binary, dtype=dtype).reshape(shape)

# Insert a vector into the database
def insert_vector(vector):
    binary_vector = array_to_binary(vector)
    cursor.execute('INSERT INTO vectors (vector) VALUES (?)', (binary_vector,))
    conn.commit()

# Retrieve and print all vectors from the database
def retrieve_vectors():
    cursor.execute('SELECT * FROM vectors')
    rows = cursor.fetchall()
    return [binary_to_array(row[1]) for row in rows]

# Test functions
@pytest.fixture(scope='module')
def setup_database():
    # Setup code: clear the table before tests
    cursor.execute('DELETE FROM vectors')
    conn.commit()
    yield
    # Teardown code: close the connection after tests
    conn.close()

@pytest.mark.usefixtures('setup_database')
def test_insert_and_retrieve_vector():
    # Create a random vector
    vector = np.random.rand(10).astype(np.float32)
    
    # Insert the vector into the database
    insert_vector(vector)
    
    # Retrieve vectors and check
    retrieved_vectors = retrieve_vectors()
    assert len(retrieved_vectors) == 1
    assert np.allclose(retrieved_vectors[0], vector)
