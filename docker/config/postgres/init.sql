-- Initialize PostgreSQL with pgvector extension

-- Create extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for vector database
CREATE SCHEMA IF NOT EXISTS vector_store;

-- Create collections table to track collections
CREATE TABLE IF NOT EXISTS vector_store.collections (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    dimension INTEGER NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Function to create a new collection table
CREATE OR REPLACE FUNCTION vector_store.create_collection(
    collection_name TEXT,
    dimension INTEGER,
    metadata JSONB DEFAULT NULL
) RETURNS VOID AS $$
DECLARE
    table_name TEXT;
    collection_exists BOOLEAN;
BEGIN
    -- Check if collection exists
    SELECT EXISTS(
        SELECT 1 FROM vector_store.collections WHERE name = collection_name
    ) INTO collection_exists;
    
    IF collection_exists THEN
        RAISE EXCEPTION 'Collection % already exists', collection_name;
    END IF;
    
    -- Create collection record
    INSERT INTO vector_store.collections (name, dimension, metadata)
    VALUES (collection_name, dimension, metadata);
    
    -- Create table for collection vectors
    table_name := 'vector_store.' || collection_name;
    
    EXECUTE format('
        CREATE TABLE %s (
            id TEXT PRIMARY KEY,
            vector vector(%s) NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )', table_name, dimension);
    
    -- Create index for vector similarity search
    EXECUTE format('
        CREATE INDEX %s_vector_idx ON %s USING ivfflat (vector vector_l2_ops)
        WITH (lists = 100)', collection_name, table_name);
    
END;
$$ LANGUAGE plpgsql;

-- Function to drop a collection
CREATE OR REPLACE FUNCTION vector_store.drop_collection(
    collection_name TEXT
) RETURNS VOID AS $$
DECLARE
    table_name TEXT;
    collection_exists BOOLEAN;
BEGIN
    -- Check if collection exists
    SELECT EXISTS(
        SELECT 1 FROM vector_store.collections WHERE name = collection_name
    ) INTO collection_exists;
    
    IF NOT collection_exists THEN
        RAISE EXCEPTION 'Collection % does not exist', collection_name;
    END IF;
    
    -- Drop collection table
    table_name := 'vector_store.' || collection_name;
    EXECUTE format('DROP TABLE IF EXISTS %s', table_name);
    
    -- Remove collection record
    DELETE FROM vector_store.collections WHERE name = collection_name;
END;
$$ LANGUAGE plpgsql;

-- Create a test collection
SELECT vector_store.create_collection('test_collection', 384, '{"description": "Test collection for RAG"}'::jsonb); 