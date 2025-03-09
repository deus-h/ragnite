# Docker-Based Testing Infrastructure for Vector Databases

This directory contains Docker Compose configuration and scripts for setting up and testing vector databases for the RAG Research project.

## Overview

The setup includes the following vector databases:

1. **ChromaDB**: A simple, in-memory/file-based vector database with a Python-native API
2. **PostgreSQL with pgvector**: SQL-based vector database leveraging the power of PostgreSQL
3. **Qdrant**: Scalable vector search engine with advanced filtering capabilities

## Directory Structure

```
docker/
├── docker-compose.yml     # Main Docker Compose configuration
├── .env                   # Environment variables for configuration
├── config/                # Configuration files for databases
│   ├── chromadb/          # ChromaDB configuration
│   ├── postgres/          # PostgreSQL configuration
│   └── qdrant/            # Qdrant configuration
└── scripts/               # Utility scripts for testing
    ├── setup_test_data.py # Script to populate test data
    └── run_benchmarks.py  # Script to benchmark performance
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ with required packages:
  - `chromadb`
  - `psycopg2-binary`
  - `qdrant-client`
  - `matplotlib` (for plotting benchmark results)

### Starting the Databases

You can start specific databases using profiles:

```bash
# Start only ChromaDB
docker-compose --profile chromadb up -d

# Start only PostgreSQL with pgvector
docker-compose --profile postgres up -d

# Start only Qdrant
docker-compose --profile qdrant up -d

# Start all databases
docker-compose --profile all up -d
```

### Stopping the Databases

```bash
# Stop all databases
docker-compose down
```

### Loading Test Data

Run the setup script to populate the databases with test data:

```bash
# Load data into all databases
python scripts/setup_test_data.py

# Load data into a specific database
python scripts/setup_test_data.py --db chromadb
```

### Running Benchmarks

Run the benchmark script to test database performance:

```bash
# Benchmark all databases
python scripts/run_benchmarks.py

# Benchmark a specific database
python scripts/run_benchmarks.py --db chromadb

# Generate a plot of the results
python scripts/run_benchmarks.py --plot
```

Benchmark results will be saved to the `results/` directory.

## Configuration

You can customize the database settings by modifying the `.env` file:

```
# ChromaDB Settings
CHROMADB_PORT=8000

# PostgreSQL Settings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=vectordb
POSTGRES_PORT=5432

# Qdrant Settings
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
```

## Using with the RAG Utility Tools

The Docker environment is designed to work seamlessly with the Vector Database connectors in the RAG Utility Tools:

```python
from tools.src.vector_db import get_database_connector

# Connect to ChromaDB
connector = get_database_connector("chromadb", host="localhost", port=8000)
connector.connect()

# Create a collection
collection = connector.create_collection("my_collection", dimension=384)

# Add vectors to the collection
connector.add_vectors(
    "my_collection",
    vectors=[[0.1, 0.2, 0.3] * 128] * 10,
    ids=[f"id_{i}" for i in range(10)]
)

# Search for similar vectors
results = connector.search(
    "my_collection",
    query_vector=[0.1, 0.2, 0.3] * 128,
    top_k=5
)
```

## Troubleshooting

If you encounter issues with the Docker environment:

1. **Database connection issues**:
   - Make sure the corresponding database is running: `docker ps`
   - Check the logs: `docker-compose logs chromadb`

2. **Permission issues**:
   - Make sure the data directories have appropriate permissions

3. **Port conflicts**:
   - Edit the `.env` file to change the port mappings if needed 