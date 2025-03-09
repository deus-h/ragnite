"""
Vector Database Data Migration

This module provides tools for migrating data between different vector databases.
It supports migrating collections, vectors, and metadata while preserving the 
structure and relationships of the data.
"""

import os
import time
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MigrationConfig:
    """
    Configuration for data migration between vector databases.
    """
    batch_size: int = 100
    include_metadata: bool = True
    include_documents: bool = True
    skip_existing: bool = True
    preserve_ids: bool = True
    collection_mapping: Optional[Dict[str, str]] = None
    source_filter: Optional[Dict[str, Any]] = None
    include_collections: Optional[List[str]] = None
    exclude_collections: Optional[List[str]] = None
    transform_vector_fn: Optional[Callable[[List[float]], List[float]]] = None
    transform_metadata_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    progress_bar: bool = True

@dataclass
class MigrationResult:
    """
    Result of a data migration operation.
    """
    success: bool
    source_db: str
    target_db: str
    collections_migrated: List[str]
    vectors_migrated: int
    errors: List[str]
    skipped: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "success": self.success,
            "source_db": self.source_db,
            "target_db": self.target_db,
            "collections_migrated": self.collections_migrated,
            "vectors_migrated": self.vectors_migrated,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration_seconds": self.duration_seconds
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            str: JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def __str__(self) -> str:
        """
        String representation of migration results.
        
        Returns:
            str: Formatted migration results
        """
        lines = [
            f"Migration from {self.source_db} to {self.target_db}:",
            f"  Success: {self.success}",
            f"  Collections migrated: {', '.join(self.collections_migrated) if self.collections_migrated else 'None'}",
            f"  Vectors migrated: {self.vectors_migrated}",
            f"  Skipped: {self.skipped}",
            f"  Duration: {self.duration_seconds:.2f} seconds"
        ]
        
        if self.errors:
            lines.append("  Errors:")
            for error in self.errors:
                lines.append(f"    - {error}")
        
        return "\n".join(lines)

class BaseVectorDBMigrator(ABC):
    """
    Base abstract class for vector database migrators.
    
    Migrators provide tools to migrate data between different vector databases.
    """
    
    def __init__(
        self,
        source_db: Any,
        target_db: Any,
        config: Optional[MigrationConfig] = None
    ):
        """
        Initialize BaseVectorDBMigrator.
        
        Args:
            source_db: Source vector database connector instance
            target_db: Target vector database connector instance
            config: Migration configuration
        """
        self.source_db = source_db
        self.target_db = target_db
        self.config = config or MigrationConfig()
    
    @abstractmethod
    def migrate(
        self,
        collection_names: Optional[List[str]] = None
    ) -> MigrationResult:
        """
        Migrate data from source to target database.
        
        Args:
            collection_names: List of collections to migrate (if None, migrate all)
            
        Returns:
            MigrationResult: Result of the migration
        """
        pass
    
    @abstractmethod
    def migrate_collection(
        self,
        source_collection: str,
        target_collection: Optional[str] = None
    ) -> Tuple[int, int, List[str]]:
        """
        Migrate a single collection from source to target database.
        
        Args:
            source_collection: Name of the source collection
            target_collection: Name of the target collection (if None, use same as source)
            
        Returns:
            Tuple[int, int, List[str]]: Tuple of (vectors migrated, skipped, errors)
        """
        pass
    
    def _check_connections(self):
        """
        Check if both source and target databases are connected.
        
        Raises:
            ConnectionError: If either database is not connected
        """
        if not hasattr(self.source_db, 'is_connected') or not self.source_db.is_connected():
            raise ConnectionError("Source database is not connected.")
        
        if not hasattr(self.target_db, 'is_connected') or not self.target_db.is_connected():
            raise ConnectionError("Target database is not connected.")
    
    def _get_collections_to_migrate(
        self,
        collection_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get the list of collections to migrate based on config and provided names.
        
        Args:
            collection_names: Explicit list of collections to migrate
            
        Returns:
            List[str]: List of collection names to migrate
        """
        if collection_names:
            # Use explicitly provided collection names
            return collection_names
        
        # Get all collections from source database
        all_collections = self.source_db.list_collections()
        
        # Apply include/exclude filters from config
        if self.config.include_collections:
            collections = [c for c in all_collections if c in self.config.include_collections]
        elif self.config.exclude_collections:
            collections = [c for c in all_collections if c not in self.config.exclude_collections]
        else:
            collections = all_collections
        
        return collections
    
    def _get_target_collection_name(self, source_collection: str) -> str:
        """
        Get the target collection name based on source collection and mapping.
        
        Args:
            source_collection: Source collection name
            
        Returns:
            str: Target collection name
        """
        if self.config.collection_mapping and source_collection in self.config.collection_mapping:
            return self.config.collection_mapping[source_collection]
        
        return source_collection

class VectorDBMigrator(BaseVectorDBMigrator):
    """
    Migrator for transferring data between any two vector databases.
    """
    
    def migrate(
        self,
        collection_names: Optional[List[str]] = None
    ) -> MigrationResult:
        """
        Migrate data from source to target database.
        
        Args:
            collection_names: List of collections to migrate (if None, migrate all)
            
        Returns:
            MigrationResult: Result of the migration
        """
        self._check_connections()
        
        start_time = time.time()
        total_migrated = 0
        total_skipped = 0
        all_errors = []
        collections_migrated = []
        
        try:
            # Get collections to migrate
            collections = self._get_collections_to_migrate(collection_names)
            
            if not collections:
                logger.warning("No collections found to migrate.")
                return MigrationResult(
                    success=True,
                    source_db=self.source_db.__class__.__name__,
                    target_db=self.target_db.__class__.__name__,
                    collections_migrated=[],
                    vectors_migrated=0,
                    errors=["No collections found to migrate"],
                    skipped=0,
                    duration_seconds=time.time() - start_time
                )
            
            # Migrate each collection
            for source_collection in collections:
                target_collection = self._get_target_collection_name(source_collection)
                
                logger.info(f"Migrating collection '{source_collection}' to '{target_collection}'")
                
                try:
                    migrated, skipped, errors = self.migrate_collection(source_collection, target_collection)
                    
                    total_migrated += migrated
                    total_skipped += skipped
                    all_errors.extend(errors)
                    
                    if migrated > 0:
                        collections_migrated.append(source_collection)
                    
                    logger.info(f"Collection '{source_collection}' migration complete: "
                              f"{migrated} vectors migrated, {skipped} skipped, {len(errors)} errors")
                
                except Exception as e:
                    logger.error(f"Error migrating collection '{source_collection}': {str(e)}")
                    all_errors.append(f"Collection '{source_collection}': {str(e)}")
            
            # Determine overall success
            success = total_migrated > 0 and len(all_errors) == 0
            
            return MigrationResult(
                success=success,
                source_db=self.source_db.__class__.__name__,
                target_db=self.target_db.__class__.__name__,
                collections_migrated=collections_migrated,
                vectors_migrated=total_migrated,
                errors=all_errors,
                skipped=total_skipped,
                duration_seconds=time.time() - start_time
            )
        
        except Exception as e:
            logger.error(f"Error during migration: {str(e)}")
            return MigrationResult(
                success=False,
                source_db=self.source_db.__class__.__name__,
                target_db=self.target_db.__class__.__name__,
                collections_migrated=collections_migrated,
                vectors_migrated=total_migrated,
                errors=[f"Migration error: {str(e)}"],
                skipped=total_skipped,
                duration_seconds=time.time() - start_time
            )
    
    def migrate_collection(
        self,
        source_collection: str,
        target_collection: Optional[str] = None
    ) -> Tuple[int, int, List[str]]:
        """
        Migrate a single collection from source to target database.
        
        Args:
            source_collection: Name of the source collection
            target_collection: Name of the target collection (if None, use same as source)
            
        Returns:
            Tuple[int, int, List[str]]: Tuple of (vectors migrated, skipped, errors)
        """
        self._check_connections()
        
        # Set target collection name if not provided
        if target_collection is None:
            target_collection = source_collection
        
        errors = []
        migrated_count = 0
        skipped_count = 0
        
        try:
            # Check if source collection exists
            if not self.source_db.collection_exists(source_collection):
                raise ValueError(f"Source collection '{source_collection}' does not exist.")
            
            # Get source collection info
            source_info = self.source_db.get_collection_info(source_collection)
            
            # Check if target collection exists and create if not
            if not self.target_db.collection_exists(target_collection):
                # Extract dimension and other info from source
                dimension = source_info.get("dimension")
                if not dimension:
                    raise ValueError(f"Could not determine dimension for collection '{source_collection}'")
                
                # Get distance metric if available
                distance_metric = source_info.get("distance_metric", "cosine")
                
                # Create target collection
                self.target_db.create_collection(
                    collection_name=target_collection,
                    dimension=dimension,
                    distance_metric=distance_metric,
                    metadata={"migrated_from": source_collection}
                )
                logger.info(f"Created target collection '{target_collection}' with dimension {dimension}")
            else:
                # Verify target collection compatibility
                target_info = self.target_db.get_collection_info(target_collection)
                
                source_dim = source_info.get("dimension")
                target_dim = target_info.get("dimension")
                
                if source_dim and target_dim and source_dim != target_dim:
                    # Dimensions don't match - we'll need to transform vectors
                    if not self.config.transform_vector_fn:
                        raise ValueError(f"Source dimension ({source_dim}) does not match "
                                      f"target dimension ({target_dim}). "
                                      f"Provide a transform_vector_fn in the configuration.")
                    logger.warning(f"Source dimension ({source_dim}) does not match "
                                 f"target dimension ({target_dim}). "
                                 f"Will use provided transform_vector_fn.")
            
            # Get all vectors from source collection
            # Note: This is a simplified approach that loads all vectors in memory
            # For large collections, we would need to implement pagination
            
            # For now, let's assume there's a way to get all vectors from the collection
            # This would depend on the specific database connector implementations
            
            # Here's how we would implement it:
            
            # 1. Get collection count
            count = self.source_db.count_vectors(source_collection, filter=self.config.source_filter)
            
            if count == 0:
                logger.warning(f"Source collection '{source_collection}' is empty.")
                return migrated_count, skipped_count, errors
            
            # 2. Process in batches
            batch_size = self.config.batch_size
            num_batches = (count + batch_size - 1) // batch_size
            
            # Set up progress bar if requested
            pbar = None
            if self.config.progress_bar:
                try:
                    pbar = tqdm(total=count, desc=f"Migrating {source_collection}")
                except:
                    logger.warning("tqdm not available, not showing progress bar")
            
            # Get access to the underlying batch processing if available
            if hasattr(self.source_db, "get_collection_vectors_batch"):
                # Process using batched access
                offset = 0
                while offset < count:
                    try:
                        # Get batch of vectors
                        batch_vectors, batch_ids, batch_metadata, batch_documents = self.source_db.get_collection_vectors_batch(
                            collection_name=source_collection,
                            offset=offset,
                            limit=batch_size,
                            filter=self.config.source_filter
                        )
                        
                        # Apply vector transformation if needed
                        if self.config.transform_vector_fn:
                            batch_vectors = [self.config.transform_vector_fn(v) for v in batch_vectors]
                        
                        # Apply metadata transformation if needed
                        if self.config.transform_metadata_fn and batch_metadata:
                            batch_metadata = [
                                self.config.transform_metadata_fn(m) if m else m 
                                for m in batch_metadata
                            ]
                        
                        # Handle ID mapping
                        if not self.config.preserve_ids:
                            # If not preserving IDs, we'll replace them with new ones
                            batch_ids = [f"migrated_{i}" for i in range(offset, offset + len(batch_vectors))]
                        
                        # Add vectors to target collection
                        if batch_vectors:
                            try:
                                self.target_db.add_vectors(
                                    collection_name=target_collection,
                                    vectors=batch_vectors,
                                    ids=batch_ids,
                                    metadata=batch_metadata if self.config.include_metadata else None,
                                    documents=batch_documents if self.config.include_documents else None
                                )
                                migrated_count += len(batch_vectors)
                            except Exception as e:
                                error_msg = f"Error adding batch starting at offset {offset}: {str(e)}"
                                logger.error(error_msg)
                                errors.append(error_msg)
                        
                        # Update progress bar
                        if pbar:
                            pbar.update(len(batch_vectors))
                        
                        # Move to next batch
                        offset += batch_size
                    
                    except Exception as e:
                        error_msg = f"Error processing batch at offset {offset}: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        offset += batch_size  # Still advance to avoid infinite loop
            else:
                # Fallback to one-by-one processing
                # This is much slower but works with any connector
                logger.warning(f"Batch vector retrieval not supported. Falling back to one-by-one processing.")
                
                # Get all vector IDs
                # Note: This is a simplified approach and may not work with all connectors
                # For real implementation, we would need a way to get all vector IDs
                
                if hasattr(self.source_db, "get_all_vector_ids"):
                    all_ids = self.source_db.get_all_vector_ids(source_collection, filter=self.config.source_filter)
                else:
                    logger.error("Neither get_collection_vectors_batch nor get_all_vector_ids is supported by the source database.")
                    return migrated_count, skipped_count, errors
                
                # Process in batches
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i+batch_size]
                    
                    batch_vectors = []
                    batch_metadata = []
                    
                    # Process each ID
                    for id_str in batch_ids:
                        try:
                            # Get vector and metadata
                            vector, metadata = self.source_db.get_vector(source_collection, id_str)
                            
                            # Apply vector transformation if needed
                            if self.config.transform_vector_fn:
                                vector = self.config.transform_vector_fn(vector)
                            
                            # Apply metadata transformation if needed
                            if self.config.transform_metadata_fn:
                                metadata = self.config.transform_metadata_fn(metadata)
                            
                            # Extract document if included in metadata
                            document = metadata.pop("document", None) if metadata else None
                            
                            # Add to batch
                            batch_vectors.append(vector)
                            batch_metadata.append(metadata)
                            
                            # Handle document
                            if document:
                                if "documents" not in locals():
                                    documents = []
                                documents.append(document)
                            
                        except Exception as e:
                            error_msg = f"Error retrieving vector {id_str}: {str(e)}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                    
                    # Add batch to target
                    if batch_vectors:
                        try:
                            # Handle ID mapping
                            if not self.config.preserve_ids:
                                # If not preserving IDs, replace with new ones
                                target_ids = [f"migrated_{j}" for j in range(i, i + len(batch_vectors))]
                            else:
                                target_ids = batch_ids
                            
                            self.target_db.add_vectors(
                                collection_name=target_collection,
                                vectors=batch_vectors,
                                ids=target_ids,
                                metadata=batch_metadata if self.config.include_metadata else None,
                                documents=documents if self.config.include_documents and "documents" in locals() else None
                            )
                            migrated_count += len(batch_vectors)
                        except Exception as e:
                            error_msg = f"Error adding batch starting at index {i}: {str(e)}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                    
                    # Update progress bar
                    if pbar:
                        pbar.update(len(batch_ids))
            
            # Close progress bar
            if pbar:
                pbar.close()
            
            return migrated_count, skipped_count, errors
        
        except Exception as e:
            logger.error(f"Error migrating collection '{source_collection}': {str(e)}")
            errors.append(f"Collection migration error: {str(e)}")
            
            # Close progress bar if open
            if 'pbar' in locals() and pbar:
                pbar.close()
            
            return migrated_count, skipped_count, errors

def get_migrator(
    source_db: Any,
    target_db: Any,
    config: Optional[MigrationConfig] = None
) -> BaseVectorDBMigrator:
    """
    Get a migrator for transferring data between vector databases.
    
    Args:
        source_db: Source vector database connector instance
        target_db: Target vector database connector instance
        config: Migration configuration
        
    Returns:
        BaseVectorDBMigrator: Vector database migrator
    """
    return VectorDBMigrator(source_db, target_db, config)

def migrate_data(
    source_db: Any,
    target_db: Any,
    collection_names: Optional[List[str]] = None,
    config: Optional[MigrationConfig] = None,
) -> MigrationResult:
    """
    Migrate data between vector databases.
    
    Args:
        source_db: Source vector database connector instance
        target_db: Target vector database connector instance
        collection_names: List of collections to migrate (if None, migrate all)
        config: Migration configuration
        
    Returns:
        MigrationResult: Result of the migration
    """
    migrator = get_migrator(source_db, target_db, config)
    return migrator.migrate(collection_names) 