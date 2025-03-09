## Retrieval

### Hybrid Searchers
- ✓ Base hybrid searcher interface
- ✓ VectorKeywordHybridSearcher for combining vector and keyword search
- ✓ BM25VectorHybridSearcher for combining BM25 and vector search
- ✓ MultiIndexHybridSearcher for searching across multiple indices
- ✓ WeightedHybridSearcher for combining multiple search strategies with automatic weight tuning
- ✓ Factory function for creating hybrid searchers

## Re-Rankers
- ✓ Base re-ranker interface
- ✓ Keyword re-ranker
- ✓ Semantic re-ranker
- ✓ Hybrid re-ranker
- ✓ VectorKeywordHybridSearcher for combining vector and keyword search
- ✓ BM25VectorHybridSearcher for combining BM25 and vector search
- ✓ MultiIndexHybridSearcher for searching across multiple indices
- ✓ WeightedHybridSearcher for combining multiple search strategies with automatic weight tuning
- ✓ Factory function for creating hybrid searchers

### Filter Builders
- ✓ Base filter builder interface
- ✓ MetadataFilterBuilder for building filters based on metadata
- ✓ DateFilterBuilder for building date-based filters
- ✓ NumericFilterBuilder for building numeric filters
- ✓ CompositeFilterBuilder for building complex filters
- ✓ Factory function for creating filter builders
- ✓ Example scripts for all filter builders 