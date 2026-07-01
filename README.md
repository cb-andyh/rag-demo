## RAG Demo using Couchbase, Streamlit, LangChain, and OpenAI

This is a demo app built to chat with your custom PDFs using the vector search capabilities of Couchbase to augment the OpenAI results in a Retrieval-Augmented-Generation (RAG) model.

The demo uses `CouchbaseQueryVectorStore` (`chat_with_pdf_query.py`), which leverages Couchbase Vector Search (Hyperscale/Composite Vector Indexes) with the Query and Indexing Services.

The demo also uses a **question-level semantic cache** backed by Couchbase to avoid repeated LLM calls for semantically similar questions — saving time and cost. On a cache hit, the previous RAG response is returned and streamed directly without calling OpenAI.

### How does it work?

You can upload your PDFs with custom data & ask questions about the data in the chat box.

For each question, you will get two answers:

- one using RAG (Couchbase logo)
- one using pure LLM - OpenAI (🤖).

For RAG, we are using LangChain, Couchbase Vector Search & OpenAI. We fetch parts of the PDF relevant to the question using Vector search & add it as the context to the LLM. The LLM is instructed to answer based on the context from the Vector Store.

**Semantic Cache:** RAG responses are cached at the question level. The user's question is embedded and compared against previously cached questions using Couchbase FTS vector search. If a semantically similar question is found (score ≥ 0.9), the cached response is returned immediately. Otherwise, the RAG chain is invoked and the response is stored for future reuse.

> Note: The streaming of cached responses is purely for visual experience — the response is streamed character-by-character from the local string without calling OpenAI.



## Setup Instructions

### Install dependencies

  `pip install -r requirements.txt`

### Set the environment secrets

Copy the `secrets.example.toml` file in `.streamlit` folder and rename it to `secrets.toml` and replace the placeholders with the actual values for your environment.

```toml
OPENAI_API_KEY = "<open_ai_api_key>"
DB_CONN_STR = "<connection_string_for_couchbase_cluster>"
DB_USERNAME = "<username_for_couchbase_cluster>"
DB_PASSWORD = "<password_for_couchbase_cluster>"
DB_BUCKET = "<name_of_bucket_to_store_documents>"
DB_SCOPE = "<name_of_scope_to_store_documents>"
DB_COLLECTION = "<name_of_collection_to_store_documents>"
CACHE_COLLECTION = "<name_of_collection_to_cache_llm_responses>"
CACHE_INDEX = "<name_of_fts_index_for_semantic_cache>"
AUTH_ENABLED = "False"
LOGIN_PASSWORD = "<password_to_access_the_streamlit_app>"
```



## Couchbase Bucket, Scope, and Collection Setup

### Bucket, Scope, and Collection Structure

This demo organizes data using one bucket, one scope, and two collections:

```
pdf-docs (bucket)
└── shared (scope)
    ├── docs                    (collection — PDF chunks + embeddings)
    └── cached_llm_responses    (collection — cached question/response pairs)
```

- **`docs`** — Each document is a chunk of an uploaded PDF, stored by `CouchbaseQueryVectorStore` with its text, embedding vector, and metadata (e.g. source file, page number).
- **`cached_llm_responses`** — Each document is a cached question/answer pair used by the semantic cache, with the following structure:

  ```json
  {
    "text": "<original question>",
    "embedding": [/* 1536-dimensional OpenAI embedding vector */],
    "response": "<RAG response string>"
  }
  ```

The bucket, scope, and collection names above are just the defaults used in this demo — you can name them whatever you like, as long as the names match the `DB_BUCKET`, `DB_SCOPE`, `DB_COLLECTION`, and `CACHE_COLLECTION` values in your `secrets.toml`.

<!-- SCREENSHOT: Couchbase UI showing the "pdf-docs" bucket with the "shared" scope and its "docs" / "cached_llm_responses" collections -->

### Creating the Bucket, Scope, and Collections

**Couchbase Capella:**

1. Open your cluster and go to the **Buckets** tab → **Create Bucket** → name it `pdf-docs` (set the memory quota as needed) → **Create**.
2. Click into the `pdf-docs` bucket → **Scopes & Collections** → **Add Scope** → name it `shared`.
3. Within the `shared` scope, **Add Collection** → name it `docs`.
4. **Add Collection** again → name it `cached_llm_responses`.

**Couchbase Server (self-managed):**

1. Open the Couchbase Web Console → **Buckets** → **Add Bucket** → name it `pdf-docs` (set the memory quota as needed) → **Add Bucket**.
2. Go to the bucket → **Scopes & Collections** → **Add Scope** → name it `shared`.
3. Within the `shared` scope, **Add Collection** → name it `docs`.
4. **Add Collection** again → name it `cached_llm_responses`.

<!-- SCREENSHOT: "Create Bucket" dialog -->
<!-- SCREENSHOT: "Add Scope" dialog showing "shared" -->
<!-- SCREENSHOT: "Add Collection" dialog showing "docs" and "cached_llm_responses" -->

### Example Configuration Values

Using the structure above, a filled-in `secrets.toml` would look like this:

```toml
OPENAI_API_KEY = "sk-<your-openai-api-key>"
DB_CONN_STR = "couchbases://cb.xxxxxxxx.cloud.couchbase.com"
DB_USERNAME = "rag-demo-user"
DB_PASSWORD = "<your-couchbase-password>"
DB_BUCKET = "pdf-docs"
DB_SCOPE = "shared"
DB_COLLECTION = "docs"
CACHE_COLLECTION = "cached_llm_responses"
CACHE_INDEX = "cached_llm_responses_index"
AUTH_ENABLED = "False"
LOGIN_PASSWORD = "changeme"
```

`CACHE_INDEX` (`cached_llm_responses_index` above) refers to the FTS index created in [Create the Semantic Cache FTS Index](#create-the-semantic-cache-fts-index) — it is a name you choose when you create that index, not a collection.



## Couchbase Vector Search (Hyperscale/Composite)

For the full tutorial on Couchbase Vector Search approach, please visit [Developer Portal - Couchbase Vector Search](https://developer.couchbase.com/tutorial-python-langchain-pdf-chat-query).

### Prerequisites
- Couchbase Server 8.0+ or Couchbase Capella

This approach uses `CouchbaseQueryVectorStore` which leverages Couchbase's Hyperscale and Composite Vector Indexes (built on Global Secondary Index infrastructure). The vector search is performed using SQL++ queries with cosine similarity distance metric.

### Understanding Vector Index Types

Couchbase offers different types of vector indexes for Couchbase Vector Search:

**Hyperscale Vector Indexes (BHIVE)**
- Best for pure vector searches - content discovery, recommendations, semantic search
- High performance with low memory footprint - designed to scale to billions of vectors
- Optimized for concurrent operations - supports simultaneous searches and inserts
- Use when: You primarily perform pure vector-only queries without complex scalar filtering
- Ideal for: Large-scale semantic search, recommendation systems, content discovery

**Composite Vector Indexes**
- Best for filtered vector searches - combines vector search with scalar value filtering
- Efficient pre-filtering - scalar attributes reduce the vector comparison scope
- Use when: Your queries combine vector similarity with scalar filters that eliminate large portions of data
- Ideal for: Compliance-based filtering, user-specific searches, time-bounded queries

**Choosing the Right Index Type**
- Start with Hyperscale Vector Index for pure vector searches and large datasets
- Use Composite Vector Index when scalar filters significantly reduce your search space
- Consider your dataset size: Hyperscale scales to billions, Composite works well for tens of millions to billions

For more details, see the [Couchbase Vector Index documentation](https://docs.couchbase.com/server/current/vector-index/use-vector-indexes.html).

### Index Configuration (Optional)

While the application works without creating indexes manually, you can optionally create a vector index for better performance.

> **Important:** The vector index should be created **after** ingesting the documents (uploading PDFs).

**Using LangChain:**

You can create the index programmatically after uploading your PDFs:

```python
from langchain_couchbase.vectorstores import IndexType

# Create a vector index on the collection
vector_store.create_index(
    index_name="idx_vector",
    dimension=1536,
    similarity="cosine",
    index_type=IndexType.BHIVE,  # or IndexType.COMPOSITE
    index_description="IVF,SQ8"
)
```

For more details on the `create_index()` method, see the [LangChain Couchbase API documentation](https://couchbase-ecosystem.github.io/langchain-couchbase/langchain_couchbase.html#langchain_couchbase.vectorstores.query_vector_store.CouchbaseQueryVectorStore.create_index).

**Understanding Index Configuration Parameters:**

The `description` parameter controls how Couchbase optimizes vector storage and search performance:

**Format:** `'IVF[<centroids>],{PQ|SQ}<settings>'`

**Centroids (IVF - Inverted File):**
- Controls how the dataset is subdivided for faster searches
- More centroids = faster search, slower training
- Fewer centroids = slower search, faster training
- If omitted (like `IVF,SQ8`), Couchbase auto-selects based on dataset size

**Quantization Options:**
- **SQ (Scalar Quantization)**: `SQ4`, `SQ6`, `SQ8` (4, 6, or 8 bits per dimension)
- **PQ (Product Quantization)**: `PQ<subquantizers>x<bits>` (e.g., `PQ32x8`)
- Higher values = better accuracy, larger index size

**Common Examples:**
- `IVF,SQ8` - Auto centroids, 8-bit scalar quantization (good default)
- `IVF1000,SQ6` - 1000 centroids, 6-bit scalar quantization
- `IVF,PQ32x8` - Auto centroids, 32 subquantizers with 8 bits

For detailed configuration options, see the [Quantization & Centroid Settings](https://docs.couchbase.com/server/current/vector-index/hyperscale-vector-index.html#algo_settings).

> **Note:** In Couchbase Vector Search, the distance represents the vector distance between the query and document embeddings. Lower distance indicates higher similarity, while higher distance indicates lower similarity. This demo uses cosine similarity for measuring document relevance.

### Create the Semantic Cache FTS Index

The application requires a Couchbase FTS (Search) index on the cache collection to power the question-level semantic cache. This index enables vector similarity search over cached questions so that semantically equivalent follow-up questions are served from cache instead of calling OpenAI again.

The index should be created on the **cache collection** (e.g., `pdf-docs` → `shared` → `cached_llm_responses`). Each cached document has the following structure:

```json
{
  "text": "<original question>",
  "embedding": [/* 1536-dimensional OpenAI embedding vector */],
  "response": "<RAG response string>"
}
```

#### Import the index via Couchbase UI

- [Couchbase Capella](https://docs.couchbase.com/cloud/search/import-search-index.html)

  - Copy the index definition below to a new file `cache_index.json`
  - Import the file in Capella using the instructions in the documentation.
  - Click on **Create Index** to create the index.

- [Couchbase Server](https://docs.couchbase.com/server/current/search/import-search-index.html)

  - Click on **Search** → **Add Index** → **Import**
  - Paste the following index definition in the Import screen
  - Click on **Create Index** to create the index.

#### Index Definition

Update `sourceName`, `shared.cached_llm_responses`, and `CACHE_INDEX` in `secrets.toml` to match your bucket name, scope, and collection if they differ from the defaults below.

```json
{
  "name": "cached_llm_responses_index",
  "type": "fulltext-index",
  "params": {
    "doc_config": {
      "docid_prefix_delim": "",
      "docid_regexp": "",
      "mode": "scope.collection.type_field",
      "type_field": "type"
    },
    "mapping": {
      "default_analyzer": "standard",
      "default_datetime_parser": "dateTimeOptional",
      "default_field": "_all",
      "default_mapping": {
        "dynamic": true,
        "enabled": false
      },
      "default_type": "_default",
      "docvalues_dynamic": false,
      "index_dynamic": true,
      "store_dynamic": false,
      "type_field": "_type",
      "types": {
        "shared.cached_llm_responses": {
          "dynamic": true,
          "enabled": true,
          "properties": {
            "embedding": {
              "enabled": true,
              "dynamic": false,
              "fields": [
                {
                  "dims": 1536,
                  "index": true,
                  "name": "embedding",
                  "similarity": "dot_product",
                  "type": "vector",
                  "vector_index_optimized_for": "recall"
                }
              ]
            },
            "text": {
              "enabled": true,
              "dynamic": false,
              "fields": [
                {
                  "index": true,
                  "name": "text",
                  "store": true,
                  "type": "text"
                }
              ]
            }
          }
        }
      }
    },
    "store": {
      "indexType": "scorch",
      "segmentVersion": 16
    }
  },
  "sourceType": "gocbcore",
  "sourceName": "pdf-docs",
  "sourceParams": {},
  "planParams": {
    "maxPartitionsPerPIndex": 64,
    "indexPartitions": 16,
    "numReplicas": 0
  }
}
```

### Run the Application

```bash
streamlit run chat_with_pdf_query.py
```
