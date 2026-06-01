import os
import json
import glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── IMPORTS ────────────────────────────
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not available")

# ── KNOWLEDGE BASE LOADER ──────────────
def load_all_knowledge_bases():
    """
    Load all JSON knowledge base files
    from knowledge_base directory
    """
    all_prompts = []
    kb_path = Path("knowledge_base")
    
    if not kb_path.exists():
        print("Knowledge base directory not found!")
        return []
    
    json_files = list(kb_path.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in knowledge_base!")
        return []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tool_name = data.get('tool', 'unknown')
                prompts = data.get('prompts', [])
                
                for prompt in prompts:
                    prompt['tool'] = tool_name
                    all_prompts.append(prompt)
                    
            print(f"✅ Loaded {len(prompts)} prompts from {json_file.name}")
            
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
    
    print(f"\n📚 Total prompts loaded: {len(all_prompts)}")
    return all_prompts

# ── VECTOR STORE BUILDER ───────────────
def build_vector_store(prompts):
    """
    Build ChromaDB vector store
    from loaded prompts
    """
    if not CHROMA_AVAILABLE:
        print("ChromaDB not available — using simple search")
        return None
    
    try:
        # Initialize ChromaDB
        client = chromadb.PersistentClient(
            path="./chroma_db"
        )
        
        # Use sentence transformers for embeddings
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        collection_name = "promptly_knowledge_base"
        
        # Delete existing collection if exists
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare data for indexing
        documents = []
        metadatas = []
        ids = []
        
        for i, prompt in enumerate(prompts):
            # Create searchable document
            doc = f"""
            Tool: {prompt.get('tool', '')}
            Category: {prompt.get('category', '')}
            Description: {prompt.get('description', '')}
            Tags: {' '.join(prompt.get('tags', []))}
            Prompt: {prompt.get('prompt', '')}
            """
            
            documents.append(doc.strip())
            metadatas.append({
                'tool': prompt.get('tool', ''),
                'category': prompt.get('category', ''),
                'subcategory': prompt.get('subcategory', ''),
                'description': prompt.get('description', ''),
                'prompt': prompt.get('prompt', ''),
                'tags': ','.join(prompt.get('tags', [])),
                'id': prompt.get('id', f'prompt_{i}')
            })
            ids.append(f"prompt_{i}")
        
        # Add to collection in batches
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
        
        print(f"✅ Vector store built with {len(documents)} prompts!")
        return collection
        
    except Exception as e:
        print(f"❌ Error building vector store: {e}")
        return None

# ── SEMANTIC SEARCH ────────────────────
def search_similar_prompts(
    query,
    collection,
    tool_filter=None,
    n_results=5
):
    """
    Search for similar prompts using
    semantic similarity
    """
    if collection is None:
        return []
    
    try:
        # Build where filter
        where = None
        if tool_filter and tool_filter != "All Tools":
            where = {"tool": tool_filter.lower()}
        
        # Semantic search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        # Format results
        similar_prompts = []
        
        if results and results['metadatas']:
            for i, metadata in enumerate(
                results['metadatas'][0]
            ):
                distance = results['distances'][0][i] \
                    if results.get('distances') else 0
                
                similar_prompts.append({
                    'tool': metadata.get('tool', ''),
                    'category': metadata.get('category', ''),
                    'description': metadata.get('description', ''),
                    'prompt': metadata.get('prompt', ''),
                    'tags': metadata.get('tags', '').split(','),
                    'relevance': round(1 - distance, 3)
                })
        
        return similar_prompts
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

# ── SIMPLE SEARCH FALLBACK ─────────────
def simple_search(query, prompts, tool_filter=None, n=5):
    """
    Simple keyword search fallback
    when ChromaDB unavailable
    """
    query_words = query.lower().split()
    scored = []
    
    for prompt in prompts:
        if tool_filter and tool_filter != "All Tools":
            if prompt.get('tool', '') != tool_filter.lower():
                continue
        
        score = 0
        text = (
            prompt.get('description', '') + ' ' +
            prompt.get('category', '') + ' ' +
            ' '.join(prompt.get('tags', []))
        ).lower()
        
        for word in query_words:
            if word in text:
                score += 1
        
        if score > 0:
            scored.append((score, prompt))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    return [
        {
            'tool': p.get('tool', ''),
            'category': p.get('category', ''),
            'description': p.get('description', ''),
            'prompt': p.get('prompt', ''),
            'tags': p.get('tags', []),
            'relevance': score / len(query_words)
        }
        for score, p in scored[:n]
    ]

# ── INITIALIZE RAG ─────────────────────
def initialize_rag():
    """
    Initialize the complete RAG system
    Returns collection and all prompts
    """
    print("🚀 Initializing Promptly RAG System...")
    
    # Load knowledge base
    prompts = load_all_knowledge_bases()
    
    if not prompts:
        print("❌ No prompts loaded!")
        return None, []
    
    # Build vector store
    collection = build_vector_store(prompts)
    
    if collection:
        print("✅ RAG System ready!")
    else:
        print("⚠️ Using simple search fallback")
    
    return collection, prompts

def initialize_rag():
    """
    Initialize the complete RAG system
    Returns collection and all prompts
    """
    print("🚀 Initializing Promptly RAG System...")
    prompts = load_all_knowledge_bases()
    
    if not prompts:
        print("❌ No prompts loaded!")
        return None, []
    
    collection = build_vector_store(prompts)
    
    if collection:
        print("✅ RAG System ready!")
    else:
        print("⚠️ Using simple search fallback")
    
    return collection, prompts