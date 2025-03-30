#!/usr/bin/env python3
import os
import torch
import pickle
import logging
import argparse
from typing import List, Dict, Any, Optional
from collections import Counter
import anthropic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenTitanRAG:
    """
    OpenTitan RAG system with optional query translation.
    """

    def __init__(
        self,
        docs_dir: str,
        api_key: str = None,
        model: str = "claude-3-opus-20240229",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        base_url: str = "https://github.com/lowRISC/opentitan/blob/master/"
    ):
        """Initialize the OpenTitan RAG system"""
        self.docs_dir = docs_dir
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.embedding_model_name = embedding_model
        self.device = device
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set as ANTHROPIC_API_KEY env variable")

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(
            embedding_model, device=device)

        # Initialize langchain embeddings
        self.lc_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
        )

        # Load vector store
        self.vector_store = self._load_vector_store()

        # Load document tree for metadata access (optional)
        try:
            self.document_tree = self._load_document_tree()
            self.node_summaries = self._load_node_summaries()
        except Exception as e:
            logger.warning(f"Could not load document tree: {e}")
            self.document_tree = None
            self.node_summaries = {}

        logger.info("OpenTitan RAG system initialized successfully")

    def _load_document_tree(self):
        """Load the document tree from pickle file."""
        tree_path = os.path.join(self.docs_dir, 'document_tree.pkl')
        if not os.path.exists(tree_path):
            logger.warning(f"Document tree not found at {tree_path}")
            return None

        logger.info(f"Loading document tree from {tree_path}")
        with open(tree_path, 'rb') as f:
            return pickle.load(f)

    def _load_node_summaries(self):
        """Load the node summaries from pickle file."""
        summaries_path = os.path.join(self.docs_dir, 'node_summaries.pkl')
        if not os.path.exists(summaries_path):
            logger.warning(f"Node summaries not found at {summaries_path}")
            return {}

        logger.info(f"Loading node summaries from {summaries_path}")
        with open(summaries_path, 'rb') as f:
            return pickle.load(f)

    def _load_vector_store(self):
        """Load the FAISS vector store."""
        index_path = os.path.join(self.docs_dir, 'faiss_index')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        logger.info(f"Loading vector store from {index_path}")
        # Set allow_dangerous_deserialization to True since you created this index yourself
        return FAISS.load_local(index_path, self.lc_embeddings, allow_dangerous_deserialization=True)

    def _claude_generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text using Claude API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=temperature,
            system="You are a helpful expert assistant that provides accurate, concise answers based solely on the provided context.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _create_source_url(self, source: str) -> str:
        """
        Create a URL for a source based on its path or ID
        Converts paths from text file references to proper website URLs
        """
        if source == "Unknown":
            return "#"

        # Handle numeric IDs which are likely not valid paths
        if source.isdigit() or (len(source) > 0 and source[-1] == ')' and source.rsplit('(', 1)[1][:-1].isdigit()):
            # Extract the ID if it's in the format "something(12345)"
            if source[-1] == ')' and '(' in source:
                id_part = source.rsplit('(', 1)[1][:-1]
                if id_part.isdigit():
                    source = id_part
            return f"https://opentitan.org/document-view/{source}"

        # Handle text file references that should be converted to website URLs
        if source.endswith('.txt'):
            # Check if this looks like a converted OpenTitan book path
            if 'hw_ip_otbn' in source:
                # Convert from hw_ip_otbn_doc_interfaces.txt to proper URL format
                path = source.replace('.txt', '')

                # Replace underscores with slashes for the URL path
                parts = path.split('_')
                url_path = []

                # Start building the path - handle hw/ip parts
                for i, part in enumerate(parts):
                    if part in ['hw', 'ip', 'otbn', 'dv', 'doc'] and i < len(parts) - 1:
                        url_path.append(part)
                    elif i == len(parts) - 1:  # Last part is the page name
                        url_path.append(part)
                    else:
                        # Skip parts that don't fit the pattern
                        continue

                # Join the parts with slashes and create the URL
                url_path_str = '/'.join(url_path)
                return f"https://opentitan.org/book/{url_path_str}.html"

        # Use the default GitHub URL as fallback for files that don't match patterns
        # Clean up the source path if needed
        if source.startswith('./') or source.startswith('/'):
            source = source.lstrip('./')

        # For real files in the GitHub repo, keep the GitHub URL
        if source.endswith(('.c', '.h', '.rs', '.py', '.cpp', '.json', '.html')):
            return f"{self.base_url}{source}"

        # For other sources, try to make an educated guess
        return f"https://opentitan.org/search?q={source.replace(' ', '+')}"

    def _add_missing_source_links(self, answer: str, source_to_url: Dict[int, str]) -> str:
        """Add clickable URLs to any source references that don't have them"""
        import re

        # Find all source references without links: Source X or [Source X]
        source_pattern = r'(?:\[)?Source\s+(\d+)(?:\])?(?!\()'

        def replace_with_link(match):
            source_num = int(match.group(1))
            if source_num in source_to_url:
                return f"[Source {source_num}]({source_to_url[source_num]})"
            return match.group(0)

        # Replace all source references with proper links
        linked_answer = re.sub(source_pattern, replace_with_link, answer)

        return linked_answer

    def translate_query(self, query: str) -> List[str]:
        """
        Translate a complex query into simpler component questions
        """
        logger.info(f"Translating query into components: {query}")

        # Use Claude to generate relevant sub-queries
        breakdown_prompt = f"""Break down this complex query into 3-5 simpler component questions that would help answer the main question. 
Return ONLY the list of questions, one per line, without any additional text.

Main question: {query}"""

        response = self._claude_generate(breakdown_prompt, temperature=0.2)

        # Clean up response and extract questions
        sub_queries = [
            line.strip() for line in response.split('\n')
            if line.strip() and "?" in line
        ]

        # Always include the original query
        if query not in sub_queries:
            sub_queries.append(query)

        logger.info(f"Generated {len(sub_queries)} sub-queries")
        return sub_queries

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_clusters: bool = True,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not loaded")

        logger.info(f"Retrieving documents for query: {query}")

        # Perform retrieval
        if include_clusters:
            retrieved_docs = self.vector_store.similarity_search_with_score(
                query, k=top_k)
        else:
            # Only retrieve document chunks (not cluster summaries)
            retrieved_docs = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
                filter={"type": "document"}
            )

        # Format results
        results = []
        for doc, score in retrieved_docs:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })

        # Print results if verbose
        if verbose:
            print(f"\nQuery: {query}")
            print("\nRetrieved Documents:")
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
                print(f"Type: {result['metadata']['type']}")
                source = result['metadata'].get(
                    'source', result['metadata'].get('node_id', 'N/A'))
                print(f"Source: {source}")
                print(f"Content: {result['content'][:100]}...")

        return results

    def aggregate_documents(
        self,
        sub_queries: List[str],
        top_k: int = 5,
        include_clusters: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve and aggregate documents for multiple sub-queries
        """
        # Track document frequencies and content
        all_docs = []
        doc_sources = []
        doc_contents = {}

        # Retrieve documents for each sub-query
        for i, sub_query in enumerate(sub_queries):
            logger.info(f"Retrieving for sub-query {i+1}: {sub_query}")

            docs = self.retrieve(
                sub_query,
                top_k=top_k,
                include_clusters=include_clusters,
                verbose=False
            )

            all_docs.extend(docs)

            # Track document sources and content
            for doc in docs:
                source = doc['metadata'].get(
                    'source', doc['metadata'].get('node_id', 'Unknown'))
                doc_sources.append(source)
                doc_contents[source] = doc['content']

        # Count source frequencies
        source_counter = Counter(doc_sources)

        # Sort documents by frequency
        unique_sources = list(source_counter.keys())
        unique_sources.sort(key=lambda s: source_counter[s], reverse=True)

        logger.info(
            f"Retrieved {len(all_docs)} total documents across {len(unique_sources)} unique sources")

        return {
            "all_docs": all_docs,
            "source_counter": source_counter,
            "doc_contents": doc_contents,
            "unique_sources": unique_sources
        }

    def generate_answer(
        self,
        query: str,
        documents: List[Dict[str, Any]] = None,
        aggregated_results: Dict[str, Any] = None,
        top_sources: int = 7,
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Generate an answer based on retrieved documents with clickable source links
        """
        logger.info(f"Generating answer for query: {query}")

        # Create a mapping of source index to URL
        source_to_url = {}
        source_index_mapping = {}

        # Format context based on retrieval method
        if aggregated_results:
            # Use weighted aggregation results
            source_counter = aggregated_results["source_counter"]
            doc_contents = aggregated_results["doc_contents"]
            unique_sources = aggregated_results["unique_sources"]

            # Build weighted context with URLs
            context = ""
            for i, source in enumerate(unique_sources[:top_sources]):
                if source in doc_contents:
                    # Create source URL
                    source_url = self._create_source_url(source)
                    source_to_url[i+1] = source_url
                    source_index_mapping[source] = i+1

                    # Add to context with URL
                    context += f"[Source {i+1}: {source} (appeared {source_counter[source]} times)]\nURL: {source_url}\n{doc_contents[source]}\n\n"
        else:
            # Use standard retrieval results
            context = ""
            for i, doc in enumerate(documents[:top_sources]):
                source = doc['metadata'].get(
                    'source', doc['metadata'].get('node_id', 'Unknown'))

                # Create source URL
                source_url = self._create_source_url(source)
                source_to_url[i+1] = source_url
                source_index_mapping[source] = i+1

                # Add to context with URL
                context += f"[Source {i+1}: {source}]\nURL: {source_url}\n{doc['content']}\n\n"

        # Use custom or default prompt template
        if prompt_template is None:
            prompt = f"""Answer the following question based ONLY on the information in the provided context.
If you cannot answer the question based solely on the context, say "I don't have enough information in the provided context to answer this question."

Context information (including source URLs):
{context}

Question: {query}

Your answer should be thorough but concise, providing specific information from the context that directly addresses the question.
When you reference a source, include the source number and the clickable URL like this: [Source X](URL).
For example, if you're referencing Source 1, write: [Source 1]({source_to_url.get(1, "#")})."""
        else:
            prompt = prompt_template.format(context=context, query=query)

        # Generate answer
        answer = self._claude_generate(prompt, temperature=0.0)

        # Process the answer to ensure all source references have URLs
        answer = self._add_missing_source_links(answer, source_to_url)

        return answer

    def process_query(
        self,
        query: str,
        use_query_translation: bool = False,
        top_k: int = 5,
        top_sources: int = 7,
        include_clusters: bool = True,
        prompt_template: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query using the OpenTitan RAG system
        """
        logger.info(f"Processing query: {query}")
        results = {"query": query}

        if use_query_translation:
            # Step 1: Translate query into sub-queries
            sub_queries = self.translate_query(query)
            results["sub_queries"] = sub_queries

            # Step 2: Retrieve and aggregate documents
            aggregated_results = self.aggregate_documents(
                sub_queries, top_k, include_clusters)

            retrieval_stats = {
                "total_docs": len(aggregated_results["all_docs"]),
                "unique_sources": len(aggregated_results["unique_sources"]),
                "source_frequencies": dict(aggregated_results["source_counter"])
            }
            results["retrieval_stats"] = retrieval_stats

            # Step 3: Generate answer
            answer = self.generate_answer(
                query,
                aggregated_results=aggregated_results,
                top_sources=top_sources,
                prompt_template=prompt_template
            )
        else:
            # Standard retrieval and generation
            documents = self.retrieve(
                query, top_k, include_clusters, verbose)
            results["documents"] = documents

            # Generate answer
            answer = self.generate_answer(
                query,
                documents=documents,
                top_sources=top_sources,
                prompt_template=prompt_template
            )

        results["answer"] = answer
        return results


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="OpenTitan RAG System")
    parser.add_argument("--docs_dir", default="raptor_output",
                        help="Directory containing document index files")
    parser.add_argument("--query", required=True,
                        help="The user query to process")
    parser.add_argument("--translate", action="store_true",
                        help="Use query translation for complex queries")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of documents to retrieve per query")
    parser.add_argument("--top_sources", type=int, default=7,
                        help="Number of top sources to include in final context")
    parser.add_argument("--include_clusters", action="store_true", default=True,
                        help="Include cluster summaries in retrieval")
    parser.add_argument("--model", default="claude-3-opus-20240229",
                        help="Claude model to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed retrieval results")
    parser.add_argument("--base_url", default="https://github.com/lowRISC/opentitan/blob/master/",
                        help="Base URL for source document links")

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your-api-key")
        return

    # Initialize and run
    rag = OpenTitanRAG(args.docs_dir, model=args.model, base_url=args.base_url)
    results = rag.process_query(
        query=args.query,
        use_query_translation=args.translate,
        top_k=args.top_k,
        top_sources=args.top_sources,
        include_clusters=args.include_clusters,
        verbose=args.verbose
    )

    # Print results
    print("\n" + "="*80)

    if args.translate:
        print("QUERY BREAKDOWN:")
        for i, sub_query in enumerate(results["sub_queries"]):
            print(f"{i+1}. {sub_query}")

        print("\n" + "="*80)
        print("RETRIEVAL STATISTICS:")
        print(
            f"- Total documents retrieved: {results['retrieval_stats']['total_docs']}")
        print(
            f"- Unique sources: {results['retrieval_stats']['unique_sources']}")
        print("\nTop sources by frequency:")

        for source, count in sorted(
            results['retrieval_stats']['source_frequencies'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            print(f"- {source}: {count} occurrences")
    else:
        if args.verbose:
            print("RETRIEVED DOCUMENTS:")
            for i, doc in enumerate(results["documents"][:5]):
                source = doc['metadata'].get(
                    'source', doc['metadata'].get('node_id', 'Unknown'))
                print(f"{i+1}. Source: {source} (Score: {doc['score']:.4f})")
                print(f"   {doc['content'][:100]}...")

    print("\n" + "="*80)
    print("FINAL ANSWER:")
    print(results["answer"])
    print("="*80)


if __name__ == "__main__":
    main()
