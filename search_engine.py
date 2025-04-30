import json
import os
import re
import math
import random
import numpy as np
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('stopwords', quiet=True)
DATA_FOLDER = "indexing/project_output"

class SearchEngine:
    def __init__(self, data_folder=DATA_FOLDER):
        # Initialize the Porter Stemmer and stopwords
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load data from files
        self.data_folder = data_folder
        self.load_indexes()
        self.load_url_map()
        
        # For tracking relevance feedback
        self.query_history = {}
        self.relevance_feedback = {}
        
        # For association clusters - no precomputed co-occurrence matrix
        self.term_co_occurrence_cache = {}  # Cache for query-specific matrices
        
        # Pre-computed clusters (will be initialized on first use)
        self.document_vectors = None
        self.term_index = None
        self.metric_clusters = None
        self.scalar_clusters = None
        
        print("Search engine initialized and ready for queries.")
    
    def load_indexes(self):
        """Load all index and ranking files"""
        print("Loading indexes and ranking data...")
        
        # Load inverted index (X1)
        with open(os.path.join(self.data_folder, 'inverted_index.json'), 'r') as f:
            self.inverted_index = json.load(f)
        
        # Load TF-IDF scores (X2)
        with open(os.path.join(self.data_folder, 'tf_idf_scores.json'), 'r') as f:
            self.tf_idf = json.load(f)
            
        # Load PageRank scores (X3)
        with open(os.path.join(self.data_folder, 'pagerank_scores.json'), 'r') as f:
            self.pagerank = json.load(f)
            
        # Load HITS scores (X3)
        with open(os.path.join(self.data_folder, 'hits_scores.json'), 'r') as f:
            hits_data = json.load(f)
            self.hubs = hits_data["hubs"]
            self.authorities = hits_data["authorities"]
        
        print("Indexes loaded successfully.")
    
    def load_url_map(self):
        """Load URL mapping data"""
        self.url_map = {}
        self.reverse_url_map = {}
        
        # Try to find either url_ids.jsonl or test_url_ids.jsonl
        url_map_files = ['url_ids.jsonl']
        
        for url_file in url_map_files:
            if os.path.exists(url_file):
                print(f"Loading URL map from {url_file}")
                with open(url_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            doc_id = str(data['url_id'])
                            url = data['url']
                            self.url_map[doc_id] = url
                            self.reverse_url_map[url] = doc_id
                break
        
        if not self.url_map:
            print("Warning: URL mapping file not found.")
    
    def preprocess_query(self, query_text):
        """Process the query text - tokenize, remove stopwords, stem"""
        # Tokenize
        tokens = re.findall(r'\w+', query_text.lower())
        
        # Remove stopwords and stem
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                processed_tokens.append(self.ps.stem(token))
                
        return processed_tokens
    
    def get_matching_docs(self, query_tokens):
        """Get all documents that match at least one query term"""
        matching_docs = set()
        
        for token in query_tokens:
            # Get documents containing the token
            if token in self.inverted_index:
                for doc_id in self.inverted_index[token]:
                    matching_docs.add(doc_id)
        
        return matching_docs
    
    def calculate_query_vector(self, query_tokens):
        """Calculate TF-IDF vector for the query"""
        query_tf = defaultdict(int)
        
        # Count term frequencies in query
        for token in query_tokens:
            query_tf[token] += 1
        
        # Calculate TF-IDF scores
        query_vector = {}
        for token, freq in query_tf.items():
            # Log normalization for TF
            tf = 1 + math.log(freq) if freq > 0 else 0
            
            # Use IDF from corpus if available
            idf = 1.0  # default
            if token in self.inverted_index:
                # Calculate IDF based on document frequency
                N = len(set().union(*[set(self.tf_idf[doc].keys()) for doc in self.tf_idf]))
                df = len(set(self.inverted_index[token]))
                idf = math.log(N / (1 + df))
            
            query_vector[token] = tf * idf
        
        return query_vector
    
    def calculate_cosine_similarity(self, query_vector, doc_id):
        """Calculate cosine similarity between query and document"""
        dot_product = 0.0
        query_magnitude = 0.0
        doc_magnitude = 0.0
        
        # Calculate dot product
        for term, query_weight in query_vector.items():
            if term in self.inverted_index and doc_id in self.tf_idf and term in self.tf_idf[doc_id]:
                doc_weight = self.tf_idf[doc_id][term]
                dot_product += query_weight * doc_weight
        
        # Calculate magnitudes
        for weight in query_vector.values():
            query_magnitude += weight ** 2
        
        if doc_id in self.tf_idf:
            for weight in self.tf_idf[doc_id].values():
                doc_magnitude += weight ** 2
        
        # Calculate similarity
        magnitude = math.sqrt(query_magnitude) * math.sqrt(doc_magnitude)
        if magnitude > 0:
            return dot_product / magnitude
        else:
            return 0.0
    
    def _build_term_co_occurrence_for_query(self, query_tokens):
        """Build term co-occurrence matrix specifically for this query"""
        print("Building targeted co-occurrence matrix for query terms...")
        co_occurrence = defaultdict(Counter)
        
        # Get a cache key for this query (sorted query tokens)
        cache_key = tuple(sorted(query_tokens))
        
        # Check if we have a cached matrix for this query
        if cache_key in self.term_co_occurrence_cache:
            print("Using cached co-occurrence matrix for this query")
            return self.term_co_occurrence_cache[cache_key]
        
        # Find documents that contain at least one query term
        matching_docs = self.get_matching_docs(query_tokens)
        print(f"Found {len(matching_docs)} documents containing query terms")
        
        # Limit document sample size for very large result sets
        max_docs = 5000
        if len(matching_docs) > max_docs:
            print(f"Sampling {max_docs} documents out of {len(matching_docs)} matches")
            matching_docs = set(random.sample(list(matching_docs), max_docs))
        
        # Get most common terms for efficiency
        term_doc_frequency = Counter()
        for term, doc_list in self.inverted_index.items():
            # Only count frequency in matching docs
            term_doc_frequency[term] = len(set(doc_list).intersection(matching_docs))
        
        # Take top terms plus ensure query terms are included
        common_terms = set([term for term, _ in term_doc_frequency.most_common(3000)])
        for term in query_tokens:
            common_terms.add(term)
        
        # Process matching documents
        for i, doc_id in enumerate(matching_docs):
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i}/{len(matching_docs)} documents for co-occurrence matrix...")
            
            # Get terms from the document, filtering to common terms only
            if doc_id in self.tf_idf:
                doc_terms = [term for term in self.tf_idf[doc_id].keys() if term in common_terms]
                
                # Limit terms per document to avoid quadratic explosion
                if len(doc_terms) > 100:
                    # Make sure query terms stay in the sample
                    query_terms_in_doc = [t for t in query_tokens if t in doc_terms]
                    other_terms = [t for t in doc_terms if t not in query_tokens]
                    # Sample from other terms
                    if len(other_terms) > (100 - len(query_terms_in_doc)):
                        other_terms = random.sample(other_terms, 100 - len(query_terms_in_doc))
                    doc_terms = query_terms_in_doc + other_terms
                
                # Build co-occurrence counts
                for i, term1 in enumerate(doc_terms):
                    for term2 in doc_terms[i+1:]:
                        co_occurrence[term1][term2] += 1
                        co_occurrence[term2][term1] += 1
        
        print(f"Co-occurrence matrix built with {len(co_occurrence)} terms.")
        
        # Cache the result for future use with this query
        self.term_co_occurrence_cache[cache_key] = co_occurrence
        
        # Limit cache size to avoid memory issues
        if len(self.term_co_occurrence_cache) > 20:  # Keep only 20 most recent matrices
            oldest_key = next(iter(self.term_co_occurrence_cache))
            del self.term_co_occurrence_cache[oldest_key]
        
        return co_occurrence
    
    def _create_document_vectors(self):
        """Create vector representation of documents for clustering"""
        print("Creating document vectors for clustering...")
        
        # Get all unique terms
        all_terms = set()
        for doc_id in self.tf_idf:
            all_terms.update(self.tf_idf[doc_id].keys())
        
        # Sort terms for consistent indexing
        term_index = {term: i for i, term in enumerate(sorted(all_terms))}
        
        # Create document vectors
        doc_vectors = {}
        for doc_id in self.tf_idf:
            vector = np.zeros(len(term_index))
            for term, weight in self.tf_idf[doc_id].items():
                vector[term_index[term]] = weight
            doc_vectors[doc_id] = vector
        
        return doc_vectors, term_index
    
    def _initialize_clusters(self, n_clusters=10):
        """Initialize document clusters for query expansion"""
        print(f"Initializing {n_clusters} document clusters...")
        
        # Create document vectors if not already done
        if self.document_vectors is None:
            self.document_vectors, self.term_index = self._create_document_vectors()
        
        # Extract vectors and document IDs
        doc_ids = list(self.document_vectors.keys())
        vectors = np.array([self.document_vectors[doc_id] for doc_id in doc_ids])
        
        # Create metric clusters (using KMeans)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(vectors)
        
        # Create scalar clusters (using hierarchical clustering)
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = hierarchical.fit_predict(vectors)
        
        # Store cluster assignments
        metric_clusters = defaultdict(list)
        scalar_clusters = defaultdict(list)
        
        for i, doc_id in enumerate(doc_ids):
            metric_clusters[int(kmeans_labels[i])].append(doc_id)
            scalar_clusters[int(hierarchical_labels[i])].append(doc_id)
        
        return metric_clusters, scalar_clusters
    
    # --- X5: QUERY EXPANSION METHODS ---
    
    def rocchio_expansion(self, query_text, relevant_docs, non_relevant_docs, alpha=1.0, beta=0.75, gamma=0.15):
        """
        Implement Rocchio algorithm for query expansion
        
        Parameters:
        - query_text: Original query
        - relevant_docs: List of relevant document IDs
        - non_relevant_docs: List of non-relevant document IDs
        - alpha: Weight for original query
        - beta: Weight for relevant documents
        - gamma: Weight for non-relevant documents
        
        Returns:
        - Expanded query terms
        """
        # Process original query
        query_tokens = self.preprocess_query(query_text)
        query_vector = self.calculate_query_vector(query_tokens)
        
        # Initialize new query vector
        new_query = {term: weight * alpha for term, weight in query_vector.items()}
        
        # Add relevant documents component
        if relevant_docs:
            for doc_id in relevant_docs:
                if doc_id in self.tf_idf:
                    for term, weight in self.tf_idf[doc_id].items():
                        new_query[term] = new_query.get(term, 0) + (beta * weight / len(relevant_docs))
        
        # Subtract non-relevant documents component
        if non_relevant_docs:
            for doc_id in non_relevant_docs:
                if doc_id in self.tf_idf:
                    for term, weight in self.tf_idf[doc_id].items():
                        new_query[term] = new_query.get(term, 0) - (gamma * weight / len(non_relevant_docs))
        
        # Sort terms by weight and select top terms
        sorted_terms = sorted(new_query.items(), key=lambda x: x[1], reverse=True)
        top_terms = [term for term, weight in sorted_terms if weight > 0][:10]
        
        # Store query modification for reporting
        query_id = hash(query_text)
        self.query_history[query_id] = {
            'original': query_text,
            'expanded_terms': top_terms,
            'relevant_docs': relevant_docs,
            'non_relevant_docs': non_relevant_docs
        }
        
        return top_terms
    
    def association_cluster_expansion(self, query_text, num_terms=3):
        """
        Expand query using association clusters
        
        This method finds terms that frequently co-occur with query terms
        """
        query_tokens = self.preprocess_query(query_text)
        
        # Build co-occurrence matrix just for this query
        term_co_occurrence = self._build_term_co_occurrence_for_query(query_tokens)
        
        expansion_terms = Counter()
        
        # For each query term, find associated terms
        for term in query_tokens:
            if term in term_co_occurrence:
                # Get top co-occurring terms
                for related_term, count in term_co_occurrence[term].most_common(5):
                    if related_term not in query_tokens:
                        expansion_terms[related_term] += count
        
        # Select top expansion terms
        top_expansion = [term for term, _ in expansion_terms.most_common(num_terms)]
        
        # Store for reporting
        query_id = hash(query_text)
        self.query_history[query_id] = {
            'original': query_text,
            'method': 'association_cluster',
            'expanded_terms': top_expansion
        }
        
        return query_tokens + top_expansion
    
    def metric_cluster_expansion(self, query_text, num_terms=3):
        """
        Expand query using metric clusters (KMeans)
        
        This method finds the cluster most relevant to the query
        and adds common terms from that cluster
        """
        # Initialize clusters if needed
        if self.metric_clusters is None:
            self.metric_clusters, self.scalar_clusters = self._initialize_clusters()
        
        query_tokens = self.preprocess_query(query_text)
        matching_docs = self.get_matching_docs(query_tokens)
        
        # Find which cluster contains most matching documents
        cluster_matches = Counter()
        for doc_id in matching_docs:
            for cluster_id, docs in self.metric_clusters.items():
                if doc_id in docs:
                    cluster_matches[cluster_id] += 1
        
        # If no matching clusters, return original query
        if not cluster_matches:
            return query_tokens
        
        # Get most relevant cluster
        best_cluster = cluster_matches.most_common(1)[0][0]
        cluster_docs = self.metric_clusters[best_cluster]
        
        # Find common terms in the cluster
        term_frequency = Counter()
        for doc_id in cluster_docs:
            if doc_id in self.tf_idf:
                for term in self.tf_idf[doc_id]:
                    if term not in query_tokens:
                        term_frequency[term] += 1
        
        # Add top terms to query
        expansion_terms = [term for term, _ in term_frequency.most_common(num_terms)]
        
        # Store for reporting
        query_id = hash(query_text)
        self.query_history[query_id] = {
            'original': query_text,
            'method': 'metric_cluster',
            'expanded_terms': expansion_terms
        }
        
        return query_tokens + expansion_terms
    
    def scalar_cluster_expansion(self, query_text, num_terms=3):
        """
        Expand query using scalar clusters (Hierarchical)
        
        Similar to metric clusters but uses hierarchical clustering
        """
        # Initialize clusters if needed
        if self.scalar_clusters is None:
            self.metric_clusters, self.scalar_clusters = self._initialize_clusters()
        
        query_tokens = self.preprocess_query(query_text)
        matching_docs = self.get_matching_docs(query_tokens)
        
        # Find which cluster contains most matching documents
        cluster_matches = Counter()
        for doc_id in matching_docs:
            for cluster_id, docs in self.scalar_clusters.items():
                if doc_id in docs:
                    cluster_matches[cluster_id] += 1
        
        # If no matching clusters, return original query
        if not cluster_matches:
            return query_tokens
        
        # Get most relevant cluster
        best_cluster = cluster_matches.most_common(1)[0][0]
        cluster_docs = self.scalar_clusters[best_cluster]
        
        # Find common terms in the cluster
        term_weights = defaultdict(float)
        for doc_id in cluster_docs:
            if doc_id in self.tf_idf:
                for term, weight in self.tf_idf[doc_id].items():
                    if term not in query_tokens:
                        term_weights[term] += weight
        
        # Add top weighted terms to query
        expansion_terms = [term for term, _ in 
                          sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:num_terms]]
        
        # Store for reporting
        query_id = hash(query_text)
        self.query_history[query_id] = {
            'original': query_text,
            'method': 'scalar_cluster',
            'expanded_terms': expansion_terms
        }
        
        return query_tokens + expansion_terms
    
    def expand_query(self, query_text, method="rocchio", relevant_docs=None, non_relevant_docs=None):
        """
        Expand query using specified method
        
        Parameters:
        - query_text: Original query text
        - method: Expansion method to use
        - relevant_docs: List of relevant document IDs (for Rocchio)
        - non_relevant_docs: List of non-relevant document IDs (for Rocchio)
        
        Returns:
        - Expanded query tokens
        """
        if method == "rocchio":
            if not relevant_docs:
                relevant_docs = []
            if not non_relevant_docs:
                non_relevant_docs = []
            return self.rocchio_expansion(query_text, relevant_docs, non_relevant_docs)
        
        elif method == "association":
            return self.association_cluster_expansion(query_text)
        
        elif method == "metric":
            return self.metric_cluster_expansion(query_text)
        
        elif method == "scalar":
            return self.scalar_cluster_expansion(query_text)
        
        else:
            # If method not recognized, return original tokens
            return self.preprocess_query(query_text)
    
    def rank_results(self, matching_docs, query_vector, method="combined"):
        """Rank matching documents based on specified method (optimized)"""
        scores = {}
        
        # Pre-fetch all URLs for matching docs at once to avoid repeated lookups
        doc_urls = {}
        if method == "pagerank" or method == "hits" or method == "combined":
            for doc_id in matching_docs:
                doc_urls[doc_id] = self.url_map.get(doc_id, "")
        
        # TF-IDF ranking
        if method == "tfidf" or method == "combined":
            # Calculate TF-IDF similarity scores
            for doc_id in matching_docs:
                scores[doc_id] = self.calculate_cosine_similarity(query_vector, doc_id)
        
        # PageRank ranking
        if method == "pagerank" or method == "combined":
            # Look up pagerank scores
            pagerank_scores = {}
            for doc_id, url in doc_urls.items():
                if url in self.pagerank:
                    pagerank_scores[doc_id] = self.pagerank[url]
            
            # Apply PageRank scores
            for doc_id, pagerank_score in pagerank_scores.items():
                if method == "pagerank":
                    scores[doc_id] = pagerank_score
                else:  # combined
                    scores[doc_id] = scores.get(doc_id, 0) * 0.7 + pagerank_score * 0.3
        
        # HITS ranking
        if method == "hits" or method == "combined":
            # Look up authority scores
            authority_scores = {}
            for doc_id, url in doc_urls.items():
                if url in self.authorities:
                    authority_scores[doc_id] = self.authorities[url]
            
            # Apply authority scores
            for doc_id, auth_score in authority_scores.items():
                if method == "hits":
                    scores[doc_id] = auth_score
                else:  # combined
                    scores[doc_id] = scores.get(doc_id, 0) * 0.8 + auth_score * 0.2
        
        # Sort by score in descending order
        ranked_results = [(doc_id, scores.get(doc_id, 0)) for doc_id in matching_docs]
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_results
    
    def search(self, query_text, ranking_method="combined", expansion_method=None, 
               relevant_docs=None, non_relevant_docs=None, max_results=10):
        """
        Search for documents matching the query with optional query expansion
        
        Parameters:
        - query_text: User query
        - ranking_method: Method to rank results
        - expansion_method: Query expansion method (None, 'rocchio', 'association', 'metric', 'scalar')
        - relevant_docs: List of relevant document IDs (for Rocchio)
        - non_relevant_docs: List of non-relevant document IDs (for Rocchio)
        - max_results: Maximum number of results to return
        
        Returns:
        - List of search results
        """
        # Perform query expansion if requested
        if expansion_method:
            expanded_tokens = self.expand_query(
                query_text, 
                method=expansion_method,
                relevant_docs=relevant_docs,
                non_relevant_docs=non_relevant_docs
            )
            original_tokens = self.preprocess_query(query_text)
            
            print(f"Original query: {' '.join(original_tokens)}")
            print(f"Expanded query: {' '.join(expanded_tokens)}")
            
            query_tokens = expanded_tokens
        else:
            query_tokens = self.preprocess_query(query_text)
        
        if not query_tokens:
            return [], query_tokens
        
        # Get matching documents
        matching_docs = self.get_matching_docs(query_tokens)
        if not matching_docs:
            return [], query_tokens
        
        # Calculate query vector
        query_vector = self.calculate_query_vector(query_tokens)
        
        # Rank results
        ranked_results = self.rank_results(matching_docs, query_vector, ranking_method)
        
        # Format results
        results = []
        for doc_id, score in ranked_results[:max_results]:
            url = self.url_map.get(doc_id, "Unknown URL")
            results.append({
                "doc_id": doc_id,
                "url": url,
                "score": score
            })
        
        return results, query_tokens
    
    def get_relevance_feedback(self, query_text, results, max_feedback=5):
        """Simulate or collect relevance feedback"""
        # In a real system, this would ask the user for feedback
        # For testing purposes, we'll simulate feedback
        
        print("\nPlease mark relevant documents (y/n):")
        
        relevant_docs = []
        non_relevant_docs = []
        
        for i, result in enumerate(results[:max_feedback], 1):
            response = input(f"{i}. {result['url']} - Relevant? (y/n): ")
            if response.lower() == 'y':
                relevant_docs.append(result['doc_id'])
            else:
                non_relevant_docs.append(result['doc_id'])
        
        # Store feedback for reporting
        query_id = hash(query_text)
        self.relevance_feedback[query_id] = {
            'query': query_text,
            'relevant': relevant_docs,
            'non_relevant': non_relevant_docs
        }
        
        return relevant_docs, non_relevant_docs
    
    def test_expansion_methods(self, queries, expansion_method):
        """Test query expansion method on a set of queries"""
        results = []
        
        for query in queries:
            print(f"\nTesting query: {query}")
            
            # Original search
            original_results, _ = self.search(query)
            
            # Expanded search
            expanded_results, expanded_tokens = self.search(query, expansion_method=expansion_method)
            
            # Calculate improvement
            if original_results and expanded_results:
                # In a real scenario, we'd measure precision/recall
                # Here we'll just compare result counts
                improvement = len(expanded_results) - len(original_results)
            else:
                improvement = 0
            
            results.append({
                'query': query,
                'original_count': len(original_results),
                'expanded_count': len(expanded_results),
                'expanded_tokens': expanded_tokens,
                'improvement': improvement
            })
            
            print(f"Original results: {len(original_results)}")
            print(f"Expanded results: {len(expanded_results)}")
            print(f"Expanded query: {' '.join(expanded_tokens)}")
        
        return results
    
    def test_rocchio(self, queries):
        """Test Rocchio algorithm on a set of queries"""
        results = []
        
        for query in queries:
            print(f"\nTesting Rocchio on query: {query}")
            
            # Original search
            original_results, _ = self.search(query)
            
            if not original_results:
                print("No results found for original query.")
                continue
            
            # Get relevance feedback
            relevant_docs, non_relevant_docs = self.get_relevance_feedback(query, original_results)
            
            # Expanded search using Rocchio
            expanded_results, expanded_tokens = self.search(
                query, 
                expansion_method="rocchio",
                relevant_docs=relevant_docs,
                non_relevant_docs=non_relevant_docs
            )
            
            # Calculate improvement
            if original_results and expanded_results:
                # In a real scenario, we'd measure precision/recall
                improvement = len(expanded_results) - len(original_results)
            else:
                improvement = 0
            
            results.append({
                'query': query,
                'original_count': len(original_results),
                'expanded_count': len(expanded_results),
                'expanded_tokens': expanded_tokens,
                'improvement': improvement,
                'relevant_docs': relevant_docs,
                'non_relevant_docs': non_relevant_docs
            })
            
            print(f"Original results: {len(original_results)}")
            print(f"Expanded results: {len(expanded_results)}")
            print(f"Expanded query: {' '.join(expanded_tokens)}")
        
        return results
    
    def generate_test_report(self, rocchio_results, association_results, metric_results, scalar_results):
        """Generate a report of test results for all methods"""
        report = {
            'rocchio': {
                'queries_tested': len(rocchio_results),
                'avg_improvement': sum(r['improvement'] for r in rocchio_results) / len(rocchio_results) if rocchio_results else 0,
                'results': rocchio_results
            },
            'association': {
                'queries_tested': len(association_results),
                'avg_improvement': sum(r['improvement'] for r in association_results) / len(association_results) if association_results else 0,
                'results': association_results
            },
            'metric': {
                'queries_tested': len(metric_results),
                'avg_improvement': sum(r['improvement'] for r in metric_results) / len(metric_results) if metric_results else 0,
                'results': metric_results
            },
            'scalar': {
                'queries_tested': len(scalar_results),
                'avg_improvement': sum(r['improvement'] for r in scalar_results) / len(scalar_results) if scalar_results else 0,
                'results': scalar_results
            }
        }
        
        # Save report to file
        with open('query_expansion_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nTest report generated and saved to 'query_expansion_report.json'")
        
        return report

# Command-line interface
if __name__ == "__main__":
    # Initialize search engine
    search_engine = SearchEngine()
    
    while True:
        print("\n===== Museum Search Engine =====")
        print("1. Search")
        print("2. Search with Query Expansion")
        print("3. Test Rocchio Algorithm (20 queries)")
        print("4. Test Other Expansion Methods (50 queries)")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            # Basic search
            query = input("\nEnter your search query: ")
            
            print("\nRanking methods:")
            print("1. TF-IDF")
            print("2. PageRank")
            print("3. HITS")
            print("4. Combined (Default)")
            method_choice = input("Choose ranking method (1-4): ")
            
            method_map = {
                "1": "tfidf",
                "2": "pagerank",
                "3": "hits",
                "4": "combined"
            }
            ranking_method = method_map.get(method_choice, "combined")
            
            results, _ = search_engine.search(query, ranking_method)
            
            if results:
                print(f"\nFound {len(results)} results for '{query}':")
                print("-" * 60)
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['url']}")
                    print(f"   Score: {result['score']:.4f}")
                    print(f"   Document ID: {result['doc_id']}")
                    print("-" * 60)
            else:
                print(f"No results found for '{query}'")
                
        elif choice == '2':
            # Search with query expansion
            query = input("\nEnter your search query: ")
            
            print("\nExpansion methods:")
            print("1. Rocchio Algorithm (requires feedback)")
            print("2. Association Clusters")
            print("3. Metric Clusters")
            print("4. Scalar Clusters")
            expansion_choice = input("Choose expansion method (1-4): ")
            
            expansion_map = {
                "1": "rocchio",
                "2": "association",
                "3": "metric",
                "4": "scalar"
            }
            expansion_method = expansion_map.get(expansion_choice, "association")
            
            # Perform initial search for Rocchio feedback
            relevant_docs = []
            non_relevant_docs = []
            
            if expansion_method == "rocchio":
                initial_results, _ = search_engine.search(query)
                if initial_results:
                    relevant_docs, non_relevant_docs = search_engine.get_relevance_feedback(query, initial_results)
                else:
                    print("No initial results for relevance feedback. Using default expansion.")
                    expansion_method = "association"
            
            # Perform expanded search
            results, expanded_tokens = search_engine.search(
                query, 
                expansion_method=expansion_method,
                relevant_docs=relevant_docs,
                non_relevant_docs=non_relevant_docs
            )
            
            if results:
                print(f"\nFound {len(results)} results for expanded query '{' '.join(expanded_tokens)}':")
                print("-" * 60)
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['url']}")
                    print(f"   Score: {result['score']:.4f}")
                    print(f"   Document ID: {result['doc_id']}")
                    print("-" * 60)
            else:
                print(f"No results found for expanded query '{' '.join(expanded_tokens)}'")
        
        elif choice == '3':
            # Test Rocchio Algorithm
            print("\nTesting Rocchio Algorithm on 20 queries...")
            
            # Sample queries for testing Rocchio
            test_queries = [
                "art museum", "modern paintings", "renaissance sculpture", 
                "ancient artifacts", "egyptian collection", "abstract art",
                "impressionism", "cubism", "surrealism", "contemporary art",
                "native american art", "asian ceramics", "islamic art",
                "medieval art", "photography exhibition", "greek statues",
                "european paintings", "american artists", "african masks",
                "pop art"
            ]
            
            # Run tests
            rocchio_results = search_engine.test_rocchio(test_queries)
            
            # Display summary
            successful = sum(1 for r in rocchio_results if r['improvement'] > 0)
            print(f"\nRocchio testing completed.")
            print(f"Queries tested: {len(rocchio_results)}")
            print(f"Queries with improved results: {successful}")
        
        elif choice == '4':
            # Test other expansion methods
            print("\nTesting other expansion methods on 50 queries...")
            
            # Sample queries for testing other methods
            test_queries = [
                "museum collection", "oil painting", "marble sculpture", 
                "ancient pottery", "egyptian mummies", "abstract expressionism",
                "french impressionism", "pablo picasso", "salvador dali", "installation art",
                "indigenous artifacts", "chinese pottery", "persian miniatures",
                "gothic architecture", "portrait photography", "roman busts",
                "dutch masters", "american modernism", "tribal masks",
                "andy warhol", "japanese prints", "ceramic arts", "textile design",
                "bronze casting", "religious iconography", "landscape painting",
                "art deco", "art nouveau", "bauhaus", "minimalism",
                "conceptual art", "performance art", "video installation", "digital art",
                "street art", "graffiti", "metalwork", "glassblowing", "jewelry design",
                "furniture design", "tapestry", "manuscript illumination", "calligraphy",
                "printmaking", "lithography", "etching", "watercolor", "pastel drawing",
                "collage", "mixed media"
            ]
            
            # Run tests for each method
            print("\nTesting Association Clusters...")
            association_results = search_engine.test_expansion_methods(test_queries, "association")
            
            print("\nTesting Metric Clusters...")
            metric_results = search_engine.test_expansion_methods(test_queries, "metric")
            
            print("\nTesting Scalar Clusters...")
            scalar_results = search_engine.test_expansion_methods(test_queries, "scalar")
            
            # Generate and display report
            search_engine.generate_test_report([], association_results, metric_results, scalar_results)
            
            # Display summary
            print("\nExpansion methods testing completed.")
            print(f"Association Clusters: {sum(1 for r in association_results if r['improvement'] > 0)}/{len(association_results)} queries improved")
            print(f"Metric Clusters: {sum(1 for r in metric_results if r['improvement'] > 0)}/{len(metric_results)} queries improved")
            print(f"Scalar Clusters: {sum(1 for r in scalar_results if r['improvement'] > 0)}/{len(scalar_results)} queries improved")
        
        elif choice == '5':
            print("\nThank you for using the Museum Search Engine. Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")
