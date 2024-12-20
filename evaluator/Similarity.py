from sentence_transformers import SentenceTransformer
            
            
class SimilarityCalculator:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def compute(self, s1,  s2):
        embeddings = self.model.encode([s1, s2])
        return self.model.similarity(embeddings[:1], embeddings[1:])
    
    
    
    