import os
from dotenv import load_dotenv
import openai
import numpy as np
from typing import List, Dict, Any
from data_processor import PropertyProcessor
from pdf_processor import PDFProcessor
from faiss_storage import FAISSStorage

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class RAGSystem:
    def __init__(self, csv_path: str, pdf_path: str):
        self.csv_path = csv_path
        self.pdf_path = pdf_path
        self.property_processor = PropertyProcessor(csv_path)
        self.pdf_processor = PDFProcessor(pdf_path)
        self.faiss_storage = FAISSStorage()
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI's text-embedding-3-small."""
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response['data'][0]['embedding'])
    
    def build_knowledge_base(self):
        """Build FAISS index with both CSV and PDF data."""
        # Process CSV data
        df = self.property_processor.preprocess_csv()
        property_embeddings = []
        property_metadata = []
        
        for _, row in df.iterrows():
            # Create text representation for property
            property_text = f"Property ID: {row['Property ID']}\n" \
                          f"Location: {row['Location']}\n" \
                          f"BHK: {row['BHK']}\n" \
                          f"Price: {row['Start Price']}\n" \
                          f"Amenities: {', '.join(row['Amenities'])}\n" \
                          f"Nearby: {', '.join(row['Nearby'])}"
            
            embedding = self.generate_embedding(property_text)
            property_embeddings.append(embedding)
            property_metadata.append({
                'source': 'CSV',
                'type': 'property',
                'metadata': self.property_processor.create_property_metadata(row)
            })
        
        # Process PDF data
        pdf_chunks = self.pdf_processor.extract_text_chunks()
        pdf_embeddings = []
        pdf_metadata = []
        
        for chunk in pdf_chunks:
            embedding = self.generate_embedding(chunk['text'])
            pdf_embeddings.append(embedding)
            pdf_metadata.append({
                'source': 'PDF',
                'type': 'guideline',
                'text': chunk['text'],
                'metadata': chunk['metadata']
            })
        
        # Add all embeddings to FAISS
        all_embeddings = property_embeddings + pdf_embeddings
        all_metadata = property_metadata + pdf_metadata
        
        self.faiss_storage.add_embeddings(all_embeddings, all_metadata)
        self.faiss_storage.save('knowledge_base.index')
    
    def query(self, user_query: str) -> str:
        """
        Process user query and generate response.
        
        Args:
            user_query: User's natural language query
            
        Returns:
            Generated response with source information
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(user_query)
        
        # Retrieve top-k relevant chunks
        results = self.faiss_storage.search(query_embedding, k=20)  # Increased to 20 for better context
        
        # Prepare context for GPT
        context = []
        csv_properties = []  # Store CSV properties separately
        
        # Extract query keywords for better matching
        query_keywords = set(word.lower() for word in user_query.split())
        
        for _, metadata in results:
            try:
                if metadata['source'] == 'CSV':
                    prop_data = metadata['metadata']
                    csv_properties.append({
                        'id': prop_data['property_id'],
                        'data': prop_data
                    })
                else:
                    # For PDF chunks, get the text and metadata
                    text = metadata.get('text', '')
                    if not text:
                        continue
                        
                    # Get section information
                    section = metadata['metadata'][0].get('section', '')
                    subsection = metadata['metadata'][0].get('subsection', '')
                    
                    # Check if this chunk is relevant based on keywords
                    chunk_keywords = set(word.lower() for word in text.split())
                    if any(keyword in chunk_keywords for keyword in query_keywords):
                        # Format context with section information
                        context.append(f"From PDF (Page {metadata['metadata'][0]['page']}, " \
                                     f"Section: {section}, Subsection: {subsection}):\n" \
                                     f"{text}\n")
            except KeyError as e:
                print(f"Warning: Missing key in metadata: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing chunk: {e}")
                continue
        
        # If we have CSV properties, filter them based on query
        if csv_properties:
            query_lower = user_query.lower()
            filtered_properties = []
            for prop in csv_properties:
                prop_data = prop['data']
                prop_str = f"{prop_data['location']} {prop_data['bhk']} {prop_data['price']} " \
                          f"{' '.join(prop_data['amenities'])} {' '.join(prop_data['nearby'])}".lower()
                if any(keyword in prop_str for keyword in query_lower.split()):
                    filtered_properties.append(prop)
            
            for prop in filtered_properties:
                prop_data = prop['data']
                context.append(f"Property ID: {prop_data['property_id']}\n" \
                             f"Location: {prop_data['location']}\n" \
                             f"BHK: {prop_data['bhk']}\n" \
                             f"Price: {prop_data['price']}\n" \
                             f"Amenities: {', '.join(prop_data['amenities'])}\n" \
                             f"Nearby: {', '.join(prop_data['nearby'])}")
        
        # Create prompt for GPT
        if not context:
            return "I couldn't find any relevant information in the database for your query."
            
        prompt = f"""You are a real estate assistant. Here is some context from our community guidelines:

{chr(10).join(context)}

User query: {user_query}

Please provide a clear and structured response based on the context. If the query is about community guidelines, cite the relevant sections and subsections."""
        
        try:
            # Generate response using GPT-3.5-turbo
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful real estate assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content']
        except openai.error.OpenAIError as e:
            return f"Error generating response: {str(e)}"
        
        return response['choices'][0]['message']['content']

def main():
    # Example usage
    system = RAGSystem(
        csv_path='properties.csv',
        pdf_path='guidelines.pdf'
    )
    
    # Build knowledge base (only needed once)
    system.build_knowledge_base()
    
    # CLI loop
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        response = system.query(query)
        print("\nResponse:")
        print(response)
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()
