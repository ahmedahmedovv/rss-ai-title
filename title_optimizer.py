from mistralai import Mistral
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

def load_articles():
    """Load articles from data.json file"""
    with open('data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def save_optimized_titles(articles):
    """Save optimized titles to data/optimized_titles.json"""
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / 'optimized_titles.json'
    
    # Add timestamp to track when optimizations were done
    output_data = {
        "optimization_timestamp": datetime.now().isoformat(),
        "articles": articles
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nOptimized titles saved to: {output_path}")

def optimize_title(client, title, description, retry_count=0):
    """Generate an optimized title using Mistral AI with retry logic"""
    prompt = f"""Given this original title: "{title}" 
    and description: "{description}"
    Generate a more engaging title that is:
    - Clear and concise (max 10 words)
    - Engaging but not clickbait
    - Factually accurate
    Return only the new title, nothing else."""

    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip('"')
    except Exception as e:
        if "rate limit" in str(e).lower() and retry_count < 3:
            # Wait for 2 seconds before retrying
            time.sleep(2)
            return optimize_title(client, title, description, retry_count + 1)
        raise e

def main():
    # Get API key from .env file
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env file")
        
    # Initialize Mistral client with API key
    client = Mistral(api_key=api_key)
    
    # Load articles
    articles = load_articles()
    
    print("\n=== Title Optimization Tool ===\n")
    
    # Process each article
    optimized_articles = []
    for i, article in enumerate(articles[:5], 1):  # Limit to first 5 for demo
        original_title = article.get('title', '')
        description = article.get('description', '')
        
        print(f"\nArticle {i}:")
        print(f"Original: {original_title}")
        
        try:
            # Add a small delay between requests
            if i > 1:
                time.sleep(1)
                
            optimized_title = optimize_title(client, original_title, description)
            print(f"Optimized: {optimized_title}")
            print("-" * 50)
            
            # Store optimized article data
            optimized_articles.append({
                "original_title": original_title,
                "optimized_title": optimized_title,
                "description": description,
                "link": article.get('link', ''),
                "published": article.get('published', ''),
                "optimized_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Error optimizing title: {str(e)}")
    
    # Save results
    save_optimized_titles(optimized_articles)

if __name__ == "__main__":
    main()