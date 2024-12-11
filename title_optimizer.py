from mistralai import Mistral
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import logging

# Load environment variables from .env file
load_dotenv()

def setup_logging():
    """Configure logging for the application"""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_file = log_dir / f'title_optimizer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logging.info('Starting title optimization process')

def load_articles():
    """Load articles from data.json file"""
    with open('data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_existing_optimizations():
    """Load existing optimized titles if they exist"""
    output_path = Path('data') / 'optimized_titles.json'
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Create a dictionary of original_title -> optimized_article for easy lookup
            return {
                article['original_title']: article 
                for article in data.get('articles', [])
            }
    return {}

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
    logging.info(f'Optimizing title: "{title}"')
    
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
        optimized = response.choices[0].message.content.strip('"')
        logging.info(f'Successfully optimized title to: "{optimized}"')
        return optimized
    except Exception as e:
        if "rate limit" in str(e).lower() and retry_count < 3:
            logging.warning(f'Rate limit hit, retrying (attempt {retry_count + 1}/3)')
            time.sleep(2)
            return optimize_title(client, title, description, retry_count + 1)
        logging.error(f'Error optimizing title: {str(e)}')
        raise e

def main():
    setup_logging()
    
    try:
        # Get API key from .env file
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logging.error("MISTRAL_API_KEY not found in .env file")
            raise ValueError("MISTRAL_API_KEY not found in .env file")
        
        # Initialize Mistral client with API key
        client = Mistral(api_key=api_key)
        
        # Load articles and existing optimizations
        articles = load_articles()
        existing_optimizations = load_existing_optimizations()
        
        print("\n=== Title Optimization Tool ===\n")
        
        # Process each article
        optimized_articles = []
        for i, article in enumerate(articles[:5], 1):  # Limit to first 5 for demo
            original_title = article.get('title', '')
            description = article.get('description', '')
            
            print(f"\nArticle {i}:")
            print(f"Original: {original_title}")
            
            # Check if we already have an optimization for this title
            if original_title in existing_optimizations:
                existing_article = existing_optimizations[original_title]
                print(f"Optimized (existing): {existing_article['optimized_title']}")
                print("-" * 50)
                optimized_articles.append(existing_article)
                continue
                
            try:
                # Add a small delay between requests
                if i > 1:
                    time.sleep(1)
                    
                optimized_title = optimize_title(client, original_title, description)
                print(f"Optimized (new): {optimized_title}")
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
        
        logging.info(f'Successfully processed {len(optimized_articles)} articles')
        
    except Exception as e:
        logging.error(f'Fatal error in main process: {str(e)}')
        raise

if __name__ == "__main__":
    main()
