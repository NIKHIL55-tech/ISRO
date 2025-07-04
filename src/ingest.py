#!/usr/bin/env python3
"""
Website Structure Crawler & Knowledge Graph Builder

This tool crawls websites (static/dynamic) and creates a comprehensive
knowledge graph of the site structure, content, and relationships.
"""

import requests
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
import json
import time
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Set, Dict, List, Optional, Tuple
import re
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import networkx as nx
import matplotlib.pyplot as plt
from textstat import flesch_reading_ease
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import mimetypes
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
import PyPDF2
import pdfplumber
from docx import Document as DocxDocument
import openpyxl
import csv
from PIL import Image
from PIL.ExifTags import TAGS
import magic  # python-magic for file type detection
import feedparser  # For RSS/Atom feeds
import tempfile

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

@dataclass
class PageContent:
    """Represents extracted content from a web page or document"""
    url: str
    title: str
    description: str
    keywords: List[str]
    headings: Dict[str, List[str]]  # h1, h2, h3, etc.
    content_text: str
    word_count: int
    reading_difficulty: float
    images: List[Dict[str, str]]
    links: List[Dict[str, str]]
    forms: List[Dict[str, str]]
    meta_tags: Dict[str, str]
    structured_data: List[Dict]
    content_hash: str
    page_type: str  # home, product, blog, etc.
    content_categories: List[str]
    key_phrases: List[str]
    file_type: str  # html, pdf, xml, docx, etc.
    file_size: Optional[int]  # in bytes
    last_modified: Optional[str]
    document_metadata: Dict[str, str]  # PDF metadata, EXIF data, etc.
    tables: List[List[List[str]]]  # For spreadsheets and structured data
    raw_content: Optional[bytes]  # For binary files

@dataclass 
class SiteStructure:
    """Represents the overall site structure"""
    domain: str
    total_pages: int
    page_hierarchy: Dict[str, List[str]]
    content_types: Dict[str, int]
    file_types: Dict[str, int]  # Distribution of file types found
    internal_links: List[Tuple[str, str]]
    external_links: List[Tuple[str, str]]
    broken_links: List[str]
    downloadable_files: List[Dict[str, str]]  # PDFs, docs, etc.
    sitemaps: List[str]
    robots_txt: Optional[str]
    crawl_stats: Dict[str, int]

class WebsiteCrawler:
    def __init__(self, 
                 start_url: str, 
                 max_pages: int = 100,
                 max_depth: int = 5,
                 delay: float = 1.0,
                 use_selenium: bool = False,
                 respect_robots: bool = True,
                 download_files: bool = True,
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB limit
                 supported_formats: List[str] = None):
        
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.base_url = f"{urlparse(start_url).scheme}://{self.domain}"
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.use_selenium = use_selenium
        self.respect_robots = respect_robots
        self.download_files = download_files
        self.max_file_size = max_file_size
        
        # Supported file formats
        if supported_formats is None:
            self.supported_formats = [
                'html', 'htm', 'xml', 'rss', 'atom', 'sitemap',
                'pdf', 'doc', 'docx', 'txt', 'rtf',
                'xls', 'xlsx', 'csv',
                'ppt', 'pptx',
                'jpg', 'jpeg', 'png', 'gif', 'svg', 'webp',
                'mp3', 'mp4', 'avi', 'mov', 'wmv',
                'zip', 'rar', '7z', 'tar', 'gz'
            ]
        else:
            self.supported_formats = supported_formats
        
        # Crawl state
        self.visited_urls: Set[str] = set()
        self.to_visit: deque = deque([(start_url, 0)])  # (url, depth)
        self.page_contents: Dict[str, PageContent] = {}
        self.broken_links: List[str] = []
        self.external_links: Set[str] = set()
        self.downloadable_files: List[Dict[str, str]] = []
        
        # File processing handlers (commented out until implemented)
        # self.file_processors = {
        #     'pdf': self._process_pdf,
        #     'xml': self._process_xml,
        #     'docx': self._process_docx,
        #     'doc': self._process_doc,
        #     'xlsx': self._process_xlsx,
        #     'xls': self._process_xls,
        #     'csv': self._process_csv,
        #     'txt': self._process_txt,
        #     'rss': self._process_rss,
        #     'atom': self._process_atom,
        #     'jpg': self._process_image,
        #     'jpeg': self._process_image,
        #     'png': self._process_image,
        #     'gif': self._process_image,
        #     'svg': self._process_svg
        # }
        
        # Create temp directory for downloads
        self.temp_dir = tempfile.mkdtemp()
        
        # Robots.txt handling
        self.robots_parser = None
        if respect_robots:
            self._load_robots_txt()
        
        # Selenium setup
        self.driver = None
        if use_selenium:
            self._setup_selenium()
        
        # Content analysis
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_robots_txt(self):
        """Load and parse robots.txt"""
        try:
            robots_url = urljoin(self.base_url, '/robots.txt')
            self.robots_parser = RobotFileParser()
            self.robots_parser.set_url(robots_url)
            self.robots_parser.read()
        except Exception as e:
            self.logger.warning(f"Could not load robots.txt: {e}")

    def _setup_selenium(self):
        """Setup Selenium WebDriver for dynamic content"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-images')
        chrome_options.add_argument('--disable-javascript')  # Can be enabled for SPA
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
        except Exception as e:
            self.logger.error(f"Failed to setup Selenium: {e}")
            self.use_selenium = False

    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        if not self.robots_parser:
            return True
        return self.robots_parser.can_fetch('*', url)

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and sorting query params"""
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        sorted_query = '&'.join(f"{k}={'&'.join(v)}" for k, v in sorted(query_params.items()))
        
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}" + (f"?{sorted_query}" if sorted_query else "")

    def _is_internal_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain"""
        return urlparse(url).netloc == self.domain or urlparse(url).netloc == ''

    def _extract_content_selenium(self, url: str) -> Optional[BeautifulSoup]:
        """Extract content using Selenium for dynamic pages"""
        try:
            self.driver.get(url)
            # Wait for dynamic content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            # Additional wait for AJAX content
            time.sleep(2)
            html = self.driver.page_source
            return BeautifulSoup(html, 'html.parser')
        except Exception as e:
            self.logger.error(f"Selenium extraction failed for {url}: {e}")
            return None

    def _extract_content_requests(self, url: str) -> Optional[BeautifulSoup]:
        """Extract content using requests for static pages"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; WebCrawler/1.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            self.logger.error(f"Request extraction failed for {url}: {e}")
            return None

    def _classify_page_type(self, soup: BeautifulSoup, url: str) -> str:
        """Classify the type of page based on content and URL patterns, with special handling for home/start_url and MOSDAC-specific patterns."""
        url_lower = url.lower()
        base_url = self.start_url.rstrip('/')
        url_no_trailing = url.rstrip('/')

        # 1. Home page: exact match to start_url
        if url_no_trailing == base_url:
            return 'home'

        # 2. MOSDAC-relevant patterns
        if any(p in url_lower for p in ['/faq', '/faqs', '/frequently-asked']):
            return 'faq'
        if any(p in url_lower for p in ['/data', '/archive', '/catalog', '/product', '/download']):
            return 'data'
        if any(p in url_lower for p in ['/register', '/registration', '/signup']):
            return 'registration'
        if any(p in url_lower for p in ['/login', '/signin', '/uops']):
            return 'login'
        if any(p in url_lower for p in ['/logout', '/signout']):
            return 'logout'
        if any(p in url_lower for p in ['/about', '/contact', '/team', '/help']):
            return 'about'
        if any(p in url_lower for p in ['/news', '/updates', '/events']):
            return 'news'
        if any(p in url_lower for p in ['/blog', '/article', '/post']):
            return 'blog'

        # 3. Content-based classification
        title = soup.find('title')
        title_text = title.get_text().lower() if title else ''
        if any(word in title_text for word in ['faq', 'frequently asked','faq-page']):
            return 'faq'
        if any(word in title_text for word in ['data', 'archive', 'catalog', 'product', 'download']):
            return 'data'
        if any(word in title_text for word in ['register', 'registration', 'signup']):
            return 'registration'
        if any(word in title_text for word in ['login', 'sign in', 'uops']):
            return 'login'
        if any(word in title_text for word in ['logout', 'sign out']):
            return 'logout'
        if any(word in title_text for word in ['about', 'contact', 'team', 'help']):
            return 'about'
        if any(word in title_text for word in ['news', 'update', 'event']):
            return 'news'
        if any(word in title_text for word in ['blog', 'article', 'post']):
            return 'blog'

        # 4. Fallback
        return 'general'

    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases using simple NLP techniques"""
        # Tokenize and clean
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word.isalpha() and word not in self.stop_words and len(word) > 2]
        
        # Get most common words
        word_freq = Counter(words)
        common_words = [word for word, _ in word_freq.most_common(max_phrases)]
        
        # Simple phrase extraction (bigrams)
        phrases = []
        for i in range(len(words) - 1):
            if words[i] in common_words and words[i+1] in common_words:
                phrases.append(f"{words[i]} {words[i+1]}")
        
        phrase_freq = Counter(phrases)
        key_phrases = [phrase for phrase, _ in phrase_freq.most_common(max_phrases//2)]
        
        return common_words[:max_phrases//2] + key_phrases

    def _analyze_page_content(self, soup: BeautifulSoup, url: str) -> PageContent:
        """Extract and analyze content from a page"""
        # Basic metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ''
        
        description = soup.find('meta', attrs={'name': 'description'})
        description_text = description.get('content', '') if description else ''
        
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        keywords_list = keywords.get('content', '').split(',') if keywords else []
        keywords_list = [k.strip() for k in keywords_list if k.strip()]
        
        # Headings
        headings = {}
        for i in range(1, 7):
            h_tags = soup.find_all(f'h{i}')
            if h_tags:
                headings[f'h{i}'] = [h.get_text().strip() for h in h_tags]
        
        # Main content
        content_tags = soup.find_all(['p', 'div', 'article', 'section'])
        content_text = ' '.join([tag.get_text().strip() for tag in content_tags])
        content_text = re.sub(r'\s+', ' ', content_text).strip()
        
        # Content metrics
        word_count = len(content_text.split())
        reading_difficulty = flesch_reading_ease(content_text) if content_text else 0
        
        # Images
        images = []
        for img in soup.find_all('img'):
            images.append({
                'src': img.get('src', ''),
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        
        # Links
        links = []
        for link in soup.find_all('a', href=True):
            href = urljoin(url, link['href'])
            links.append({
                'url': href,
                'text': link.get_text().strip(),
                'title': link.get('title', ''),
                'internal': self._is_internal_url(href)
            })
        
        # Forms
        forms = []
        for form in soup.find_all('form'):
            forms.append({
                'action': form.get('action', ''),
                'method': form.get('method', 'get'),
                'inputs': len(form.find_all('input'))
            })
        
        # Meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            if name and content:
                meta_tags[name] = content
        
        # Structured data (JSON-LD)
        structured_data = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data.append(data)
            except:
                pass
        
        # Content hash for deduplication
        content_hash = hashlib.md5(content_text.encode()).hexdigest()
        
        # Page classification
        page_type = self._classify_page_type(soup, url)
        
        # Content categories (simple heuristic)
        content_categories = []
        if any(word in content_text.lower() for word in ['product', 'buy', 'price', 'cart']):
            content_categories.append('commerce')
        if any(word in content_text.lower() for word in ['blog', 'article', 'post', 'news']):
            content_categories.append('editorial')
        if any(word in content_text.lower() for word in ['service', 'solution', 'consulting']):
            content_categories.append('services')
        
        # Key phrases
        key_phrases = self._extract_key_phrases(content_text)
        
        return PageContent(
            url=url,
            title=title_text,
            description=description_text,
            keywords=keywords_list,
            headings=headings,
            content_text=content_text,
            word_count=word_count,
            reading_difficulty=reading_difficulty,
            images=images,
            links=links,
            forms=forms,
            meta_tags=meta_tags,
            structured_data=structured_data,
            content_hash=content_hash,
            page_type=page_type,
            content_categories=content_categories,
            key_phrases=key_phrases,
            file_type='html',
            file_size=None,
            last_modified=None,
            document_metadata={},
            tables=[],
            raw_content=None
        )

    def crawl(self) -> SiteStructure:
        """Main crawling method"""
        self.logger.info(f"Starting crawl of {self.start_url}")
        
        crawl_start_time = time.time()
        pages_crawled = 0
        errors = 0
        
        try:
            while self.to_visit and pages_crawled < self.max_pages:
                url, depth = self.to_visit.popleft()
                
                if url in self.visited_urls or depth > self.max_depth:
                    continue
                
                if self.respect_robots and not self._can_fetch(url):
                    self.logger.info(f"Skipping {url} due to robots.txt")
                    continue
                
                self.logger.info(f"Crawling ({pages_crawled+1}/{self.max_pages}): {url}")
                
                # Extract content
                soup = None
                if self.use_selenium:
                    soup = self._extract_content_selenium(url)
                
                if not soup:  # Fallback to requests
                    soup = self._extract_content_requests(url)
                
                if not soup:
                    self.broken_links.append(url)
                    errors += 1
                    continue
                
                # Analyze page content
                try:
                    page_content = self._analyze_page_content(soup, url)
                    self.page_contents[url] = page_content
                    self.visited_urls.add(url)
                    pages_crawled += 1
                    
                    # Add internal links to crawl queue
                    for link_info in page_content.links:
                        link_url = self._normalize_url(link_info['url'])
                        if (link_info['internal'] and 
                            link_url not in self.visited_urls and 
                            not any(link_url == queued_url for queued_url, _ in self.to_visit)):
                            self.to_visit.append((link_url, depth + 1))
                        elif not link_info['internal']:
                            self.external_links.add(link_url)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {url}: {e}")
                    errors += 1
                
                # Rate limiting
                time.sleep(self.delay)
                
        except KeyboardInterrupt:
            self.logger.info("Crawl interrupted by user")
        
        finally:
            if self.driver:
                self.driver.quit()
        
        crawl_time = time.time() - crawl_start_time
        
        # Build site structure
        site_structure = self._build_site_structure(pages_crawled, errors, crawl_time)
        
        self.logger.info(f"Crawl completed: {pages_crawled} pages in {crawl_time:.2f}s")
        return site_structure

    def _build_site_structure(self, pages_crawled: int, errors: int, crawl_time: float) -> SiteStructure:
        """Build the final site structure object"""
        # Page hierarchy (URL path based)
        page_hierarchy = defaultdict(list)
        for url in self.visited_urls:
            path = urlparse(url).path
            parts = [p for p in path.split('/') if p]
            
            if len(parts) == 0:
                page_hierarchy['root'].append(url)
            else:
                parent = '/'.join(parts[:-1]) if len(parts) > 1 else 'root'
                page_hierarchy[parent].append(url)
        
        # Content types
        content_types = defaultdict(int)
        for page in self.page_contents.values():
            content_types[page.page_type] += 1
        
        # Internal links
        internal_links = []
        for page in self.page_contents.values():
            for link in page.links:
                if link['internal']:
                    internal_links.append((page.url, link['url']))
        
        # External links
        external_links = [(page.url, ext_url) for page in self.page_contents.values() 
                         for link in page.links if not link['internal'] 
                         for ext_url in [link['url']]]
        
        # Crawl statistics
        crawl_stats = {
            'pages_crawled': pages_crawled,
            'errors': errors,
            'crawl_time_seconds': crawl_time,
            'avg_time_per_page': crawl_time / pages_crawled if pages_crawled > 0 else 0,
            'total_internal_links': len(internal_links),
        }

        return SiteStructure(
            domain=self.domain,
            total_pages=len(self.visited_urls),
            page_hierarchy=dict(page_hierarchy),
            content_types=dict(content_types),
            file_types=getattr(self, 'file_types', {}),
            internal_links=internal_links,
            external_links=external_links,
            broken_links=self.broken_links,
            downloadable_files=getattr(self, 'downloadable_files', []),
            sitemaps=[],  # Could be filled with sitemap URLs if available
            robots_txt=None,  # Could be filled with robots.txt content if desired
            crawl_stats=crawl_stats
        )

    def build_knowledge_graph(self) -> nx.DiGraph:
        """Build a NetworkX graph representing the site structure and content relationships"""
        G = nx.DiGraph()
        
        # Add nodes for each page
        for url, content in self.page_contents.items():
            G.add_node(
                url,
                title=content.title,
                page_type=content.page_type,
                word_count=content.word_count,
                reading_difficulty=content.reading_difficulty,
                content_categories=(', '.join(content.content_categories)
                                    if isinstance(content.content_categories, list)
                                    else str(content.content_categories)),
                key_phrases=(', '.join(content.key_phrases)
                             if isinstance(content.key_phrases, list)
                             else str(content.key_phrases))
            )
        
        # Add edges for internal links
        for source_url, target_url in self.site_structure.internal_links if hasattr(self, 'site_structure') else []:
            if source_url in G.nodes and target_url in G.nodes:
                G.add_edge(source_url, target_url, relationship='internal_link')
        
        # Add edges for content similarity (based on key phrases)
        for url1, content1 in self.page_contents.items():
            for url2, content2 in self.page_contents.items():
                if url1 != url2:
                    # Simple similarity based on common key phrases
                    common_phrases = set(content1.key_phrases) & set(content2.key_phrases)
                    if len(common_phrases) >= 2:  # Threshold for similarity
                        G.add_edge(
                            url1, url2,
                            relationship='content_similarity',
                            similarity_score=len(common_phrases),
                            common_phrases=', '.join(common_phrases)
                        )
        
        return G

    def export_results(self, output_format: str = 'json', filename: str = None):
        """Export crawl results in various formats to data/processed/ directory"""
        # Set output directory
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
        os.makedirs(output_dir, exist_ok=True)
        if not filename:
            filename = f"crawl_results_{self.domain.replace('.', '_')}"
        filepath_base = os.path.join(output_dir, filename)

        if output_format.lower() == 'json':
            data = {
                'site_structure': asdict(self.site_structure) if hasattr(self, 'site_structure') else {},
                'pages': {url: asdict(content) for url, content in self.page_contents.items()}
            }
            with open(f"{filepath_base}.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        elif output_format.lower() == 'csv':
            import pandas as pd
            pages_data = []
            for url, content in self.page_contents.items():
                pages_data.append({
                    'url': url,
                    'title': content.title,
                    'page_type': content.page_type,
                    'word_count': content.word_count,
                    'reading_difficulty': content.reading_difficulty,
                    'num_images': len(content.images),
                    'num_links': len(content.links),
                    'content_categories': ', '.join(content.content_categories),
                    'key_phrases': ', '.join(content.key_phrases[:5])  # Top 5
                })
            df = pd.DataFrame(pages_data)
            df.to_csv(f"{filepath_base}.csv", index=False)


    def visualize_site_structure(self, save_plot: bool = True):
        """Create a visualization of the site structure"""
        if not hasattr(self, 'knowledge_graph'):
            self.knowledge_graph = self.build_knowledge_graph()
        
        plt.figure(figsize=(15, 10))
        
        # Create layout
        pos = nx.spring_layout(self.knowledge_graph, k=1, iterations=50)
        
        # Color nodes by page type
        page_types = nx.get_node_attributes(self.knowledge_graph, 'page_type')
        color_map = {'home': 'red', 'blog': 'blue', 'product': 'green', 
                    'about': 'orange', 'general': 'gray'}
        node_colors = [color_map.get(page_types.get(node, 'general'), 'gray') 
                      for node in self.knowledge_graph.nodes()]
        
        # Draw the graph
        nx.draw(self.knowledge_graph, pos, 
                node_color=node_colors,
                node_size=300,
                with_labels=False,
                edge_color='gray',
                alpha=0.7,
                arrows=True)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=page_type)
                         for page_type, color in color_map.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"Site Structure: {self.domain}")
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f"site_structure_{self.domain.replace('.', '_')}.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
def main():
    """Example usage of the WebsiteCrawler"""
    
    # Configure the crawler
    crawler = WebsiteCrawler(
        start_url="https://www.mosdac.gov.in",  # Replace with target website
        max_pages=50,
        max_depth=3,
        delay=1.0,
        use_selenium=False,  # Set to True for dynamic content
        respect_robots=True
    )
    
    # Perform the crawl
    site_structure = crawler.crawl()
    crawler.site_structure = site_structure
    
    # Build knowledge graph
    knowledge_graph = crawler.build_knowledge_graph()
    crawler.knowledge_graph = knowledge_graph
    
    # Print summary
    print(f"\n=== Crawl Summary ===")
    print(f"Domain: {site_structure.domain}")
    print(f"Total pages crawled: {site_structure.total_pages}")
    print(f"Page types found: {site_structure.content_types}")
    print(f"Internal links: {len(site_structure.internal_links)}")
    print(f"External links: {len(site_structure.external_links)}")
    print(f"Broken links: {len(site_structure.broken_links)}")
    
    # Export results
    crawler.export_results('json')
    crawler.export_results('csv')
    
    # Visualize
    crawler.visualize_site_structure()
    
    print(f"\nKnowledge graph created with {knowledge_graph.number_of_nodes()} nodes and {knowledge_graph.number_of_edges()} edges")

if __name__ == "__main__":
    main()