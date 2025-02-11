import streamlit as st
import os
import json
import concurrent.futures
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from pydantic import BaseModel, Field
from typing import List
from metaphor_python import Metaphor
from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc
from eventregistry import *

# Load API keys from Streamlit secrets
st.set_page_config(page_title="News & Research Summarizer", layout="wide")
st.title("🔎 AI-Powered News & Research Summarizer")

# Load secrets (Set these in .streamlit/secrets.toml)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
METAPHOR_API_KEY = st.secrets["METAPHOR_API_KEY"]
ELSEVIER_API_KEY = st.secrets["ELSEVIER_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize APIs
els_client = ElsClient(ELSEVIER_API_KEY)
metaphor_client = Metaphor(api_key=METAPHOR_API_KEY)
er = EventRegistry(apiKey=NEWS_API_KEY)

# User Input
topic = st.text_input("Enter a topic to fetch news & research articles", "Large Language Models")

if st.button("Fetch & Summarize"):
    with st.spinner("Fetching articles..."):
        num_cores = os.cpu_count()

        # max_workers = min(4, num_cores // 2)
        max_workers = 3

        start = time.time()

        topic = "Large language Models"
        q = QueryArticlesIter(
            lang="eng",
            categoryUri=QueryItems.OR(["news/Technology", "events/Technology"]),
            keywords=QueryItems.OR([topic]),
            ignoreKeywords=QueryItems.OR(["NASDAQ", "Stock", "Market", "Investment"]),
        )

        def fetch_news_article(pap):
            return {
                'type': 'NEWS',
                'title': pap.get('title'),
                'date': pap.get('date'),
                'authors': [a.get('name') for a in (pap.get('authors') or []) if a.get('name')],
                'body': pap.get('body'),
                'url': pap.get('url'),
                'source': pap.get('source'),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            news_data = list(executor.map(fetch_news_article, q.execQuery(er, sortBy="date", maxItems=5)))


        def get_metaphor_articles(query, num_results=10, include_domains=['www.sciencedirect.com']):
            """Fetch articles using the Metaphor API and extract PII identifiers."""
            try:
                response = metaphor_client.search(query, num_results=num_results, include_domains=include_domains)
                
                pattern = re.compile(r'pii/(.*?)$')
                return [match.group(1) for result in response.results 
                        if (match := pattern.search(result.url))]
            except Exception as e:
                print(f"Error fetching Metaphor articles: {e}")
                return []

        def fetch_research_paper(pii):
            """Fetch research paper metadata from Elsevier API using PII."""
            try:
                pii_doc = FullDoc(sd_pii=pii)
                if pii_doc.read(els_client):
                    core_data = pii_doc.data['coredata']
                    authors = []

                    if isinstance(core_data.get('dc:creator'), list):  
                        authors = [author.get('$') for author in core_data['dc:creator'] if author.get('$')]
                    elif isinstance(core_data.get('dc:creator'), dict):  
                        authors = [core_data['dc:creator'].get('$')]

                    return {
                        'type': 'Research Paper',
                        'title': core_data.get('dc:title', 'Unknown Title'),
                        'date': core_data.get('prism:coverDate', 'Unknown Date'),
                        'authors': authors,
                        'body': core_data.get('dc:description', 'No Description'),
                        'url': [link.get('@href') for link in core_data.get('link', []) if link.get('@rel') != 'self'],
                        'publication': core_data.get('prism:publicationName', 'Unknown Publication'),
                    }
            except Exception as e:
                print(f"Error fetching research paper for PII {pii}: {e}")
                return None

        output = get_metaphor_articles(topic, 7)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            research_data = list(filter(None, executor.map(fetch_research_paper, output))) 

        full_length_data = research_data + news_data

        def summarize_article(article):
            title = article.get('title')
            text = article.get('body')  
            if not text: 
                return None
            
            class TitleFilter(BaseModel):
                title: str = Field(description = "Title of the summary")
                summary: str = Field(description="Summary of the given data")

            model = ChatOpenAI(model="gpt-4o-mini").bind_tools([TitleFilter])

            system_prompt = "You are an AI system trained on sumarization"
            user_prompt = """
            You are required to summazrize the given data into not more than 4 sentances. The sumary must be created by keeping
            the topic in mind as these articles are fetched with respect to the topic.
            The output must feel natural and should not feel the text was summarized.
            The topic name is {topic}
            title: 
            {title}

            text:
            {text}

            Give the output as JSON containing the title and summary.
            """
            prompt_template = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("user", user_prompt)]
            )
            str_parser = JsonOutputToolsParser()

            chain = prompt_template | model | str_parser

            output = chain.invoke({'topic': topic, 'title': title, 'text': text})
            article['body'] = output[0]['args']['summary']
            return article

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(summarize_article, full_length_data))

        summary_list = [res for res in results if res is not None]
    # Display Results in Streamlit
    st.subheader("📢 Summarized Articles")
    for article in summary_list:
        st.markdown(f"### {article['title']}")
        st.markdown(f"**{article['type']}** | {article.get('date', 'Unknown Date')}")
        if 'authors' in article and article['authors']:
            st.markdown(f"**Authors:** {', '.join(article['authors'])}")
        st.markdown(article['body'])
        if 'url' in article:
            if isinstance(article['url'], list):
                for link in article['url']:
                    st.markdown(f"[Read More]({link})")
            else:
                st.markdown(f"[Read More]({article['url']})")
        st.markdown("---")