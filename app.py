import streamlit as st
import os
import json
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

        # Fetch news articles
        q = QueryArticlesIter(
            lang="eng",
            categoryUri=QueryItems.OR(["news/Technology", "events/Technology"]),
            keywords=QueryItems.OR([topic]),
            ignoreKeywords=QueryItems.OR(["NASDAQ", "Stock", "Market", "Investment"]),
        )
        
        news_data_all = []
        for pap in q.execQuery(er, sortBy="date", maxItems=10):
            pap_dict = {
                'type': 'NEWS',
                'title': pap.get('title'),
                'date': pap.get('date'),
                'authors': [author.get('name') for author in (pap.get('authors') or []) if author.get('name')],
                'body': pap.get('body'),
                'url': pap.get('url'),
                'source': pap.get('source'),
            }
            news_data_all.append(pap_dict)

        title_names = '\n'.join([f"{i+1}. " + item['title'] for i, item in enumerate(news_data_all)])

        # Title Filtering with AI
        class TitleFilter(BaseModel):
            items: List[str] = Field(description="List of relevant titles")

        model = ChatOpenAI(model="gpt-4o-mini").bind_tools([TitleFilter])

        system_prompt = "You are an AI trained to filter relevant news articles."
        user_prompt = """
        You need to filter out at least reasonably relevant news article titles related to the topic.
        The topic is: {topic}
        Titles:
        {titles}

        Give output as JSON containing the list of exact titles.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )
        str_parser = JsonOutputToolsParser()

        chain = prompt_template | model | str_parser
        output = chain.invoke({'topic': topic, 'titles': title_names})
        output = [i['args'] for i in output]

        # Filtering News Data
        news_data = [j for j in news_data_all if j['title'] in output[0]['items']]

        # Fetch Research Papers using Metaphor API
        def get_metaphor_articles(query, num_results=5):
            try:
                response = metaphor_client.search(query, num_results=num_results, include_domains=['www.sciencedirect.com'])
                articles = [{"url": result.url} for result in response.results]
                return articles
            except Exception as e:
                return []

        metaphor_articles = get_metaphor_articles(topic)
        research_data = []
        for article in metaphor_articles:
            url = article["url"]
            pii = url.split("pii/")[-1] if "pii/" in url else None
            if pii:
                pii_doc = FullDoc(sd_pii=pii)
                if pii_doc.read(els_client):
                    temp_dict = {
                        'type': 'RESEARCH PAPER',
                        'title': pii_doc.data['coredata']['dc:title'],
                        'date': pii_doc.data['coredata']['prism:coverDate'],
                        'body': pii_doc.data['coredata']['dc:description'],
                        'url': [i.get('@href') for i in pii_doc.data['coredata']['link'] if i.get('@rel') != 'self'],
                        'publication': pii_doc.data['coredata']['prism:publicationName'],
                    }
                    if isinstance(pii_doc.data['coredata']['dc:creator'], list):
                        temp_dict['authors'] = [i.get('$') for i in pii_doc.data['coredata']['dc:creator'] if i.get('$')]
                    else:
                        temp_dict['authors'] = pii_doc.data['coredata']['dc:creator'].get('$') if pii_doc.data['coredata']['dc:creator'] else None



                    research_data.append(temp_dict)

        # Summarization
        class SummaryOutput(BaseModel):
            title: str = Field(description="Title of the summary")
            summary: str = Field(description="Summarized content")

        model = ChatOpenAI(model="gpt-4o-mini").bind_tools([SummaryOutput])

        system_prompt = "You are an AI trained to summarize articles."
        user_prompt = """
        Summarize the following article into at most 4 sentences while keeping the topic in mind.
        The topic is: {topic}
        Title: {title}
        Text: {text}
        
        Provide output as JSON with title and summary.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )
        str_parser = JsonOutputToolsParser()

        chain = prompt_template | model | str_parser

        full_length_data = news_data + research_data
        summary_list = []

        for item in full_length_data:
            title = item.get('title')
            text = item.get('body')
            output = chain.invoke({'topic': topic, 'title': title, 'text': text})
            item['body'] = output[0]['args']['summary']
            summary_list.append(item)

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