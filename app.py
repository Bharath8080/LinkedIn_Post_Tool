import streamlit as st
import requests
import json
import os
from datetime import datetime
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from langchain_openai import ChatOpenAI
from composio_langchain import ComposioToolSet
from dotenv import load_dotenv
import time
from langchain.schema import HumanMessage
from langchain.callbacks.base import BaseCallbackHandler

# Load environment variables
load_dotenv()

# Define supported languages
languages = [
    "English", "Hindi", "Gujarati", "Bengali", "Tamil", 
    "Telugu", "Kannada", "Malayalam", "Punjabi", "Marathi", 
    "Urdu", "Assamese", "Odia", "Sanskrit", "Korean", 
    "Japanese", "Arabic", "French", "German", "Spanish", 
    "Portuguese", "Russian", "Chinese", "Vietnamese", "Thai", 
    "Indonesian", "Turkish", "Polish", "Ukrainian", "Dutch", 
    "Italian", "Greek", "Hebrew", "Persian", "Swedish", 
    "Norwegian", "Danish", "Finnish", "Czech", "Hungarian", 
    "Romanian", "Bulgarian", "Croatian", "Serbian", "Slovak", 
    "Slovenian", "Estonian", "Latvian", "Lithuanian", "Malay", 
    "Tagalog", "Swahili"
]

# Streaming callback handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# Initialize the ChatOpenAI model - base instance for caching
@st.cache_resource
def get_base_chat_model(api_key):
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.two.ai/v2",
        model="sutra-v2",
        temperature=0.7,
    )

# Create a streaming version of the model with callback handler
def get_streaming_chat_model(api_key, callback_handler=None):
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.two.ai/v2",
        model="sutra-v2",
        temperature=0.7,
        streaming=True,
        callbacks=[callback_handler] if callback_handler else None
    )

# Function to translate content using Sutra
def translate_content(content, target_language, api_key):
    """Translate content to target language using Sutra"""
    if not api_key:
        return None, "Sutra API key is required"
    
    try:
        chat = get_base_chat_model(api_key)
        system_message = f"""You are a professional translator. Translate the following LinkedIn post to {target_language}.
        Important guidelines:
        1. Maintain the professional tone and style
        2. Keep all hashtags and translate them appropriately
        3. Preserve all formatting and line breaks
        4. Keep emojis that make sense in {target_language}
        5. Ensure the translation is natural and engaging
        6. Maintain the same length and structure
        7. Keep any technical terms or brand names unchanged
        8. Ensure the call-to-action is culturally appropriate
        
        Post to translate:
        {content}
        """
        messages = [HumanMessage(content=system_message)]
        response = chat.invoke(messages)
        return response.content, None
    except Exception as e:
        return None, f"Translation failed: {str(e)}"

# Function to search web content using Serper
@st.cache_data(ttl=3600)  # Cache for 1 hour
def search_web_content(query, search_type="search", api_key="", num_results=20):
    """Search web content using Serper API"""
    if not api_key:
        return None, "Serper API key is required"
    
    url = f"https://google.serper.dev/{search_type}"
    payload = json.dumps({
        "q": query,
        "num": num_results
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as e:
        return None, f"Search failed: {str(e)}"

# Function to clean text content
def clean_text_content(text):
    """Clean text content by removing unwanted characters and formatting"""
    if not text:
        return text
    
    # Remove parentheses and their contents
    import re
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove any remaining parentheses
    text = text.replace('(', '').replace(')', '')
    
    return text.strip()

# Function to generate LinkedIn post using GPT-4
def generate_linkedin_post(content_data, custom_prompt="", api_key="", target_language="English"):
    """Generate LinkedIn post using OpenAI GPT-4"""
    if not api_key:
        return None, "OpenAI API key is required"
    
    # Clean target language
    target_language = target_language.strip()
    
    # Extract relevant information from search results
    content_summary = ""
    links = []
    
    if 'organic' in content_data:
        for item in content_data['organic'][:5]:  # Use top 5 results
            title = clean_text_content(item.get('title', ''))
            snippet = clean_text_content(item.get('snippet', ''))
            content_summary += f"• {title}: {snippet}\n"
            if item.get('link'):
                links.append(item['link'])
    
    if 'news' in content_data:
        for item in content_data['news'][:3]:  # Use top 3 news items
            title = clean_text_content(item.get('title', ''))
            snippet = clean_text_content(item.get('snippet', ''))
            content_summary += f"• {title}: {snippet}\n"
            if item.get('link'):
                links.append(item['link'])
    
    # LinkedIn post generation prompt
    base_prompt = f"""
    Create an engaging LinkedIn post in {target_language} based on the following content. The post should:
    
    1. Be professional yet conversational
    2. Include relevant hashtags (3-5)
    3. Have a compelling hook in the first line
    4. Be between 100-300 words
    5. Include a call-to-action
    6. Use emojis strategically (6-8 maximum)
    7. Structure with short paragraphs for readability
    8. Ensure all text is properly formatted for {target_language}
    9. DO NOT use parentheses () in the text
    10. Use clear, direct language without parenthetical expressions
    
    Content to base the post on:
    {content_summary}
    
    Relevant links to potentially reference:
    {chr(10).join(links[:3])}
    
    {custom_prompt if custom_prompt else ''}
    
    Generate a LinkedIn post that will engage professional audiences and encourage interaction.
    """
    
    try:
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=api_key
        )
        response = llm.invoke(base_prompt)
        # Clean the generated content
        cleaned_content = clean_text_content(response.content)
        return cleaned_content, None
    except Exception as e:
        return None, f"Post generation failed: {str(e)}"

# Function to upload to LinkedIn using Composio
def upload_to_linkedin(post_content, author_urn, api_key):
    """Upload post to LinkedIn using Composio"""
    if not api_key:
        return None, "Composio API key is required"
    
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        prompt = hub.pull("hwchase17/openai-functions-agent")
        composio_toolset = ComposioToolSet(api_key=api_key)
        tools = composio_toolset.get_tools(actions=["LINKEDIN_CREATE_LINKED_IN_POST"])
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        
        # Task configuration
        task = f"""
        Create a LinkedIn post by author {author_urn} 
        and set visibility to PUBLIC, lifecycleState to PUBLISHED, and resharing enabled.
        
        Post content: '{post_content}'
        """
        
        result = agent_executor.invoke({"input": task})
        return result, None
    except Exception as e:
        return None, f"LinkedIn upload failed: {str(e)}"

# Page configuration
st.set_page_config(
    page_title="Multilingual LinkedIn Post Generator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #0077B5;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2E8B57;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #0077B5;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0077B5;
        margin: 1rem 0;
    }
    .generated-post {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_posts' not in st.session_state:
    st.session_state.generated_posts = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# API Keys section
st.sidebar.subheader("🔑 API Keys")
serper_api_key = st.sidebar.text_input("Serper API Key", type="password", 
                                       value=os.getenv("SERPER_API_KEY", ""))
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                                      value=os.getenv("OPENAI_API_KEY", ""))
composio_api_key = st.sidebar.text_input("Composio API Key", type="password", 
                                         value=os.getenv("COMPOSIO_API_KEY", ""))
sutra_api_key = st.sidebar.text_input("Sutra API Key", type="password",
                                     value=os.getenv("SUTRA_API_KEY", ""))

# LinkedIn configuration
st.sidebar.subheader("👤 LinkedIn Configuration")
linkedin_author_urn = st.sidebar.text_input("LinkedIn Author URN", 
                                           value=os.getenv("LINKEDIN_AUTHOR_URN", ""))

# Language selection
st.sidebar.subheader("🌐 Language Settings")
selected_language = st.sidebar.selectbox("Select post language:", languages)

# Search configuration
st.sidebar.subheader("🔍 Search Settings")
search_type = st.sidebar.selectbox("Search Type", ["search", "news"])
num_results = st.sidebar.slider("Number of Results", 5, 20, 10)

# Main content
st.markdown('<h1><img src="https://framerusercontent.com/images/9vH8BcjXKRcC5OrSfkohhSyDgX0.png" width="70" height="70" style="vertical-align: middle;"> Multilingual LinkedIn Post Tool <img src="https://cliply.co/wp-content/uploads/2021/02/372102050_LINKEDIN_ICON_TRANSPARENT_1080.gif" width="70" height="70" style="vertical-align: middle;"></h1>', unsafe_allow_html=True)
st.markdown("### Create engaging LinkedIn posts in multiple languages")


# Search query input
search_query = st.text_input(
    "Enter your search query:",
    placeholder="e.g., Agentic AI Frameworks, Latest AI trends, etc.",
    help="This will be used to search for relevant content to base your LinkedIn post on"
)

# Custom prompt for post generation
custom_prompt = st.text_area(
    "Custom instructions for post generation (optional):",
    placeholder="e.g., Focus on business applications, include statistics, make it technical, etc.",
    height=80
)

# Search button
if st.button("🔍 Search & Generate Posts", type="primary"):
    if not search_query:
        st.error("Please enter a search query")
    elif not serper_api_key:
        st.error("Please provide Serper API key in the sidebar")
    elif not openai_api_key:
        st.error("Please provide OpenAI API key in the sidebar")
    else:
        with st.spinner("🔍 Searching web content..."):
            # Search web content
            search_results, search_error = search_web_content(
                search_query, search_type, serper_api_key, num_results
            )
            
            if search_error:
                st.error(f"Search failed: {search_error}")
            else:
                st.session_state.search_results = search_results
                
                with st.spinner("🤖 Generating LinkedIn posts..."):
                    # Generate multiple post variations
                    post_variations = []
                    
                    for i in range(3):  # Generate 3 variations
                        variation_prompt = clean_text_content(custom_prompt)
                        if i == 1:
                            variation_prompt += "\n\nMake this version more casual and story-driven."
                        elif i == 2:
                            variation_prompt += "\n\nMake this version more data-driven and professional."
                        
                        # Generate post in selected language
                        post_content, post_error = generate_linkedin_post(
                            search_results, 
                            variation_prompt, 
                            openai_api_key,
                            clean_text_content(selected_language)  # Clean the selected language
                        )
                        
                        if post_content:
                            post_variations.append({
                                'content': post_content,
                                'timestamp': datetime.now(),
                                'variation': f"Variation {i+1}",
                                'language': clean_text_content(selected_language)  # Clean the language
                            })
                    
                    st.session_state.generated_posts = post_variations
                    
                    if post_variations:
                        st.success(f"✅ Generated {len(post_variations)} post variations in {clean_text_content(selected_language)}!")
                    else:
                        st.error("Failed to generate posts. Please check your API keys and try again.")

# Search Results Section
if st.session_state.search_results:
    st.markdown('<h2 class="section-header">📊 Search Results</h2>', 
               unsafe_allow_html=True)
    
    results = st.session_state.search_results
    
    # Display search statistics
    col1, col2 = st.columns(2)
    with col1:
        if 'organic' in results:
            st.metric("Organic Results", len(results['organic']))
    with col2:
        if 'news' in results:
            st.metric("News Results", len(results['news']))
    
    # Display top results
    if 'organic' in results:
        st.write("**Top Organic Results:**")
        for i, item in enumerate(results['organic'][:3], 1):
            st.write(f"{i}. **{item.get('title', 'No title')}**")
            st.write(f"_{item.get('snippet', 'No snippet')}_")
            if item.get('link'):
                st.write(f"🔗 [Read more]({item['link']})")
            st.write("---")
    
    if 'news' in results:
        st.write("**Latest News:**")
        for i, item in enumerate(results['news'][:3], 1):
            st.write(f"{i}. **{item.get('title', 'No title')}**")
            st.write(f"_{item.get('snippet', 'No snippet')}_")
            if item.get('date'):
                st.write(f"📅 {item['date']}")
            if item.get('link'):
                st.write(f"🔗 [Read more]({item['link']})")
            st.write("---")

# Generated posts section
if st.session_state.generated_posts:
    st.markdown('<h2 class="section-header">📝 Generated LinkedIn Posts</h2>', 
               unsafe_allow_html=True)
    
    for i, post_data in enumerate(st.session_state.generated_posts):
        # Clean up language display
        language_display = clean_text_content(post_data['language'])
        with st.expander(f"📄 {post_data['variation']} - {language_display}", expanded=i==0):
            # Editable post content
            edited_content = st.text_area(
                f"Edit {post_data['variation']}:",
                value=post_data['content'],
                height=200,
                key=f"post_edit_{i}"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button(f"📋 Copy", key=f"copy_{i}"):
                    cleaned_content = clean_text_content(edited_content)
                    st.code(cleaned_content, language=None)
                    st.success("✅ Post copied! You can now paste it anywhere.")
            
            with col2:
                word_count = len(edited_content.split())
                st.metric("Word Count", word_count)
                if word_count > 300:
                    st.warning("⚠️ Post might be too long")
                elif word_count < 50:
                    st.warning("⚠️ Post might be too short")
            
            with col3:
                if st.button(f"🚀 Upload to LinkedIn", key=f"upload_{i}", type="primary"):
                    if not composio_api_key:
                        st.error("Please provide Composio API key in the sidebar")
                    elif not linkedin_author_urn:
                        st.error("Please provide LinkedIn Author URN in the sidebar")
                    else:
                        # Clean the content before uploading
                        cleaned_content = clean_text_content(edited_content)
                        
                        with st.spinner("📤 Uploading to LinkedIn..."):
                            result, upload_error = upload_to_linkedin(
                                cleaned_content, linkedin_author_urn, composio_api_key
                            )
                            
                            if upload_error:
                                st.error(f"Upload failed: {upload_error}")
                            else:
                                st.markdown(f"""
                                <div class="success-box">
                                    <strong>🎉 Successfully uploaded to LinkedIn!</strong><br>
                                    Your post is now live on your LinkedIn profile in {language_display}.
                                </div>
                                """, unsafe_allow_html=True)

# Footer with usage statistics
if st.session_state.generated_posts or st.session_state.search_results:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔍 Searches Performed", 
                 1 if st.session_state.search_results else 0)
    
    with col2:
        st.metric("📝 Posts Generated", 
                 len(st.session_state.generated_posts))
    
    with col3:
        if st.button("🗑️ Clear All Data"):
            st.session_state.generated_posts = []
            st.session_state.search_results = []
            st.rerun()
