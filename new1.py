import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
from deep_translator import GoogleTranslator
from translate import Translator as OfflineTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langdetect import detect, DetectorFactory
import base64
from PIL import Image
import json
from geopy.geocoders import Nominatim 
import folium 
from streamlit_folium import folium_static 


DetectorFactory.seed = 0

# Import speech recognition and text-to-speech libraries
try:
    import speech_recognition as sr
    from gtts import gTTS
    from io import BytesIO
    import base64
    VOICE_FEATURES_AVAILABLE = True
except ImportError:
    VOICE_FEATURES_AVAILABLE = False

# Import Google API key from config
from config import GOOGLE_API_KEY

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Set the Streamlit page configuration and theme
st.set_page_config(page_title="", layout="wide")

# Define custom CSS for a more modern interface
def apply_custom_css():
    st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --background-color: #f9f9f9;
            --text-color: #333;
        }
        
        /* Body and background */
        body {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: var(--primary-color);
            font-weight: 600;
        }
        
        /* Main header with animation */
        .main-header {
            background: linear-gradient(45deg, #2c3e50, #3498db);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            animation: gradient 5s ease infinite;
            background-size: 200% 200%;
        }
        
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        
        /* Chat message styling */
        .user-message {
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .assistant-message {
            background-color: #f1f8e9;
            border-left: 5px solid #689f38;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        /* Button styling */
        .stButton button {
            background-color: var(--secondary-color);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: var(--primary-color);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transform: translateY(-2px);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f5f5f5;
            border-right: 1px solid #ddd;
        }
        
        /* Feature cards */
        .feature-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            margin-bottom: 15px;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Voice button */
        .voice-button {
            background-color: var(--accent-color);
            color: white;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            font-size: 24px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .voice-button:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: var(--secondary-color);
        }
        
        /* Hide hamburger menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Animated typing indicator */
        .typing-indicator {
            display: inline-block;
            position: relative;
        }
        
        .typing-indicator::after {
            content: '';
            position: absolute;
            width: 6px;
            height: 15px;
            background-color: var(--text-color);
            right: -12px;
            top: 2px;
            animation: blink 0.7s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
        
        /* Language selector styling */
        .language-selector {
            background-color: white;
            border-radius: 8px;
            padding: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        
        /* Input box styling */
        .chat-input {
            border-radius: 20px;
            border: 2px solid #e0e0e0;
            padding: 10px 15px;
            margin-top: 10px;
            transition: all 0.3s ease;
        }
        
        .chat-input:focus {
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Custom header
st.markdown('<div class="main-header"><h1> Rights Assistance for Youth and Society</h1><p>Your AI-powered legal assistant for accessible justice</p></div>', unsafe_allow_html=True)

# Sidebar configuration with improved styling
with st.sidebar:
    st.title("R.A.Y.S")
    col1, col2, col3 = st.columns([1, 30, 1])

    st.markdown("""
    <div class="feature-card">
        <h3>🌟 About RAYS</h3>
        <p>RAYS is designed to make legal assistance accessible to all, especially in rural and underserved communities. 
        Get accurate legal information in your preferred language through text or voice interaction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    model_mode = st.toggle("Online Mode", value=True)
    
    # Enhanced language selection with flags
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    selected_language = st.selectbox("Select your preferred language", 
                                     ["English", "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", 
                                      "Nepali", "Odia", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Voice features toggle
    if VOICE_FEATURES_AVAILABLE:
        voice_enabled = st.toggle("Enable Voice Interaction", value=True)
        if voice_enabled:
            st.markdown('<p>🎙️ Voice interaction is enabled. Click the microphone button to speak your query.</p>', unsafe_allow_html=True)
    else:
        voice_enabled = False
        st.warning("Voice libraries not installed. Run: pip install SpeechRecognition gtts")
    
    # Legal resources section
    st.markdown("""
    <div class="feature-card">
        <h3>📚 Legal Resources</h3>
        <ul>
            <li><a href="https://www.legalservicesindia.com/">Legal Services India</a></li>
            <li><a href="https://nalsa.gov.in/">National Legal Services Authority</a></li>
            <li><a href="https://doj.gov.in/">Department of Justice</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency contacts
    st.markdown("""
    <div class="feature-card">
        <h3>🆘 Emergency Contacts</h3>
        <p><strong>National Legal Aid Helpline:</strong> 15100</p>
        <p><strong>Women Helpline:</strong> 1091</p>
        <p><strong>Child Helpline:</strong> 1098</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

if "legal_centers" not in st.session_state:
    # Sample data for legal aid centers
    st.session_state.legal_centers = [
        {"name": "Delhi Legal Aid Center", "lat": 28.6139, "lon": 77.2090, "contact": "011-23385745"},
        {"name": "Mumbai Legal Services", "lat": 19.0760, "lon": 72.8777, "contact": "022-22621366"},
        {"name": "Chennai Legal Assistance", "lat": 13.0827, "lon": 80.2707, "contact": "044-25301629"},
        {"name": "Kolkata Legal Aid Society", "lat": 22.5726, "lon": 88.3639, "contact": "033-22488041"},
        {"name": "Bangalore Legal Help Center", "lat": 12.9716, "lon": 77.5946, "contact": "080-22266026"}
    ]

# Function to get user's location
def get_user_location(place_name):
    try:
        geolocator = Nominatim(user_agent="Rays_legal_assistant")
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude
        return None
    except:
        return None

# Function to create map with nearby legal aid centers
def create_legal_aid_map(user_lat, user_lon):
    m = folium.Map(location=[user_lat, user_lon], zoom_start=10)
    
    # Add user marker
    folium.Marker(
        location=[user_lat, user_lon],
        popup="Your Location",
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(m)
    
    # Add markers for legal aid centers
    for center in st.session_state.legal_centers:
        folium.Marker(
            location=[center["lat"], center["lon"]],
            popup=f"<b>{center['name']}</b><br>Contact: {center['contact']}",
            icon=folium.Icon(color="green", icon="balance-scale", prefix="fa"),
        ).add_to(m)
    
    return m

# Function to play text-to-speech audio
def text_to_speech(text, lang_code="en"):
    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Punjabi": "pa",
        "Urdu": "ur",
        "Assamese": "as",
        "Odia": "or",
        "Nepali": "ne",
        "Sindhi": "sd"
    }
    
    try:
        tts = gTTS(text=text, lang=language_codes.get(selected_language, "en"), slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio_base64 = base64.b64encode(fp.read()).decode()
        audio_html = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")

# Function to listen to voice input
def listen_for_voice():
    if not VOICE_FEATURES_AVAILABLE:
        return None

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.markdown("<p>Listening... Please speak your query.</p>", unsafe_allow_html=True)
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)
        
    try:
        language_codes = {
            "English": "en-US",
            "Hindi": "hi-IN",
            "Tamil": "ta-IN",
            "Telugu": "te-IN",
            "Kannada": "kn-IN", 
            "Malayalam": "ml-IN",
            "Bengali": "bn-IN",
            "Gujarati": "gu-IN",
            "Marathi": "mr-IN", 
            "Punjabi": "pa-IN",
            "Urdu": "ur-IN"
        }
        
        lang_code = language_codes.get(selected_language, "en-US")
        text = recognizer.recognize_google(audio, language=lang_code)
        return text
    except sr.UnknownValueError:
        st.warning("Sorry, I couldn't understand what you said.")
    except sr.RequestError:
        st.error("Could not request results; check your network connection")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    return None

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Load and process your text data (Replace this with your actual legal text data)
text_data = """
[Your legal text data here]
"""

text_chunks = get_text_chunks(text_data)
vector_store = get_vector_store(text_chunks)
db_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create tabs for different functionalities
tabs = st.tabs(["💬 Chat Assistant", "🗺️ Legal Aid Locator", "📊 Legal Awareness", "❓ FAQ" ,  "📄 Document Verification"])

with tabs[0]:  # Chat Assistant Tab
    # Create two columns for chat and voice input
    chat_col, voice_col = st.columns([5, 1])

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {message["content"]}</div>', unsafe_allow_html=True)

    
    
    def get_response_online(prompt, context):
        full_prompt = f"""
        As a legal chatbot specializing in the Indian Penal Code and Department of Justice services, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
        - Respond in a bullet-point format to clearly delineate distinct aspects of the legal query or service information.
        - Each point should accurately reflect the breadth of the legal provision or service in question, avoiding over-specificity unless directly relevant to the user's query.
        - Clarify the general applicability of the legal rules, sections, or services mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
        - Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
        - When asked about live streaming of court cases, provide the relevant links for court live streams.
        - For queries about various DoJ services or information, provide accurate links and guidance.
        - Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations or service information unless otherwise specified.
        - Conclude with a brief summary that captures the essence of the legal discussion or service information and corrects any common misinterpretations related to the topic.
        - When providing legal information, also mention if free legal aid may be available for the situation.
        - If asked about legal aid centers, mention that users can check the Legal Aid Locator tab.

        CONTEXT: {context}
        QUESTION: {prompt}
        ANSWER:
        """
        response = model.generate_content(full_prompt, stream=True)
        return response

    def get_response_offline(prompt, context):
        llm = ChatOllama(model="phi3")
        # Implement offline response generation here
        # This is a placeholder and needs to be implemented based on your offline requirements
        return "Offline mode is not fully implemented yet."

  
    def translate_answer(answer, target_language):
        try:
        # Attempt online translation
          translated_answer = GoogleTranslator(source='auto', target=target_language.lower()).translate(answer)
          return translated_answer
        except Exception as e:
           st.warning("Online translation failed, attempting offline translation.")
        try:
            # Attempt offline translation
            offline_translator = OfflineTranslator(to_lang=target_language.lower())
            translated_answer = offline_translator.translate(answer)
            return translated_answer
        except Exception as e:
            st.error(f"Offline translation failed: {str(e)}")
            return answer 

    def reset_conversation():
        st.session_state.messages = []
        st.session_state.memory.clear()

    def get_trimmed_chat_history():
        max_history = 10
        return st.session_state.messages[-max_history:]

    # Display messages with improved styling
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {message["content"]}</div>', unsafe_allow_html=True)

    # Voice input button in the voice column
    with voice_col:
        if VOICE_FEATURES_AVAILABLE and voice_enabled:
            if st.button("🎤", help="Click to speak your query"):
                with st.spinner("Listening..."):
                    voice_input = listen_for_voice()
                    if voice_input:
                        st.session_state.voice_input = voice_input

    # Handle user input (either from text or voice)
    input_prompt = None
    
    # Check if there's voice input in session state
    if hasattr(st.session_state, 'voice_input') and st.session_state.voice_input:
        input_prompt = st.session_state.voice_input
        st.session_state.voice_input = None  # Clear after use
    else:
        # Regular text input
        input_prompt = st.chat_input("Start with your legal query", key="chat_input")
    
    if input_prompt:
        st.markdown(f'<div class="user-message"><strong>You:</strong> {input_prompt}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": input_prompt})
        trimmed_history = get_trimmed_chat_history()

        with st.spinner("Thinking 💡..."):
            context = db_retriever.get_relevant_documents(input_prompt)
            context_text = "\n".join([doc.page_content for doc in context])
            
            if model_mode:
                response = get_response_online(input_prompt, context_text)
            else:
                response = get_response_offline(input_prompt, context_text)

            message_placeholder = st.empty()
            full_response = "⚠️ *Gentle reminder: We generally ensure precise information, but do double-check.* \n\n\n"
            
            if model_mode:
                for chunk in response:
                    full_response += chunk.text
                    time.sleep(0.02)  # Adjust the sleep time to control the "typing" speed
                    message_placeholder.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {full_response}</div>', unsafe_allow_html=True)
            else:
                full_response += response
                message_placeholder.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {full_response}</div>', unsafe_allow_html=True)

            # Translate the answer to the selected language
            if selected_language != "English":
                with st.spinner(f"Translating to {selected_language}..."):
                    translated_answer = translate_answer(full_response, selected_language.lower())
                    message_placeholder.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {translated_answer}</div>', unsafe_allow_html=True)
                    
                    # Play TTS for the translated response if voice is enabled
                    if VOICE_FEATURES_AVAILABLE and voice_enabled:
                        text_to_speech(translated_answer, selected_language.lower())
            else:
                # Play TTS for English response if voice is enabled
                if VOICE_FEATURES_AVAILABLE and voice_enabled:
                    text_to_speech(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Reset button
    if st.button('🗑️ Reset Conversation', on_click=reset_conversation):
        st.rerun()

with tabs[1]:  # Legal Aid Locator Tab
    st.markdown("""
    <div class="feature-card">
        <h2>🗺️ Find Legal Aid Centers Near You</h2>
        <p>Enter your location to find legal aid centers and free legal services available in your area.</p>
    </div>
    """, unsafe_allow_html=True)
    
    location_input = st.text_input("Enter your city or area", "Delhi")
    if st.button("Find Legal Aid Centers"):
        with st.spinner("Locating centers..."):
            user_location = get_user_location(location_input)
            if user_location:
                user_lat, user_lon = user_location
                legal_map = create_legal_aid_map(user_lat, user_lon)
                st.markdown("<h3>Legal Aid Centers Near You</h3>", unsafe_allow_html=True)
                folium_static(legal_map)
                
                st.markdown("""
                <div class="feature-card">
                    <h3>Free Legal Aid Eligibility</h3>
                    <p>In India, free legal services are available to:</p>
                    <ul>
                        <li>Women and children</li>
                        <li>Victims of trafficking</li>
                        <li>Persons with disabilities</li>
                        <li>Victims of mass disaster, ethnic violence, caste atrocity, flood, drought, earthquake, industrial disaster</li>
                        <li>Industrial workmen</li>
                        <li>Persons in custody</li>
                        <li>Persons with annual income less than Rs. 1,00,000 (may vary by state)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Location not found. Please try a different city or check your spelling.")

with tabs[2]:  # Legal Awareness Tab
    st.markdown("""
    <div class="feature-card">
        <h2>📊 Legal Awareness</h2>
        <p>Educational resources to understand your legal rights and responsibilities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    legal_topics = [
        "Women's Rights", 
        "Property Laws", 
        "Consumer Protection", 
        "Labor Laws", 
        "Right to Information"
    ]
    
    selected_topic = st.selectbox("Select a topic to learn about", legal_topics)
    
    # Display information based on selected topic
    topic_content = {
        "Women's Rights": """
        <div class="feature-card">
            <h3>📝 Women's Legal Rights in India</h3>
            <ul>
                <li><strong>Protection from Domestic Violence:</strong> Under the Protection of Women from Domestic Violence Act, 2005.</li>
                <li><strong>Equal Pay:</strong> The Equal Remuneration Act, 1976 mandates equal pay for equal work.</li>
                <li><strong>Maternity Benefits:</strong> The Maternity Benefit Act provides for 26 weeks of paid maternity leave.</li>
                <li><strong>Protection at Workplace:</strong> Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013.</li>
                <li><strong>Property Rights:</strong> Equal inheritance rights under the Hindu Succession (Amendment) Act, 2005.</li>
            </ul>
            <p><strong>Helplines:</strong> Women's Helpline: 1091, Domestic Abuse Helpline: 181</p>
        </div>
        """,
        
        "Property Laws": """
        <div class="feature-card">
            <h3>📝 Property Laws - Key Points</h3>
            <ul>
                <li><strong>Registration:</strong> All property transactions should be registered under the Registration Act, 1908.</li>
                <li><strong>Stamp Duty:</strong> Mandatory payment varying by state (typically 5-10% of property value).</li>
                <li><strong>Inheritance:</strong> Governed by personal laws (Hindu, Muslim, Christian, Parsi) or Indian Succession Act.</li>
                <li><strong>Tenant Rights:</strong> Protected under various Rent Control Acts in different states.</li>
                <li><strong>Land Ceiling:</strong> Restrictions on maximum land holdings in urban areas.</li>
            </ul>
        </div>
        """,
        
        "Consumer Protection": """
        <div class="feature-card">
            <h3>📝 Consumer Protection Rights</h3>
            <p>Under the Consumer Protection Act, 2019, you have the right to:</p>
            <ul>
                <li><strong>Right to Safety:</strong> Protection against hazardous goods and services.</li>
                <li><strong>Right to Information:</strong> Complete details about performance, quality, quantity, and price.</li>
                <li><strong>Right to Choose:</strong> Access to variety of goods at competitive prices.</li>
                <li><strong>Right to be Heard:</strong> Have your interests receive due consideration.</li>
                <li><strong>Right to Redressal:</strong> Fair settlement of genuine grievances.</li>
                <li><strong>Right to Consumer Education:</strong> Acquire knowledge and skills to be an informed consumer.</li>
            </ul>
            <p><strong>How to File a Complaint:</strong> Visit the nearest Consumer Forum or file online at <a href="https://consumerhelpline.gov.in">National Consumer Helpline</a></p>
        </div>
        """,
        
        "Labor Laws": """
        <div class="feature-card">
            <h3>📝 Key Labor Laws in India</h3>
            <ul>
                <li><strong>Minimum Wages Act, 1948:</strong> Ensures minimum wage payment to workers.</li>
                <li><strong>Factories Act, 1948:</strong> Regulates working conditions in factories.</li>
                <li><strong>Payment of Gratuity Act, 1972:</strong> Provides for gratuity payment to employees.</li>
                <li><strong>Employees' Provident Fund Act:</strong> Ensures retirement benefits.</li>
                <li><strong>Payment of Bonus Act, 1965:</strong> Provides for annual bonus payment.</li>
                <li><strong>Industrial Disputes Act, 1947:</strong> Mechanism for settlement of industrial disputes.</li>
            </ul>
            <p><strong>For Labor Disputes:</strong> Contact your local Labor Commissioner Office</p>
        </div>
        """,
        
        "Right to Information": """
        <div class="feature-card">
            <h3>📝 Right to Information (RTI) Act, 2005</h3>
            <p><strong>What is RTI?</strong> A law that allows citizens to request information from any public authority.</p>
            <ul>
                <li><strong>How to File an RTI:</strong> Submit application with Rs. 10 fee to the Public Information Officer (PIO).</li>
                <li><strong>Time Limit:</strong> Information must be provided within 30 days (48 hours if life/liberty is involved).</li>
                <li><strong>Appeal Process:</strong> First appeal to designated officer, second appeal to Information Commission.</li>
                <li><strong>Exemptions:</strong> Information affecting sovereignty, security, strategic interests, trade secrets, privacy, etc.</li>
            </ul>
            <p><strong>Online RTI Filing:</strong> <a href="https://rtionline.gov.in">RTI Online Portal</a></p>
        </div>
        """
    }
    
    st.markdown(topic_content[selected_topic], unsafe_allow_html=True)
    
    # Add video resources section
    st.markdown("""
    <div class="feature-card">
        <h3>📺 Educational Videos</h3>
        <p>Watch informative videos to better understand legal concepts</p>
    </div>
    """, unsafe_allow_html=True)
    
    video_col1, video_col2 = st.columns(2)
    with video_col1:
        st.markdown("#### Know Your Rights")
        st.image("https://via.placeholder.com/400x225", caption="Legal awareness video")
    
    with video_col2:
        st.markdown("#### How to File a Police Complaint")
        st.image("https://via.placeholder.com/400x225", caption="Process overview video")

with tabs[3]:  # FAQ Tab
    st.markdown("""
    <div class="feature-card">
        <h2>❓ Frequently Asked Questions</h2>
        <p>Find answers to common legal questions and concerns.</p>
    </div>
    """, unsafe_allow_html=True)

    # FAQ Accordion
    with st.expander("1. How can I get free legal aid in India?"):
        st.markdown("""
        <div class="feature-card">
            <p>Free legal aid is available through the <strong>National Legal Services Authority (NALSA)</strong> and its state-level counterparts. You can:</p>
            <ul>
                <li>Visit your nearest <strong>District Legal Services Authority (DLSA)</strong>.</li>
                <li>Call the NALSA helpline at <strong>15100</strong>.</li>
                <li>Apply online through the <a href="https://nalsa.gov.in">NALSA website</a>.</li>
            </ul>
            <p><strong>Eligibility:</strong> Women, children, SC/ST communities, victims of trafficking, and individuals with an annual income below ₹1,00,000 are eligible.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("2. What should I do if I'm a victim of domestic violence?"):
        st.markdown("""
        <div class="feature-card">
            <p>If you're facing domestic violence, take the following steps:</p>
            <ul>
                <li>Call the <strong>Women's Helpline</strong> at <strong>1091</strong> or <strong>181</strong>.</li>
                <li>File a complaint under the <strong>Protection of Women from Domestic Violence Act, 2005</strong>.</li>
                <li>Contact a <strong>Protection Officer</strong> in your district.</li>
                <li>Seek help from NGOs like <strong>Majlis</strong> or <strong>SAKHI</strong>.</li>
            </ul>
            <p><strong>Note:</strong> You can also approach the nearest police station or family court for immediate assistance.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("3. How do I file a consumer complaint?"):
        st.markdown("""
        <div class="feature-card">
            <p>To file a consumer complaint:</p>
            <ul>
                <li>Approach the <strong>Consumer Forum</strong> in your district.</li>
                <li>File a complaint online at the <a href="https://consumerhelpline.gov.in">National Consumer Helpline</a>.</li>
                <li>Provide evidence such as bills, receipts, and correspondence.</li>
            </ul>
            <p><strong>Time Limit:</strong> Complaints must be filed within 2 years of the issue.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("4. What are my rights as a tenant?"):
        st.markdown("""
        <div class="feature-card">
            <p>As a tenant, you have the following rights:</p>
            <ul>
                <li><strong>Right to a Rent Agreement:</strong> Ensure you have a written agreement.</li>
                <li><strong>Protection from Eviction:</strong> Landlords cannot evict you without proper notice.</li>
                <li><strong>Right to Essential Services:</strong> Landlords must provide water, electricity, and maintenance.</li>
                <li><strong>Security Deposit:</strong> You are entitled to the return of your deposit upon vacating.</li>
            </ul>
            <p><strong>Note:</strong> Tenant rights vary by state. Check your state's Rent Control Act for specifics.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("5. How do I file an RTI application?"):
        st.markdown("""
        <div class="feature-card">
            <p>To file an RTI application:</p>
            <ul>
                <li>Write a clear application stating the information you need.</li>
                <li>Pay a fee of ₹10 (waived for below-poverty-line applicants).</li>
                <li>Submit the application to the <strong>Public Information Officer (PIO)</strong> of the relevant department.</li>
                <li>You can file online at the <a href="https://rtionline.gov.in">RTI Online Portal</a>.</li>
            </ul>
            <p><strong>Response Time:</strong> Information must be provided within 30 days.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("6. What should I do if I'm arrested?"):
        st.markdown("""
        <div class="feature-card">
            <p>If you're arrested, remember the following:</p>
            <ul>
                <li><strong>Right to Know:</strong> You have the right to know the reason for your arrest.</li>
                <li><strong>Right to Legal Aid:</strong> You can request free legal aid from the nearest Legal Services Authority.</li>
                <li><strong>Right to Bail:</strong> For bailable offenses, you can apply for bail immediately.</li>
                <li><strong>Right to Inform:</strong> The police must inform a family member or friend about your arrest.</li>
            </ul>
            <p><strong>Note:</strong> Do not sign any documents without consulting a lawyer.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("7. How can I check the status of my court case?"):
        st.markdown("""
        <div class="feature-card">
            <p>To check the status of your court case:</p>
            <ul>
                <li>Visit the <a href="https://ecourts.gov.in">eCourts website</a>.</li>
                <li>Enter your <strong>CNR (Case Number Record)</strong>.</li>
                <li>You can also visit the court's website or contact the court clerk.</li>
            </ul>
            <p><strong>Note:</strong> You can also use the <strong>Legal Aid Locator</strong> tab to find nearby courts.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("8. What are my rights as a woman in the workplace?"):
        st.markdown("""
        <div class="feature-card">
            <p>As a woman in the workplace, you have the following rights:</p>
            <ul>
                <li><strong>Equal Pay:</strong> You are entitled to equal pay for equal work under the Equal Remuneration Act, 1976.</li>
                <li><strong>Maternity Benefits:</strong> You are entitled to 26 weeks of paid maternity leave under the Maternity Benefit Act.</li>
                <li><strong>Protection from Harassment:</strong> The Sexual Harassment of Women at Workplace Act, 2013 protects you from harassment.</li>
                <li><strong>Safe Working Conditions:</strong> Employers must provide a safe and harassment-free workplace.</li>
            </ul>
            <p><strong>Note:</strong> If you face any issues, report them to your HR department or the Internal Complaints Committee (ICC).</p>
        </div>
        """, unsafe_allow_html=True)


with tabs[4]:  # Document Verification Tab
    st.markdown("""
    <div class="feature-card">
        <h2>📄 Document Verification</h2>
        <p>Upload your document to verify its authenticity. You can upload from your local file system or cloud storage.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for file upload options
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Upload from File Explorer")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"], key="file_uploader")

    with col2:
        st.markdown("### Upload from Cloud Storage")
        cloud_option = st.selectbox("Select Cloud Storage", ["Google Drive", "Dropbox", "OneDrive"])
        if cloud_option == "Google Drive":
            st.write("Google Drive integration coming soon!")
        elif cloud_option == "Dropbox":
            st.write("Dropbox integration coming soon!")
        elif cloud_option == "OneDrive":
            st.write("OneDrive integration coming soon!")

    # Document verification logic
    if uploaded_file is not None:
        st.markdown("### Uploaded Document Details")
        file_details = {
            "Filename": uploaded_file.name,
            "File Type": uploaded_file.type,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)

        # Simulate verification logic (for demonstration purposes)
        def verify_document(file):
            # Example: Check if the file is a PDF and has a reasonable size
            if file.type == "application/pdf" and file.size < 5 * 1024 * 1024:  # Less than 5MB
                return True, "Document is verified and authentic."
            else:
                return False, "Document verification failed. Please check the file format and size."

        # Perform verification
        is_verified, verification_message = verify_document(uploaded_file)

        # Display verification result
        if is_verified:
            st.success("✅ Verified")
            st.markdown(f"<div class='assistant-message'>{verification_message}</div>", unsafe_allow_html=True)
        else:
            st.error("❌ Not Verified")
            st.markdown(f"<div class='assistant-message'>{verification_message}</div>", unsafe_allow_html=True)

        # Display the uploaded file (if PDF)
        if uploaded_file.type == "application/pdf":
            st.markdown("### Document Preview")
            base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" style="border: 1px solid #ddd;"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="background-color: #2c3e50; padding: 20px; border-radius: 10px; text-align: center; margin-top: 30px; color: white;">
    <p>RAYS- Empowering Citizens with Legal Knowledge</p>
    <p style="font-size: 0.8em;">This is an AI-powered legal assistant. For specific legal advice, consult a qualified legal professional.</p>
    <p style="font-size: 0.8em;">Emergency Contacts: National Legal Aid Helpline: 15100 | Women's Helpline: 1091 | Child Helpline: 1098</p>
</div>
""", unsafe_allow_html=True)