import streamlit as st
import requests
import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
from groq import Groq
import concurrent.futures  # For multi-threading speed

# --- 1. Bulletproof API Key Resolution ---
try:
    DEFAULT_API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")
except Exception:
    DEFAULT_API_KEY = ""

try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    GROQ_API_KEY = ""

# If the YouTube key isn't found in secrets, let the user input it via the sidebar
if not DEFAULT_API_KEY:
    st.sidebar.warning("⚠️ YouTube API Key not found in secrets.")
    DEFAULT_API_KEY = st.sidebar.text_input(
        "Enter YouTube API Key:", type="password")

# If the Groq key isn't found in secrets, let the user input it via the sidebar
if not GROQ_API_KEY:
    st.sidebar.warning("⚠️ Groq API Key not found in secrets.")
    GROQ_API_KEY = st.sidebar.text_input(
        "Enter Groq API Key:", type="password")

if not DEFAULT_API_KEY or not GROQ_API_KEY:
    st.info("Please provide both API keys in the sidebar to run the dashboard.")
    st.stop()

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)


# --- 2. Groq Sentiment Analysis Engine (MULTI-THREADED) ---
def fetch_chunk_sentiment(chunk, client):
    """Fetches sentiment for a specific chunk of comments."""
    prompt = f"""Analyze the sentiment of the following {len(chunk)} YouTube comments.
        Classify each exactly as either 'Positive', 'Negative', or 'Neutral'.
        You MUST return a valid JSON object. The JSON object must contain a single key called "sentiments".
        The value of "sentiments" must be an array of exactly {len(chunk)} strings.
        Comments:
        {json.dumps(chunk)}
        """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(chat_completion.choices[0].message.content)
        chunk_sentiments = result.get("sentiments", [])

        # --- SMART PARSING ---
        valid_labels = ['Positive', 'Negative', 'Neutral']
        processed = []

        # Step 1: Clean whatever the AI actually returned
        for s in chunk_sentiments:
            if isinstance(s, str) and s.capitalize() in valid_labels:
                processed.append(s.capitalize())
            else:
                processed.append('Neutral')

        # Step 2: If the AI missed some comments, pad the end with Neutral
        while len(processed) < len(chunk):
            processed.append('Neutral')

        # Step 3: If the AI accidentally returned too many, trim the excess
        return processed[:len(chunk)]

    except Exception as e:
        # Only trigger this if the internet completely drops
        st.warning(f"Groq API network hiccup: {e}")
        return ['Neutral'] * len(chunk)


def analyze_sentiments_with_groq(comments, client, chunk_size=20):
    """
    Splits the full list of comments into chunks and processes them in parallel
    using ThreadPoolExecutor to drastically speed up Groq API calls.
    """
    if not comments:
        return []

    chunks = [comments[i:i + chunk_size]
              for i in range(0, len(comments), chunk_size)]
    all_sentiments = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Map guarantees the results are returned in the exact original order
        results = list(executor.map(
            lambda c: fetch_chunk_sentiment(c, client), chunks))

    for res in results:
        all_sentiments.extend(res)

    return all_sentiments


# --- 3. Core Functions ---
@st.cache_data
def get_video_details(video_id, api_key):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {'part': 'snippet,statistics', 'id': video_id, 'key': api_key}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            items = response.json().get('items', [])
            if items:
                return items[0]
    except Exception as e:
        st.error(f"Error fetching video details: {e}")
    return None


def extract_credits(description):
    credits = {"Starring/Cast": "N/A", "Music Director": "N/A",
               "Singer(s)": "N/A", "Lyricist": "N/A", "Director": "N/A"}
    if not description:
        return credits

    patterns = {
        "Starring/Cast": [r"(?:Starring|Cast|Hero|Heroine)\s*:\s*(.*)"],
        "Music Director": [r"(?:Music|Composer|Music Director)\s*:\s*(.*)"],
        "Singer(s)": [r"(?:Singer|Singers|Vocals)\s*:\s*(.*)"],
        "Lyricist": [r"(?:Lyrics|Lyricist)\s*:\s*(.*)"],
        "Director": [r"(?:Director|Directed by)\s*:\s*(.*)"]
    }
    for key, p_list in patterns.items():
        for p in p_list:
            match = re.search(p, description, re.IGNORECASE)
            if match:
                credits[key] = match.group(1).split('\n')[0].strip()
                break
    return credits


@st.cache_data
def search_videos_by_name(query, api_key):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {'part': 'snippet', 'q': query,
              'type': 'video', 'maxResults': 20, 'key': api_key}
    try:
        response = requests.get(search_url, params=params)
        results = []
        if response.status_code == 200:
            for item in response.json().get('items', []):
                results.append({'id': item['id']['videoId'], 'title': item['snippet']
                               ['title'], 'channel': item['snippet']['channelTitle']})
            return results
    except Exception as e:
        st.error(f"Error during video search: {e}")
    return []


@st.cache_data
def get_comments(video_id, api_key, max_pages=2):
    """Fetches comments with basic pagination support to gather more analytical depth"""
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {'part': 'snippet', 'videoId': video_id,
              'key': api_key, 'maxResults': 100}
    comments = []
    page = 0

    try:
        while params and page < max_pages:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                break

            res_data = response.json()
            for item in res_data.get('items', []):
                text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(text)

            # Handle token tracking for next page
            next_page_token = res_data.get('nextPageToken')
            if next_page_token:
                params['pageToken'] = next_page_token
                page += 1
            else:
                break
    except Exception as e:
        st.error(f"Error extracting comments: {e}")

    return pd.DataFrame(comments, columns=['Comment']) if comments else pd.DataFrame()


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<.*?>', ' ', text)  # Clear out HTML elements safely
    text = re.sub(r'http\S+', '', text)  # Drop raw hyperlink paths
    return text.strip()


def get_top_keywords(text_series, num_words=8):
    text = " ".join(comment for comment in text_series).lower()
    words = re.findall(r'\b\w{4,}\b', text)
    junk = {'video', 'watch', 'watching', 'channel', 'subscribe',
            'like', 'comment', 'youtube', 'really', 'songs', 'song', 'br'}
    filtered_words = [w for w in words if w not in junk]
    if not filtered_words:
        return pd.Series(dtype=int)
    return pd.Series(filtered_words).value_counts().head(num_words)


# --- 4. The Streamlit Web Interface ---
st.set_page_config(
    page_title="Opinion Mining On YouTube Comments", layout="wide")

st.title("📊 Opinion Mining On YouTube Comments Dashboard")
st.write("Full Insight: Credits, Engagement, and Sentiment.")


# Handle Persistent App State Management
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
    st.session_state.df = None
    st.session_state.full_video_data = None
    st.session_state.current_video_id = ""

search_query = st.text_input("🔍 Search for a Song/Video:")

if search_query:
    search_results = search_videos_by_name(search_query, DEFAULT_API_KEY)
    if not search_results:
        st.error("No results found. Please check your query or API configurations.")
    else:
        video_options = {
            f"{res['title']} (by {res['channel']})": res['id'] for res in search_results}
        selected_video_title = st.selectbox(
            "🎯 Select Video:", list(video_options.keys()))
        selected_video_id = video_options[selected_video_title]

        # Reset layout states seamlessly if the video target changes
        if selected_video_id != st.session_state.current_video_id:
            st.session_state.report_generated = False
            st.session_state.current_video_id = selected_video_id

        if st.button("Generate Full Intelligence Report", type="primary") or st.session_state.report_generated:

            # Check if execution state needs initialization
            if not st.session_state.report_generated:
                with st.spinner("Processing Data and Analysing Sentiment via Groq..."):
                    st.session_state.full_video_data = get_video_details(
                        selected_video_id, DEFAULT_API_KEY)
                    processed_df = get_comments(
                        selected_video_id, DEFAULT_API_KEY)

                    if not processed_df.empty and st.session_state.full_video_data:
                        processed_df['Cleaned_Comment'] = processed_df['Comment'].apply(
                            clean_text)

                        # Prepare comments for the LLM (truncating extremely long ones to save tokens)
                        comment_list = [
                            c[:500] for c in processed_df['Cleaned_Comment'].tolist() if c.strip()]

                        if comment_list:
                            # NEW GROQ INFERENCE ENGINE
                            processed_df['Sentiment'] = analyze_sentiments_with_groq(
                                comment_list, groq_client)
                        else:
                            processed_df['Sentiment'] = 'Neutral'

                        st.session_state.df = processed_df
                        st.session_state.report_generated = True
                    else:
                        st.error(
                            "Unable to scrape comments or read video metrics for this selection.")

            # --- Layout Dashboard Rendering ---
            if st.session_state.report_generated and st.session_state.df is not None:
                df = st.session_state.df
                snippet = st.session_state.full_video_data['snippet']
                stats = st.session_state.full_video_data['statistics']

                # --- Top Layout ---
                col_v, col_i = st.columns([2, 1])
                with col_v:
                    st.video(
                        f"https://www.youtube.com/watch?v={selected_video_id}")
                with col_i:
                    st.subheader("📝 Song Metadata")
                    credits = extract_credits(snippet.get('description', ''))
                    publish_date = datetime.strptime(
                        snippet['publishedAt'][:10], '%Y-%m-%d')
                    days_ago = (datetime.now() - publish_date).days

                    st.metric("Total Views",
                              f"{int(stats.get('viewCount', 0)):,}")
                    st.metric("Total Likes",
                              f"{int(stats.get('likeCount', 0)):,}")
                    st.write(
                        f"📅 **Released:** {snippet['publishedAt'][:10]} ({days_ago} days ago)")
                    st.markdown("---")
                    st.write(f"🎭 **Cast:** {credits['Starring/Cast']}")
                    st.write(f"🎤 **Singer:** {credits['Singer(s)']}")
                    st.write(f"🎶 **Music:** {credits['Music Director']}")
                    st.write(f"✍️ **Lyrics:** {credits['Lyricist']}")

                st.markdown("---")

                # --- Sentiment Visuals Layout ---
                tab_all, tab_pos, tab_neg, tab_neu = st.tabs(
                    ["All Data", "Positive 🟢", "Negative 🔴", "Neutral ⚪"])
                with tab_all:
                    st.dataframe(df[['Comment', 'Sentiment']],
                                 use_container_width=True)
                with tab_pos:
                    st.dataframe(df[df['Sentiment'] == 'Positive']
                                 [['Comment']], use_container_width=True)
                with tab_neg:
                    st.dataframe(df[df['Sentiment'] == 'Negative']
                                 [['Comment']], use_container_width=True)
                with tab_neu:
                    st.dataframe(df[df['Sentiment'] == 'Neutral']
                                 [['Comment']], use_container_width=True)

                st.markdown("---")

                st.subheader("📈 Keyword & Sentiment Deep Dive")
                c_pie, c_pos, c_neg, c_neu = st.columns(4)
                counts = df['Sentiment'].value_counts()
                cmap = {'Positive': '#2ecc71',
                        'Negative': '#e74c3c', 'Neutral': '#3498db'}

                with c_pie:
                    fig1, ax1 = plt.subplots()
                    ax1.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=[
                            cmap.get(x, '#ccc') for x in counts.index])
                    ax1.set_title("Sentiment %")
                    st.pyplot(fig1)

                for col, sent, color, title in zip([c_pos, c_neg, c_neu], ['Positive', 'Negative', 'Neutral'], ['#2ecc71', '#e74c3c', '#3498db'], ['Top Positive', 'Top Negative', 'Top Neutral']):
                    with col:
                        k_data = get_top_keywords(
                            df[df['Sentiment'] == sent]['Cleaned_Comment'])
                        if not k_data.empty:
                            fig, ax = plt.subplots()
                            k_data.plot(kind='barh', color=color,
                                        ax=ax).invert_yaxis()
                            ax.set_title(title)
                            st.pyplot(fig)
                        else:
                            st.info(f"No strong trends for {sent}")

                st.download_button("Download Report", df.to_csv(
                    index=False).encode('utf-8'), "song_analysis.csv")
