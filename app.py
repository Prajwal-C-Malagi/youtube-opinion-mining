import streamlit as st
import requests
import pandas as pd
import re
import matplotlib.pyplot as plt
from transformers import pipeline
from datetime import datetime

# --- 1. Your Default Private API Key ---
DEFAULT_API_KEY = st.secrets["YOUTUBE_API_KEY"]

# --- 2. Initial Setup: Load the AI Brain ---


@st.cache_resource
def load_ai_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# --- 3. Core Functions ---


@st.cache_data
def get_video_details(video_id, api_key):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {'part': 'snippet,statistics', 'id': video_id, 'key': api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        items = response.json().get('items', [])
        if items:
            return items[0]
    return None


def extract_credits(description):
    credits = {"Starring/Cast": "N/A", "Music Director": "N/A",
               "Singer(s)": "N/A", "Lyricist": "N/A", "Director": "N/A"}
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
    response = requests.get(search_url, params=params)
    results = []
    if response.status_code == 200:
        for item in response.json().get('items', []):
            results.append({'id': item['id']['videoId'], 'title': item['snippet']
                           ['title'], 'channel': item['snippet']['channelTitle']})
    return results


@st.cache_data
def get_comments(video_id, api_key):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {'part': 'snippet', 'videoId': video_id,
              'key': api_key, 'maxResults': 100}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return pd.DataFrame()
    comments = []
    for item in response.json().get('items', []):
        text = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(text)
    return pd.DataFrame(comments, columns=['Comment'])


def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    return text.strip()


def get_top_keywords(text_series, num_words=8):
    text = " ".join(comment for comment in text_series).lower()
    words = re.findall(r'\b\w{4,}\b', text)
    junk = {'video', 'watch', 'watching', 'channel', 'subscribe',
            'like', 'comment', 'youtube', 'really', 'songs', 'song'}
    filtered_words = [w for w in words if w not in junk]
    if not filtered_words:
        return pd.Series(dtype=int)
    return pd.Series(filtered_words).value_counts().head(num_words)

# --- 4. The Streamlit Web Interface ---


st.set_page_config(
    page_title="Opinion Mining On YouTube Comments", layout="wide")

st.title("📊 Opinion Mining On YouTube Comments Dashboard")
st.write("Full Insight: Credits, Engagement, and Sentiment.")

ai_analyzer = load_ai_model()
search_query = st.text_input("🔍 Search for a Song/Video:")

if search_query:
    search_results = search_videos_by_name(search_query, DEFAULT_API_KEY)
    if not search_results:
        st.error("No results found.")
    else:
        video_options = {
            f"{res['title']} (by {res['channel']})": res['id'] for res in search_results}
        selected_video_id = video_options[st.selectbox(
            "🎯 Select Video:", list(video_options.keys()))]

        if st.button("Generate Full Intelligence Report", type="primary"):
            with st.spinner("Processing Data..."):
                full_video_data = get_video_details(
                    selected_video_id, DEFAULT_API_KEY)
                df = get_comments(selected_video_id, DEFAULT_API_KEY)

            if not df.empty and full_video_data:
                snippet = full_video_data['snippet']
                stats = full_video_data['statistics']

                # --- Top Layout ---
                col_v, col_i = st.columns([2, 1])
                with col_v:
                    st.video(
                        f"https://www.youtube.com/watch?v={selected_video_id}")
                with col_i:
                    st.subheader("📝 Song Metadata")
                    credits = extract_credits(snippet['description'])
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

                # --- AI Sentiment Batch Processing ---
                with st.spinner("🧠 AI Reading Comments..."):
                    df['Cleaned_Comment'] = df['Comment'].apply(clean_text)
                    comment_list = [c[:500]
                                    for c in df['Cleaned_Comment'].tolist() if c.strip()]

                    if comment_list:
                        batch_results = ai_analyzer(
                            comment_list, batch_size=16, truncation=True)
                        df['Sentiment'] = [r['label'].capitalize()
                                           for r in batch_results]
                    else:
                        df['Sentiment'] = 'Neutral'

                # --- Sentiment Visuals ---
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

                st.download_button("Download Report", df.to_csv(
                    index=False).encode('utf-8'), "song_analysis.csv")
