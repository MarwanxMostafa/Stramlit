import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
import io
import pickle
import time
import matplotlib.pyplot as plt
import os
from PIL import Image

# --- TensorFlow / CV Imports ---
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Plant Monitoring System",
    page_icon="🌿",
    layout="wide",
)

# --- Directories ---
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'Crop_Dataset_updated.csv')
lgb_model_path = os.path.join(current_dir, 'model.pkl')
cv_model_path = os.path.join(current_dir, 'Color_Images.h5')

# --- Custom CSS ---
def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, .stApp {
            font-family: 'Inter', sans-serif;
        }

        /* --- Main App Styling with Gradient Background --- */
        .stApp {
            background: linear-gradient(145deg, #04443c, #045341, #045c4c, #09674a, #0c7651);
            background-attachment: fixed;
            color: #E0E0E0;
        }

        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background-color: rgba(26, 34, 28, 0.8);
            backdrop-filter: blur(5px);
            border-right: 1px solid #2A4B38;
        }

        h1, h2, h3 { color: #FFFFFF; font-weight: 600; }
        
        /* --- Widget & Input Styling --- */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1A221C;
            color: #E0E0E0;
            border: 1px solid #2A4B38;
            border-radius: 8px;
            transition: border-color 0.2s;
        }
        .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
            border-color: #38FF7A;
        }

        /* --- Button Styling --- */
        .stButton>button {
            background-color: #38FF7A;
            color: #121814;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        .stButton>button:hover { background-color: #2ee06a; }
        .stButton>button:focus { box-shadow: 0 0 0 2px #1A221C, 0 0 0 4px #2ee06a; }

        /* --- Containers and Cards --- */
        [data-testid="stMetric"], .stContainer, [data-testid="stVerticalBlock"] {
            background-color: rgba(26, 34, 28, 0.7);
            backdrop-filter: blur(5px);
            border: 1px solid #2A4B38;
            border-radius: 12px;
            padding: 1rem;
        }
        
        /* --- Expander Styling --- */
        [data-testid="stExpander"] {
            background-color: transparent;
            border: none;
        }
        [data-testid="stExpander"] summary {
            background-color: rgba(42, 75, 56, 0.5);
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# --- Gemini API Configuration ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    GEMINI_API_KEY = "AIzaSyB46WFzo4-BvfkzTZ6EDCvH2txlFW0HXh0" # Placeholder

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    GEMINI_ENABLED = False
else:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_ENABLED = True


# ==========================================
# --- CROP RECOMMENDER: Load Data ---
# ==========================================
try:
    with open(lgb_model_path, "rb") as f:
        rf_model = pickle.load(f)
    crop_data = pd.read_csv(data_path)

    ideal_conditions = crop_data.groupby('label')[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()

    crop_descriptions = {
        'rice': "A staple crop thriving in wet conditions with high rainfall and warmth.", 'maize': "A versatile cereal grain that prefers warm temperatures and moderate rainfall.", 'chickpea': "A drought-resistant legume that grows best in temperate climates with low humidity.", 'kidneybeans': "A common bean that prefers warm weather and well-drained soil.", 'pigeonpeas': "A tropical legume that can withstand dry conditions and poor soil.", 'mungbean': "A small, green legume that matures quickly and thrives in warm climates.", 'mothbeans': "A drought-resistant legume, perfect for arid and semi-arid regions.", 'blackgram': "A nutritious urad bean that thrives in warm, humid conditions.", 'lentil': "An ancient, nutritious legume that grows in well-drained soil.", 'pomegranate': "A fruit-bearing shrub for semi-arid climates with mild winters and hot summers.", 'banana': "Thrives in hot, humid climates with high rainfall and nutrient-rich soil.", 'mango': "A tropical tree that requires hot weather and plenty of rainfall to produce fruit.", 'grapes': "Grows best in temperate climates with distinct seasons and deep, well-drained soil.", 'watermelon': "A vine-like plant that requires a long, hot growing season.", 'muskmelon': "A type of cantaloupe that thrives in hot, dry climates with plenty of sun.", 'apple': "Prefers cool, temperate climates and requires a period of cold weather to produce fruit.", 'orange': "A citrus tree that grows best in subtropical climates with warm temperatures.", 'papaya': "A tropical fruit that requires a warm, frost-free climate to grow.", 'coconut': "Thrives in tropical coastal areas with high humidity, rainfall, and sandy soil.", 'cotton': "A fiber crop that requires a long, frost-free period and plenty of sun.", 'jute': "A vegetable fiber crop that requires a hot, humid climate and a lot of rainfall.", 'coffee': "Prefers a tropical climate with high humidity and well-drained soil."
    }
    CROP_MODEL_LOADED = True
except FileNotFoundError:
    # We don't warn immediately here to keep the UI clean, handled in the specific section
    CROP_MODEL_LOADED = False


# ==========================================
# --- COMPUTER VISION: Helper Functions ---
# ==========================================
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", 
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_leaf_spot)", 
    "Grape___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", 
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"
] 
NUM_CLASSES = len(CLASS_NAMES)

def build_model_architecture(num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights=None, 
        input_shape=IMG_SIZE+(3,)
    )
    base.trainable = False
    inputs = Input(shape=IMG_SIZE+(3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model

@st.cache_resource
def load_cv_model(path, num_classes):
    try:
        model = build_model_architecture(num_classes)
        model.load_weights(path)
        model.compile(
            optimizer=Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model
    except Exception as e:
        st.error(f"Error loading CV model: {e}")
        return None

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ==========================================
# --- DATA ANALYSIS: Helper Functions ---
# ==========================================
def get_data_summary(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    summary = {
        "dataset_shape": str(df.shape), "dataset_head": df.head().to_markdown(),
        "dataset_info": info_str, "dataset_description": df.describe().to_markdown(),
    }
    return "\n".join(f"### {key}\n{value}" for key, value in summary.items())

def handle_ai_query(query, df_summary):
    if not GEMINI_ENABLED:
        return "The Gemini AI feature is disabled. Please configure your API key."
    prompt = f"""You are an expert data analyst assistant. Your user has uploaded a dataset and has a question. Provide a clear, concise, and helpful response. If you suggest Python code, make sure it's correct, easy to understand, and ready to be used in a script.
    Here is a summary of the dataset the user is working with:
    {df_summary}
    ---
    User's Question: "{query}"
    Your Response:"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}")
        return "Sorry, I couldn't process your request."

# --- App State Initialization ---
if "messages" not in st.session_state: st.session_state.messages = []
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df = None
if "analysis_mode" not in st.session_state: st.session_state.analysis_mode = "Crop Recommendation"
if "user_role" not in st.session_state: st.session_state.user_role = None

# ==========================================
# --- HOME PAGE / ROLE SELECTION ---
# ==========================================
if st.session_state.user_role is None:
    st.title("Welcome to the AI Analysis & Crop Prediction Hub 🤖🌾")
    st.markdown("---")
    st.subheader("Please select your role to continue:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Business User", use_container_width=True, type="primary"):
            st.session_state.user_role = "Business"
            st.rerun()
    with col2:
        if st.button("🔬 Scientific Researcher", use_container_width=True, type="secondary"):
            st.session_state.user_role = "Scientific Researcher"
            st.rerun()

# ==========================================
# --- MAIN APP INTERFACE ---
# ==========================================
else:
    # --- Sidebar ---
    with st.sidebar:
        st.title(f"🤖 {st.session_state.user_role} Hub")
        st.markdown("Choose a tool to get started.")

        # Define available modes based on the user's role
        # Both roles get Crop Rec and Disease Detection
        modes = ["Crop Recommendation", "Disease Detection"]
        
        if CROP_MODEL_LOADED:
            modes.append("Chat with AI")
            # Only Researchers get data tools
            if st.session_state.user_role == "Scientific Researcher":
                modes.extend(["Data Overview", "Plotting"])

        st.session_state.analysis_mode = st.radio(
            "Choose an analysis tool:",
            options=modes,
            index=modes.index(st.session_state.analysis_mode) if st.session_state.analysis_mode in modes else 0
        )
        
        st.markdown("---")
        
        # Show CSV Uploader ONLY for Data Analysis Modes
        if st.session_state.analysis_mode in ["Chat with AI", "Data Overview", "Plotting"]:
            st.header("Upload Data for Analysis")
            st.markdown("_(Used for Chat, Overview, & Plotting)_")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_df = df
                    st.success("File uploaded!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.session_state.uploaded_df = None
            st.markdown("---")
            
        if st.button("⬅️ Change Role"):
            st.session_state.user_role = None
            st.rerun()


    # --- Main App Body ---
    
    # ----------------------------------------
    # MODE: CROP RECOMMENDATION
    # ----------------------------------------
    if st.session_state.analysis_mode == "Crop Recommendation":
        if not CROP_MODEL_LOADED:
            st.error("Crop Recommendation module is unavailable. Please ensure 'model.pkl' is in the correct directory.")
        else:
            st.title("🌾 Smart Crop Recommendation System 🌱")
            st.markdown("### Enter your soil and climate conditions below and get personalized crop recommendations!")

            st.subheader("Model Performance")
            st.info("The model's overall accuracy on unseen data is **95.42%**. This means you can trust its recommendations.")
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.header("🔧 Soil Parameters")
                N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=600.0, value=50.0)
                P = st.number_input("Phosphorus (P)", min_value=10.0, max_value=200.0, value=50.0)
                K = st.number_input("Potassium (K)", min_value=10.0, max_value=900.0, value=50.0)
                ph = st.number_input("Soil pH", min_value=5.0, max_value=8.5, value=6.5, step=0.1)
            with col2:
                st.header("☀️ Climate Conditions")
                temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=38.0, value=25.0)
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=112.5, value=60.0)
                rainfall = st.number_input("Rainfall (mm)", min_value=200.0, max_value=3000.0, value=800.0)

            st.markdown("---")
            top_k = st.selectbox("How many crop recommendations do you want?", [3, 5, 10], index=1)

            if st.button("🚀 Recommend Crops"):
                with st.spinner("Analyzing parameters..."):
                    user_input = pd.DataFrame([{'N': N, 'P': P, 'K': K, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}])
                    if N > 200 or K > 500: st.warning("⚠️ Warning: Your N or K values are extremely high and may not be realistic for some crops.")
                    if rainfall > 2000: st.warning("⚠️ Warning: This rainfall value is very high and could indicate flooding.")
                    
                    probs = rf_model.predict_proba(user_input)[0]
                    topk_idx = probs.argsort()[-top_k:][::-1]
                    topk_crops = [(rf_model.classes_[i], probs[i]) for i in topk_idx]

                st.success(f"### 🎉 Here are your top {top_k} recommended crops!")
                st.write("---")
                result_col1, result_col2 = st.columns([1, 2])

                with result_col1:
                    for i, (crop, prob) in enumerate(topk_crops):
                        st.metric(label=f"🥇 {i+1}. {crop}" if i == 0 else f"✨ {i+1}. {crop}", value=f"{prob*100:.2f}%")
                        with st.expander(f"More info about {crop}"):
                            st.markdown(f"**About:** {crop_descriptions.get(crop, 'No description available.')}")
                            st.markdown("---")
                            st.markdown(f"#### 🔎 How Your Input Compares to Ideal **{crop}** Conditions:")
                            ideal_values = ideal_conditions.loc[crop].round(2)
                            input_values = user_input.iloc[0].round(2)
                            comparison_df = pd.DataFrame({"Feature": ideal_values.index, "Your Input": input_values.values, f"Ideal for {crop}": ideal_values.values})
                            st.table(comparison_df.set_index("Feature"))

                            st.markdown("---")
                            st.subheader("💡 Parameter Impact")
                            diff_series = ((input_values - ideal_values).abs() / ideal_values) * 100
                            diff_df = diff_series.reset_index().rename(columns={'index': 'Parameter', 0: 'Difference (%)'})
                            
                            fig, ax = plt.subplots()
                            diff_df = diff_df.sort_values(by='Difference (%)', ascending=True)
                            bars = ax.barh(diff_df['Parameter'], diff_df['Difference (%)'], color='orange')
                            ax.set_xlabel("Percentage Difference from Ideal")
                            st.pyplot(fig)
                            st.info(f"💡 **Recommendation:** Your **{diff_df.iloc[-1]['Parameter']}** value is the most different from ideal conditions for {crop}.")

                with result_col2:
                    st.subheader("📊 Prediction Probabilities")
                    prob_df = pd.DataFrame({"Crop": [c[0] for c in topk_crops], "Probability": [p[1] * 100 for p in topk_crops]})
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.barh(prob_df["Crop"], prob_df["Probability"], color="darkgreen")
                    ax.set_xlabel("Probability (%)"); ax.set_title(f"Top {top_k} Crop Recommendations"); ax.invert_yaxis()
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%", va="center")
                    ax.set_xlim(right=115)
                    st.pyplot(fig)

            st.markdown("---")
            st.header("🔮 What-if Analysis")
            st.markdown("See how your current inputs compare to the ideal conditions for any crop.")
            selected_crop = st.selectbox("Select a crop:", ideal_conditions.index)
            if selected_crop:
                st.markdown(f"#### **Your Input vs. Ideal Conditions for {selected_crop}**")
                user_input_for_analysis = pd.DataFrame([{'N': N, 'P': P, 'K': K, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}])
                ideal_values = ideal_conditions.loc[selected_crop].round(2)
                input_values = user_input_for_analysis.iloc[0].round(2)
                comparison_df = pd.DataFrame({"Feature": ideal_values.index, "Your Input": input_values.values, f"Ideal for {selected_crop}": ideal_values.values})
                st.table(comparison_df.set_index("Feature"))

    # ----------------------------------------
    # MODE: DISEASE DETECTION (Integrated CV)
    # ----------------------------------------
    elif st.session_state.analysis_mode == "Disease Detection":
        st.title("🌿 Plant Disease Detection")
        st.markdown("Upload a plant leaf image to detect potential diseases using our **EfficientNetB0 AI Model**.")
        
        if not os.path.exists(cv_model_path):
            st.error(f"⚠️ Model file not found: `Color_Images.h5`. Please place it in: {current_dir}")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.info(f"**Supported Classes:** {NUM_CLASSES} types of plants/diseases (Tomato, Potato, Corn, Apple, Grape)")
                img_file = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])
                
                if img_file is not None:
                    st.image(img_file, caption="Uploaded Image", use_container_width=True)
                    predict_btn = st.button("🔍 Analyze Image", type="primary")

                    if predict_btn:
                        with st.spinner("Loading AI Model and Analyzing..."):
                            # Load Model (Cached)
                            cv_model = load_cv_model(cv_model_path, NUM_CLASSES)
                            
                            if cv_model:
                                try:
                                    image = Image.open(img_file).convert('RGB')
                                    processed_img = preprocess_image(image)
                                    
                                    predictions = cv_model.predict(processed_img)
                                    probs = predictions[0]
                                    top_5_indices = np.argsort(probs)[-5:][::-1]

                                    # Display Results in Col2
                                    with col2:
                                        st.subheader("📊 Analysis Results")
                                        for index in top_5_indices:
                                            class_name = CLASS_NAMES[index] if index < len(CLASS_NAMES) else f"Unknown {index}"
                                            probability = probs[index]
                                            
                                            st.markdown(f"**{class_name.replace('_', ' ').replace('___', ' - ')}**")
                                            st.progress(float(probability))
                                            st.caption(f"Confidence: {probability*100:.2f}%")
                                            st.divider()
                                except Exception as e:
                                    st.error(f"Prediction Error: {e}")

            if img_file is None:
                with col2:
                    st.info("Waiting for image upload...")

    # ----------------------------------------
    # MODE: DATA ANALYSIS MODES
    # ----------------------------------------
    else:
        st.title(f"Mode: {st.session_state.analysis_mode}")
        if st.session_state.uploaded_df is None:
            st.info("👋 Welcome! Please upload a CSV file using the sidebar to use this analysis tool.")
        else:
            df = st.session_state.uploaded_df

            if st.session_state.analysis_mode == "Chat with AI":
                st.markdown("Ask the AI anything about your data. For example: `What are the key insights?`")
                if not GEMINI_ENABLED: st.error("AI Chat is disabled. Please configure your Gemini API Key.")

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]): st.markdown(message["content"])

                if prompt := st.chat_input("What would you like to ask?"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = handle_ai_query(prompt, get_data_summary(df))
                            st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

            elif st.session_state.analysis_mode == "Data Overview":
                st.subheader("Dataset Preview"); st.dataframe(df.head())
                st.subheader("Descriptive Statistics"); st.dataframe(df.describe())
                st.subheader("Data Information")
                buffer = io.StringIO(); df.info(buf=buffer); st.text(buffer.getvalue())

            elif st.session_state.analysis_mode == "Plotting":
                st.subheader("Data Visualization")
                plot_type = st.selectbox("Select Plot Type", ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart"])
                if plot_type in ["Histogram", "Box Plot"]:
                    col = st.selectbox("Select a numerical column", df.select_dtypes(include=np.number).columns)
                    if col:
                        fig = px.histogram(df, x=col) if plot_type == "Histogram" else px.box(df, y=col)
                        fig.update_layout(title=f"{plot_type} of {col}")
                        st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Scatter Plot":
                    col1, col2 = st.selectbox("X-axis", df.columns), st.selectbox("Y-axis", df.columns, index=1 if len(df.columns)>1 else 0)
                    if col1 and col2:
                        fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
                        st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Bar Chart":
                    cat_col = st.selectbox("Categorical column", df.select_dtypes(include="object").columns)
                    num_col = st.selectbox("Numerical column", df.select_dtypes(include=np.number).columns)
                    if cat_col and num_col:
                        grouped_df = df.groupby(cat_col)[num_col].mean().reset_index()
                        fig = px.bar(grouped_df, x=cat_col, y=num_col, title=f"Avg {num_col} by {cat_col}")
                        st.plotly_chart(fig, use_container_width=True)