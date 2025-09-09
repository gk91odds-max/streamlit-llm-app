import os
import re
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- .env の読み込み ---
load_dotenv()

# --- 専門家プロファイル定義 ---
EXPERT_PROFILES = {
    "summary_writer": {
        "label": "文章要約ライター",
        "system": (
            "あなたは要約専門ライターです。入力文を目的別に3段階要約（50字/150字/300字）し、"
            "重要キーワードを抽出、読み手別（上司/顧客/同僚）の要約版も提示します。固有名詞は保持。"
        ),
    },
    "proofread_polite": {
        "label": "文章校正・敬語チェッカー",
        "system": (
            "あなたは日本語校正者です。誤字脱字・助詞・冗長表現を修正し、"
            "丁寧語に統一した改稿版を提示。修正理由の簡潔な注記と、"
            "さらに良くする改善提案（3点）も示します。"
        ),
    },
}

MAX_CHARS = 1000  # 提出用なので軽めに設定

def truncate_with_notice(s: str, limit: int) -> tuple[str, bool]:
    if len(s) <= limit:
        return s, False
    return s[:limit], True

def cleanse_text(s: str) -> str:
    s = "".join(ch for ch in s if (ch.isprintable() or ch in "\n\t"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def looks_japanese(s: str) -> bool:
    return bool(re.search(r"[ぁ-んァ-ヶ一-龥]", s))

# --- LLM 呼び出し関数 ---
def ask_llm(input_text: str, expert_key: str) -> str:
    profile = EXPERT_PROFILES.get(expert_key)
    if not profile:
        return "不明な専門家が選択されました。"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", profile["system"]),
            ("human", "{user_text}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    chain = prompt | llm
    resp = chain.invoke({"user_text": input_text})
    return getattr(resp, "content", str(resp))

# --- Streamlit UI ---
st.set_page_config(page_title="LLMライティング支援", page_icon="✍️")
st.title("✍️ LLMライティング支援（要約 / 校正）")

with st.expander("アプリの使い方", expanded=True):
    st.markdown(
        """
1. 下のラジオで専門家を選ぶ  
   - **文章要約ライター** → 文章を多段階要約  
   - **文章校正・敬語チェッカー** → 誤字修正・敬語化・改善提案  
2. テキストを入力  
3. 「実行」を押すと結果が表示
        """
    )

expert_key = st.radio(
    "専門家を選択してください",
    options=list(EXPERT_PROFILES.keys()),
    format_func=lambda k: EXPERT_PROFILES[k]["label"],
    horizontal=True,
)

user_text = st.text_area(
    "入力テキスト",
    height=220,
    placeholder="ここに文章を貼り付けてください",
)

col1, col2 = st.columns([1, 3])
with col1:
    run = st.button("実行", type="primary")

if "last_input_hash" not in st.session_state:
    st.session_state.last_input_hash = None

if run:
    text = (user_text or "").strip()
    if not text:
        st.warning("入力テキストを入力してください。")
        st.stop()
    if len(text) < 5:
        st.warning("もう少し長い文章を入力してください。")
        st.stop()
    if len(text) > MAX_CHARS:
        st.info(f"長すぎるので先頭 {MAX_CHARS} 文字だけで実行します。")
        text, _ = truncate_with_notice(text, MAX_CHARS)
    text = cleanse_text(text)
    if expert_key == "proofread_polite" and not looks_japanese(text):
        st.info("校正・敬語チェッカーは日本語向けです。")
        st.stop()

    cur_hash = hashlib.md5((expert_key + "|" + text).encode("utf-8")).hexdigest()
    if st.session_state.last_input_hash == cur_hash:
        st.info("同じ内容で直前に実行されています。")
        st.stop()
    st.session_state.last_input_hash = cur_hash

    with st.spinner("LLM が処理中です…"):
        try:
            result = ask_llm(text, expert_key)
            st.markdown("### 結果")
            with st.expander("結果を表示", expanded=True):
                st.write(result)
            st.download_button("結果を保存", data=result, file_name="result.txt")
        except Exception as e:
            st.error(f"エラーが発生しました：{e}")