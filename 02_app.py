import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

warnings.filterwarnings("ignore")

#  1. 加载配置
load_dotenv()

ali_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

# ========== 2. 自动建库 + 连接 Milvus ==========
DB_URI = r'D:\ai_zj_ku\\zj_milvus.db'  # 绝对路径，云端本地都能用
connections.connect(uri=DB_URI)

collection_name = "qingdan_items"

# 检查集合是否存在，且是否有数据
if utility.has_collection(collection_name):
    collection = Collection(collection_name)
    collection.load()
    need_build = collection.num_entities == 0
else:
    need_build = True

if need_build:
    st.info("首次启动，正在建库...")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=3000),
        FieldSchema(name="item_code", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="item_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="calc_rule", dtype=DataType.VARCHAR, max_length=2000),
    ]
    schema = CollectionSchema(fields, description="清单项知识库")
    collection = Collection(name=collection_name, schema=schema)

    import csv
    data_rows = []
    with open("DE_1.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_rows.append(row)

    insert_data = []
    for row in data_rows:
        full_text = (
            f"项目编码: {row['项目编码']}\n"
            f"项目名称: {row['项目名称']}\n"
            f"项目特征: {row['项目特征']}\n"
            f"计量单位: {row['计量单位']}\n"
            f"工程量计算规则: {row['工程量计算规则']}\n"
            f"工作内容: {row['工作内容']}"
        )
        response = ali_client.embeddings.create(
            model="text-embedding-v4",
            input=full_text
        )
        vector = response.data[0].embedding
        insert_data.append({
            "vector": vector,
            "text": full_text,
            "item_code": row["项目编码"],
            "item_name": row["项目名称"],
            "calc_rule": row["工程量计算规则"]
        })

    collection.insert(insert_data)
    collection.flush()
    collection.create_index(field_name="vector", index_params={
        "index_type": "AUTOINDEX",
        "metric_type": "COSINE",
        "params": {}
    })
    collection.load()
    st.success(f"建库完成，共 {collection.num_entities} 条数据")

# 3. 系统提示词
SYSTEM_PROMPT = """你是造价清单助手和日常聊天助手。
如果用户的问题是工程造价、清单、定额、算量等专业问题，请根据参考资料回答，并补充项目编码、项目名称、项目特征、计量单位、工程量计算规则、工作内容。
如果用户只是闲聊、问候、问无关话题，可以直接回答，不用参考资料。
资料里没有的，就直说资料里面没有记录，不要编造。
"""


# 4. 问答函数
def ask(query):
    # 4.1 问题向量化
    response = ali_client.embeddings.create(
        model="text-embedding-v4",
        input=query
    )
    query_vector = response.data[0].embedding

    # 4.2 检索 Milvus
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {}},
        limit=3,
        output_fields=["text"]
    )

    # 4.3 拼上下文
    context = ""
    for hits in results:
        for hit in hits:
            context += hit.entity.get("text") + "\n\n"

    # 4.4 生成答案
    chat_response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"}
        ]
    )

    return chat_response.choices[0].message.content


# 5. 界面 
st.set_page_config(page_title="造价清单助手", page_icon="🏗️")
st.title("🏗️ 造价清单智能问答助手")
st.caption("基于《房屋建筑与装饰工程工程量计算标准》附录A · 第一版")

with st.sidebar:
    st.markdown("### 关于")
    st.markdown("**数据源**：GB/T 50854 附录A 土石方工程")
    st.markdown("**当前收录**：12条清单项")
    st.markdown("**技术栈**：RAG + Milvus + DeepSeek")
    st.markdown("---")
    st.markdown("### 试试这些问题")
    st.markdown("- 挖沟槽土方怎么算量")
    st.markdown("- 平整场地的工作内容")
    st.markdown("- 010102002是什么项目")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("问一个造价问题，比如：挖沟槽土方怎么算量"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            answer = ask(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})