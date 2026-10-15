
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,  # 定义集合 整体表机构
    DataType,         #字段类型
    utility           # 辅助工具判断集合是否存在
)
from langchain_community.document_loaders import CSVLoader
from openai import OpenAI
import os
from dotenv import load_dotenv
from pymilvus.orm import collection
import csv
import streamlit as st


#系统提示词
SYSTEM_PROMPT = """你是造价清单助手和日常聊天助手。
 如果用户的问题是工程造价、清单、定额、算量等专业问题，请根据参考资料回答，并补充项目编码、项目名称、项目特征、计量单位、工程量计算规则、工作内容。
 如果用户只是闲聊、问候、问无关话题，可以直接回答，不用参考资料。
 资料里没有的，就直说资料里面没有记录，不要编造。
"""

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"))

#创建数据库
connections.connect(uri=r"D:\ai_zj_ku\zj_milvus.db") # 创建数据库 在lite 中 只能有一个库

#定义集合名称 ,如果有就先删掉旧的集合
collection_name = "qingdan_items"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)


# 先把定额文件加载文档在py中使用, 再存入集合中
loader_DE = CSVLoader(r"D:\\造价定额资料\\DE_1.csv",encoding="utf-8")  #先用一个变量名接收文件
documents=loader_DE.load()  #把文件转化为文档 document 文档
#查看输出内容
print(len(documents))
print(documents[0].page_content)
print(documents[0].metadata)

#设计 Collection Schema

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=3000),
    FieldSchema(name="item_code", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="item_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="calc_rule", dtype=DataType.VARCHAR, max_length=2000),
]
schema = CollectionSchema(fields, description="清单项知识库") #构建集合框架
collection = Collection(name=collection_name, schema=schema) # 创建向量表(集合) 结合名字和框架

#定义向量化函数 调阿里云
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=text
    )
    return response.data[0].embedding
data_rows = []
with open(r"D:\造价定额资料\DE_1.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_rows.append(row)

print(f"读取到 {len(data_rows)} 条数据")
#逐条向量化 准备插入  insert插入
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
    vector = get_embedding(full_text)
    insert_data.append({
        "vector": vector,
        "text": full_text,
        "item_code": row["项目编码"],
        "item_name": row["项目名称"],
        "calc_rule": row["工程量计算规则"]
    })
    print(f"已向量化: {row['项目名称']}")

# 4. 插入到 Milvus, collection
collection.insert(insert_data)
collection.flush()  # 刷新集合
print(f"成功插入 {collection.num_entities} 条数据")

# 5. 创建索引并加载
index_params = {
    "index_type": "AUTOINDEX",
    "metric_type": "COSINE",
    "params": {}
}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()
print("索引创建完成，可以开始检索了")

# 定义一个函数，传入查询问题，返回相似的文档
def ask(query):
    # 1. 检索相关清单项 定义函数 query:查询
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=query)
    query_vector = response.data[0].embedding  # 107~111 把客户问题向量化
     #把客户问题向量化后在库里搜索答案
    results= collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {}},
        limit=3,
        output_fields=["item_name", "calc_rule", "text"]
    )
    # 2. 拼接参考资料
    context = ""
    for hits in results:
        for hit in hits:
            context += hit.entity.get("text") + "\n\n"

    # 发给deepseek
    deepseek_client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )

    chat_response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"}
        ]
    )

    return chat_response.choices[0].message.content

print(ask("你好"))

# ========== 5. 界面 ==========
st.set_page_config(page_title="造价清单助手", page_icon="🏗️")
st.title("🏗️ 造价清单智能问答助手")
st.caption("基于《房屋建筑与装饰工程工程量计算标准》附录A · 第一版")

# 侧边栏：项目说明
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

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
# 输入框
if prompt := st.chat_input("问一个造价问题，比如：挖沟槽土方怎么算量"):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            answer = ask(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})