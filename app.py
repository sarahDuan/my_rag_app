import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 页面配置
st.set_page_config(page_title="企业级 RAG 需求分析师", page_icon="📑", layout="wide")

# 侧边栏：配置与知识库管理
with st.sidebar:
    st.header("⚙️ 系统设置")

    # API Key 输入 (为了安全，建议在网页上输入，而不是硬编码)
    api_key = st.text_input("智谱 API Key", type="password", help="在智谱开放平台获取")

    if not api_key:
        st.warning("⚠️ 请输入 API Key 以启动服务")
        st.stop()

    st.divider()

    st.subheader("📚 知识库管理")
    st.info("上传公司背景文档构建向量库（首次使用需上传）")

    uploaded_files = st.file_uploader(
        "上传业务文档 (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    kb_dir = "./temp_kb"
    if uploaded_files and st.button("🚀 构建/更新知识库"):
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir)

        with st.spinner("正在解析文档并构建向量索引..."):
            # 1. 保存上传的文件
            saved_paths = []
            for file in uploaded_files:
                path = os.path.join(kb_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                saved_paths.append(path)

            # 2. 加载与切片
            loaders = []
            for path in saved_paths:
                if path.endswith(".pdf"):
                    loaders.append(PyPDFLoader(path))
                elif path.endswith(".docx"):
                    loaders.append(Docx2txtLoader(path))
                else:
                    loaders.append(TextLoader(path, encoding="utf-8"))

            docs = []
            for loader in loaders:
                docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            split_docs = splitter.split_documents(docs)

            # 3. 向量化
            embeddings = OpenAIEmbeddings(
                model="embedding-2",
                api_key=api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )

            # 创建或更新向量库
            vectordb = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=kb_dir
            )
            st.success(f"✅ 知识库构建完成！共收录 {len(split_docs)} 个片段。")

# 主界面
st.title("📑 企业级 RAG 智能需求分析师")
st.markdown("""
结合**公司业务背景知识**与**新需求文档**，自动生成深度分析报告，识别合规风险。
""")

# 初始化模型
llm = ChatOpenAI(
    model="glm-4",
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.3
)

# 检查知识库是否存在
if not os.path.exists(kb_dir) or not os.listdir(kb_dir):
    st.error("❌ 请先在左侧侧边栏上传业务文档并点击‘构建知识库’。")
    st.stop()

# 加载向量库
embeddings = OpenAIEmbeddings(model="embedding-2", api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")
vectordb = Chroma(persist_directory=kb_dir, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 上传待分析的需求文档
st.divider()
st.subheader("📄 步骤 2：上传新需求文档进行分析")
req_file = st.file_uploader("上传需求文档 (PDF, DOCX)", type=["pdf", "docx"])

if req_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(req_file.name)[1]) as tmp_file:
        tmp_file.write(req_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("🔍 开始深度分析", type="primary"):
        try:
            # 1. 加载需求文档
            if tmp_path.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = Docx2txtLoader(tmp_path)
            req_docs = loader.load()
            req_text = "\n".join([d.page_content for d in req_docs])


            # 2. 构建 RAG 链
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)


            prompt = ChatPromptTemplate.from_template("""
            你是一位资深的需求分析专家。请结合【公司背景知识】和【新需求文档】进行分析。

            【公司背景知识】(来自向量库检索):
            {context}

            【新需求文档内容】:
            {document_text}

            【任务】:
            对比新需求与公司背景知识，输出深度分析报告。若发现违规或冲突，请高亮指出！

            报告结构：
            1. **项目概述**
            2. **合规性检查** (重点：是否违反公司规范？)
            3. **功能需求分析**
            4. **风险提示**
            5. **待确认问题**
            """)

            rag_chain = (
                    {"context": retriever | format_docs, "document_text": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )

            # 3. 流式输出结果
            st.subheader("📊 分析报告")
            response_placeholder = st.empty()
            full_response = ""

            for chunk in rag_chain.stream(req_text):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # 提供下载
            st.download_button(
                label="📥 下载报告 (Markdown)",
                data=full_response,
                file_name="analysis_report.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"发生错误: {e}")
        finally:
            os.unlink(tmp_path)  # 清理临时文件
else:
    st.info("👆 请上传需求文档以开始分析。")