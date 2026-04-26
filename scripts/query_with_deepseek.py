import os
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ====== 配置区 ======
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = "data/processed/faiss.index"
META_PATH = "data/processed/meta.json"
CONTACTS_PATH = "data/contacts.json"
TOP_K = 6
MIN_SCORE = 0.25
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# ====================

# 添加环境变量设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_contacts():
    if not os.path.exists(CONTACTS_PATH):
        return {}
    with open(CONTACTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def deepseek_chat(messages, temperature=0.15):
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("没检测到 DEEPSEEK_API_KEY：先 setx，然后重开终端")

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def normalize(v: np.ndarray) -> np.ndarray:
    # 方便用 inner product 当 cosine
    v = v.astype("float32")
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm


def retrieve(query: str, model, index, meta, top_k=TOP_K):
    q_emb = model.encode([query], show_progress_bar=False)
    q_emb = normalize(np.array(q_emb))

    scores, idxs = index.search(q_emb, top_k)  # scores: inner product (cosine)
    hits = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        hits.append((float(s), meta[i]))
    return hits


def build_prompt(query: str, hits):
    # 把检索到的chunk整理成“证据”
    ctx_lines = []
    for rank, (score, item) in enumerate(hits, 1):
        text = item.get("text", "")
        src = item.get("source", "unknown")
        ctx_lines.append(f"[{rank}] (score={score:.3f}, source={src})\n{text}")

    context = "\n\n".join(ctx_lines)
    prompt = f"""你是本科教务 AI 助手。

请严格根据给定资料回答学生问题，不允许编造任何政策、流程、条件或时间信息。

回答规则：
1. 先直接回答学生最关心的结论。
2. 如果有适用条件、限制、例外情况，再单独说明。
3. 如果问题涉及办理流程，请按步骤回答。
4. 如果资料不足以支持回答，请明确说“资料未覆盖”，不要猜测。
5. 回答要简洁、清楚、像教务老师解释规则一样。
6. 最后列出你使用的资料来源。

【用户问题】
{query}

【资料摘录】
{context}
"""
    return prompt


def fallback_answer(query: str, contacts: dict):
    q = query.lower()

    # 1. 考试 / 四六级 / 成绩相关 —— 优先级最高
    if any(k in q for k in ["四六级", "四级", "六级", "cet", "考试", "补考", "特考", "成绩", "成绩单", "uiuc transfer", "学位转换"]):
        c = contacts.get("exam", {})

    # 2. 学籍 / 交换 / srtp / 暑研 / 助教 等
    elif any(k in q for k in ["学籍", "学期报到", "报到", "学位登记", "学位证", "照片", "科研训练", "srtp", "暑研", "交换", "海外交换", "助教"]):
        c = contacts.get("student_status", {})

    # 3. 课程 / 选退课 / 毕业审核
    elif any(k in q for k in ["课程", "选课", "退课", "毕业审核", "培养方案"]):
        c = contacts.get("course", {})

    # 4. 一般性的“毕业 / 学分”问题，默认也先给课程方向
    elif any(k in q for k in ["毕业", "学分"]):
        c = contacts.get("course", {})

    # 5. 其他兜底
    else:
        c = contacts.get("default", {})

    name = c.get("name", "教务老师")
    email = c.get("email", "（请填写邮箱）")
    note = c.get("note", "")

    return f"""资料未覆盖。

建议联系：
{name}
邮箱：{email}

{note}
"""
def is_contact_list_question(query: str) -> bool:
    q = query.strip().lower()

    keywords = [
        "联系谁",
        "找谁",
        "联系人",
        "联系方式",
        "老师联系方式",
        "教务老师",
        "contact",
        "who should i contact",
        "who to contact"
    ]

    return any(k in q for k in keywords)


def contact_list_answer(contacts: dict) -> str:
    course = contacts.get("course", {})
    exam = contacts.get("exam", {})
    student_status = contacts.get("student_status", {})
    default = contacts.get("default", {})

    return f"""如果资料未覆盖，或者你需要进一步确认，可以根据问题类型联系对应老师：

1. {course.get("name", "刘钰")}
邮箱：{course.get("email", "yuliu@intl.zju.edu.cn")}
负责：{course.get("note", "课程设置，选课，退课，毕业审核")}

2. {exam.get("name", "孙佳怡")}
邮箱：{exam.get("email", "jiayisun@intl.zju.edu.cn")}
负责：{exam.get("note", "成绩登录，考试安排，补考，特考，四六级考试，学位转换(UIUC transfer)，开具成绩单")}

3. {student_status.get("name", "王若沁")}
邮箱：{student_status.get("email", "ruoqinwang@intl.zju.edu.cn")}
负责：{student_status.get("note", "学籍管理，学期报到，学位登记，学位证照片，本科生科研训练，SRTP，暑研，海外交换项目，助教管理")}

4. {default.get("name", "邵昉伟")}
邮箱：{default.get("email", "fangweishao@intl.zju.edu.cn")}
负责：{default.get("note", "其他无法判断或综合性问题")}
"""
def is_capability_question(query: str) -> bool:
    q = query.strip().lower()

    keywords = [
        "你会什么",
        "你能干什么",
        "你可以做什么",
        "你有什么功能",
        "你能回答什么",
        "你是谁",
        "你是干嘛的",
        "你能帮我什么",
        "你可以帮我什么",
        "what can you do",
        "who are you",
        "what are you",
        "your function",
        "your capabilities"
    ]

    return any(k in q for k in keywords)


def capability_answer():
    return """我是 ZJUI 本科教务 AI 助手，可以基于已收录的教务资料回答部分本科教务相关问题。

我目前可以帮助你查询和理解一些常见事项，例如：
- 课程设置、选课、退课、毕业审核
- 成绩、考试、补考、特考、四六级、成绩单
- 学籍管理、学期报到、学位登记、SRTP、暑研、海外交换等

需要注意的是，我的回答主要依据当前知识库中的文件和网页资料。如果资料没有覆盖相关问题，我不会随便编造，而是会尽量提供对应教务老师或办公室的联系方式，方便你进一步确认。"""
def rewrite_query(query: str) -> str:
    q = query.strip()
    rewrite_parts = [q]

    if "课程" in q or "选课" in q:
        rewrite_parts.append("curriculum course list elective required")

    if "毕业" in q or "学分" in q:
        rewrite_parts.append("degree requirements credits")

    return " ".join(rewrite_parts)


def rerank_by_source(query: str, hits):
    reranked = []

    for score, item in hits:
        source = item.get("source", "").lower()
        bonus = 0.0

        if "课程" in query and "curriculum" in source:
            bonus += 0.2

        reranked.append((score + bonus, item))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked


def ask_question(query: str):
    contacts = load_contacts()
    if is_capability_question(query):
        return {
            "answer": capability_answer(),
            "sources": [],
            "fallback": False,
        }
    if is_contact_list_question(query):
        return {
            "answer": contact_list_answer(contacts),
            "sources": [],
            "fallback": False,
        }        
    model = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rewritten_query = rewrite_query(query)
    hits = retrieve(rewritten_query, model, index, meta, TOP_K)
    hits = rerank_by_source(query, hits)

    if not hits or hits[0][0] < MIN_SCORE:
        return {
            "answer": fallback_answer(query, contacts),
            "sources": [],
            "fallback": True,
        }

    prompt = build_prompt(query, hits)

    messages = [
        {
            "role": "system",
            "content": """你是本科教务 AI 助手。
只能根据给定资料回答问题，不允许编造政策、规则或流程。

回答规则：
1. 先给直接结论
2. 再说明适用条件
3. 如果涉及流程，按步骤回答
4. 如果资料不足，请回答“资料未覆盖”

回答风格要像教务老师解释规则一样，清晰、简洁。"""
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    ans = deepseek_chat(messages)
    fallback_markers = [
        "资料未覆盖",
        "无法确定",
        "无法回答",
        "资料不足",
        "没有足够依据",
        "未提供相关信息",
        "无法从提供的资料中"
    ]

    if any(marker in ans for marker in fallback_markers):
        return {
            "answer": fallback_answer(query, contacts),
            "sources": [],
            "fallback": True,
        }
    source_list = []
    for score, item in hits[:5]:
        source_list.append({
            "source": item.get("source", "unknown"),
            "score": round(score, 3)
        })

    return {
        "answer": ans,
        "sources": source_list,
        "fallback": False,
    }


def main():
    print("✅ RAG + DeepSeek 已启动（输入 q 退出）")
    while True:
        query = input("\n请输入问题（q退出）：").strip()
        if not query or query.lower() == "q":
            break

        result = ask_question(query)
        print("\n" + result["answer"])


if __name__ == "__main__":
    main()