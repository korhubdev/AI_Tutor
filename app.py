import os
import json
import base64
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# 0) Keys (OpenAI) - secrets 우선
# =========================
# 로컬에서는 .env도 허용하되, 배포는 secrets 권장
DOTENV = Path("/home/metacomm/rubric/.env")
if DOTENV.exists():
    load_dotenv(DOTENV, override=True)

OPENAI_KEY = None
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    OPENAI_KEY = st.secrets["openai"]["api_key"]
else:
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.stop()

client = OpenAI(api_key=OPENAI_KEY)

FT_MODEL = "ft:gpt-4.1-mini-2025-04-14:korhub::D3cx1Qc8:ckpt-step-787"


# =========================
# 1) GitHub secrets
# =========================
def get_github_config():
    """
    secrets.toml 의 [github] 섹션에서 읽음
    """
    gh = st.secrets.get("github", {})
    token = gh.get("token")
    repo = gh.get("repo")        # "owner/repo"
    branch = gh.get("branch", "main")
    base_path = gh.get("base_path", "feedback")
    return token, repo, branch, base_path


def github_put_file(token: str, repo: str, branch: str, path_in_repo: str, content_bytes: bytes, commit_message: str):
    """
    GitHub REST API: Create or Update a file contents
    - token: personal access token (repo 권한)
    - repo: "owner/repo"
    - path_in_repo: "feedback/2026-02-02_xxx.json"
    """
    api_url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "streamlit-ai-tutor",
    }

    # 1) 기존 파일 존재 여부 확인 (있으면 sha 필요)
    r_get = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=20)
    sha = None
    if r_get.status_code == 200:
        sha = r_get.json().get("sha")
    elif r_get.status_code not in (404,):
        raise RuntimeError(f"GitHub GET 실패: {r_get.status_code} {r_get.text}")

    # 2) 업로드 payload
    b64 = base64.b64encode(content_bytes).decode("utf-8")
    payload = {
        "message": commit_message,
        "content": b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r_put = requests.put(api_url, headers=headers, json=payload, timeout=20)
    if r_put.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT 실패: {r_put.status_code} {r_put.text}")

    return r_put.json()


# =========================
# 2) Local save
# =========================
DATA_DIR = Path(__file__).parent / "data" / "feedback"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_feedback_local(payload: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_user = str(payload.get("reviewer", "unknown")).replace("/", "_")
    filename = f"{ts}_{safe_user}.json"
    out_path = DATA_DIR / filename
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# =========================
# 3) Streamlit config & session init
# =========================
st.set_page_config(page_title="자기소개서 자동 첨삭 프로그램", layout="wide")

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""

if "expert_score" not in st.session_state:
    st.session_state.expert_score = 3
if "expert_comment" not in st.session_state:
    st.session_state.expert_comment = ""
if "role" not in st.session_state:
    st.session_state.role = "expert"
if "user_id" not in st.session_state:
    st.session_state.user_id = "reviewer"


# =========================
# 4) Constants
# =========================
QUESTIONS = {
    "Q1. 업무 중 의사소통이 어려웠던 경험과 이를 극복한 방법을 구체적으로 말씀해 주십시오.": "의사소통",
    "Q2. 업무 또는 활동 중 직면한 문제를 해결한 경험을 구체적으로 말씀해 주십시오.": "문제해결",
    "Q3. 한정된 자원(시간, 예산, 인력 등)을 효율적으로 관리한 경험을 구체적으로 말씀해 주십시오.": "자원관리",
    "Q4. 팀원이나 동료와의 갈등을 해결한 경험을 구체적으로 말씀해 주십시오.": "대인관계",
    "Q5. 본인의 역량 개발을 위해 노력한 경험을 구체적으로 말씀해 주십시오.": "자기개발",
    "Q6. 우리 기관에 지원한 이유와 입사 후 기여할 수 있는 점을 말씀해 주십시오.": "조직이해",
    "Q7. 원칙과 현실 사이에서 갈등했던 경험과 그때의 선택을 말씀해 주십시오.": "직업윤리",
}

SYSTEM = """너는 면접/서술형 답변을 평가하는 루브릭 기반 평가자다.
반드시 아래 출력 형식만 사용해야 한다. 형식을 벗어나거나 다른 문장은 출력하지 마라.

[중요 규칙]
- '모범 문장'은 RESPONSE에서 발췌(인용)하지 말고, 부족한 요소를 보완한 '개선된 예시 문장'을 새로 작성하라.
- RESPONSE의 문장을 그대로 복사하거나 일부만 바꾸는 방식은 금지한다(직접 인용 금지).
- 모범 문장은 RESPONSE의 보완 포인트(상황 구체화/행동 단계/결과·교훈 등)를 반영하라.

출력 형식:
- 점수:
- 긍정 평가 요소:
- 부족한 요소:
- 개선 제안:
- 모범 문장:
- 점수 상승 근거:
"""


# =========================
# 5) UI
# =========================
st.title("자기소개서 자동 첨삭 프로그램")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1) 질문 선택")
    q = st.selectbox("진행할 질문을 선택하세요.", list(QUESTIONS.keys()))

    if st.button("이 질문으로 진행", type="primary"):
        st.session_state.selected_question = q
        st.session_state.analysis_result = ""
        st.rerun()

    if st.session_state.selected_question:
        st.info(st.session_state.selected_question)
        competency = QUESTIONS.get(st.session_state.selected_question)
        st.caption(f"평가 기준(역량): **{competency}**")

        st.subheader("2) 답변 입력")
        st.session_state.answer = st.text_area(
            "답변",
            value=st.session_state.answer,
            height=260,
            placeholder="여기에 답변을 입력/붙여넣기 하세요.",
        )

    st.subheader("3) AI 분석")
    if not st.session_state.selected_question:
        st.warning("질문을 먼저 선택해 주세요.")
    elif not st.session_state.answer.strip():
        st.warning("답변을 입력해 주세요.")
    else:
        competency = QUESTIONS.get(st.session_state.selected_question)

        if st.button("분석 실행", type="primary"):
            USER = f"""[INSTRUCTION]
다음 답변을 {competency} 기준으로 평가하라.

[RESPONSE]
{st.session_state.answer}
"""
            with st.spinner("분석 중..."):
                resp = client.responses.create(
                    model=FT_MODEL,
                    input=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER},
                    ],
                    temperature=0.2,
                    max_output_tokens=350,
                )
                st.session_state.analysis_result = resp.output_text or ""

        if st.session_state.analysis_result:
            st.text_area("분석 결과", st.session_state.analysis_result, height=420)
        else:
            st.info("분석 실행 버튼을 누르면 결과가 표시됩니다.")

with right:
    st.markdown("#### 전문가 코멘트")  # 글씨 크게 안 쓰게 subheader 대신

    if not st.session_state.selected_question:
        st.info("왼쪽에서 질문을 선택하면 전문가 코멘트를 작성할 수 있어요.")
        st.stop()

    with st.expander("왼쪽 결과 보기(참조)", expanded=True):
        st.markdown("**면접 질문**")
        st.write(st.session_state.selected_question)
        st.markdown("**지원자 답변**")
        st.write(st.session_state.answer if st.session_state.answer else "— (아직 입력 없음) —")
        st.markdown("**AI 분석 결과**")
        st.write(st.session_state.analysis_result if st.session_state.analysis_result else "— (아직 분석 없음) —")

    with st.form("expert_form", clear_on_submit=False):
        score = st.slider(
            "전문가가 생각하는 점수(5점 만점)",
            min_value=1, max_value=5,
            value=int(st.session_state.expert_score),
            step=1,
        )
        comment = st.text_area(
            "추가 코멘트 및 조언",
            value=st.session_state.expert_comment,
            height=220,
            placeholder="강점/개선점/구체적 조언을 작성하세요.",
        )

        # ✅ GitHub 업로드 옵션 (토큰/레포 설정이 되어 있을 때만 의미 있음)
        upload_to_github = st.checkbox("GitHub에도 저장", value=True)

        submitted = st.form_submit_button("피드백 저장", type="primary")

    if submitted:
        st.session_state.expert_score = score
        st.session_state.expert_comment = comment

        payload = {
            "role": st.session_state.role,
            "reviewer": st.session_state.user_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "question": st.session_state.selected_question,
            "competency": QUESTIONS.get(st.session_state.selected_question),
            "answer": st.session_state.answer,
            "ai_result": st.session_state.analysis_result,
            "expert_score": score,
            "expert_comment": comment,
        }

        # 1) 로컬 저장
        local_path = save_feedback_local(payload)

        # 2) GitHub 업로드
        gh_msg = None
        if upload_to_github:
            token, repo, branch, base_path = get_github_config()
            if not (token and repo):
                gh_msg = "GitHub secrets가 설정되지 않아 업로드를 건너뜀"
            else:
                # repo 내 경로: base_path/YYYY-MM-DD/파일.json 형태 추천
                day = datetime.now().strftime("%Y-%m-%d")
                fname = local_path.name
                path_in_repo = f"{base_path}/{day}/{fname}"

                try:
                    github_put_file(
                        token=token,
                        repo=repo,
                        branch=branch,
                        path_in_repo=path_in_repo,
                        content_bytes=local_path.read_bytes(),
                        commit_message=f"Add feedback {fname}",
                    )
                    gh_msg = f"GitHub 업로드 완료: {path_in_repo}"
                except Exception as e:
                    gh_msg = f"GitHub 업로드 실패: {e}"

        st.success(f"로컬 저장 완료 ✅ ({local_path})")
        if gh_msg:
            st.info(gh_msg)

        st.markdown("##### 저장된 피드백 미리보기")
        st.json(payload, expanded=False)


with st.expander("디버그 정보(선택)"):
    st.write("selected_question:", st.session_state.selected_question)
    st.write("answer_length:", len(st.session_state.answer))
    st.write("model:", FT_MODEL)
