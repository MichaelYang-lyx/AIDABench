"""
XHX Pipeline Agent — wraps the Raccoon office-chat API as an AIDABench agent.

The Raccoon client logic is implemented inline here so this file has no
dependency on raccoon_newppl_chat_conversation.py (which can be deleted).
"""

import base64
import json
import os
import re
import threading
import time
import uuid
from urllib.parse import quote

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Raccoon API client (ported from raccoon_newppl_chat_conversation.py)
# ---------------------------------------------------------------------------

_SANDBOX_FILE_EXT_RE = re.compile(
    r"\.(?:xlsx?|xlsm|csv|docx?|pptx?|pdf|png|jpe?g|gif|webp|bmp|svg|md|json|txt|py|ts|html?|xml|gz|zip)$",
    re.IGNORECASE,
)


def _sandbox_path_looks_like_file(p: str) -> bool:
    if not p or not p.startswith("sandbox:/mnt/data/"):
        return False
    base = p.rsplit("/", 1)[-1]
    if not base or "." not in base:
        return False
    return bool(_SANDBOX_FILE_EXT_RE.search(base))


class _RaccoonClient:
    """Low-level Raccoon API client with token caching and file upload support."""

    _token_lock = threading.Lock()
    _cached_tokens = None  # (access_token, refresh_token, refresh_time)

    def __init__(
        self,
        project_uuid: str = "",
        enable_web_search: bool = False,
        deep_think: bool = False,
    ):
        self.host = os.environ["RACCOON_HOST"]
        en_phone = os.environ["RACCOON_PHONE"]
        en_password = os.environ["RACCOON_PASSWORD"]
        self.project_uuid = project_uuid
        self.enable_web_search = enable_web_search
        self.deep_think = deep_think

        cached = self.__class__._cached_tokens
        if cached:
            self.token, self.refresh_token, self.token_refresh_time = cached
        else:
            self.token, self.refresh_token = self._login(en_phone, en_password)
            self.token_refresh_time = time.time()
            self.__class__._cached_tokens = (self.token, self.refresh_token, self.token_refresh_time)

    def _parse_retry_seconds(self, message):
        if not message:
            return 2
        m = re.search(r"retry after:(\d+)s", message)
        return max(1, int(m.group(1))) if m else 2

    def _login(self, en_phone, en_password, max_retry=3):
        url = self.host + "/api/web/auth/v1/login_with_password"
        headers = {"Content-Type": "application/json"}
        data = {"nation_code": "86", "phone": en_phone, "password": en_password}
        with self.__class__._token_lock:
            cached = self.__class__._cached_tokens
            if cached:
                return cached[0], cached[1]
            last_err = ""
            for i in range(max_retry):
                resp = requests.post(url=url, headers=headers, json=data)
                if resp.status_code == 200:
                    d = json.loads(resp.text)["data"]
                    access_token, refresh_token = d["access_token"], d["refresh_token"]
                    self.__class__._cached_tokens = (access_token, refresh_token, time.time())
                    return access_token, refresh_token
                last_err = resp.text
                if "limiter_phone_verify_error" in resp.text and i < max_retry - 1:
                    time.sleep(self._parse_retry_seconds(resp.text) + 1)
            raise Exception(f"Failed to login: {last_err}")

    def _refresh_token_if_needed(self, interval=3600):
        if time.time() - self.token_refresh_time <= interval:
            return
        url = self.host + "/api/web/auth/v1/refresh"
        resp = requests.post(url, headers={"Content-Type": "application/json"},
                             json={"refresh_token": self.refresh_token})
        if resp.status_code == 200:
            d = json.loads(resp.text)["data"]
            self.token = d["access_token"]
            self.refresh_token = d["refresh_token"]
            self.token_refresh_time = time.time()
            self.__class__._cached_tokens = (self.token, self.refresh_token, self.token_refresh_time)
        else:
            raise Exception(f"Failed to refresh token: {resp.text}")

    def _common_headers(self):
        return {
            "Authorization": "Bearer " + self.token,
            "X-Org-Code": "",
            "Referer": "",
            "X-Raccoon-Language": "zh",
            "X-Client-Platform": "web",
        }

    def create_session(self):
        url = self.host + "/api/web/office/v3/sessions"
        headers = {**self._common_headers(), "Content-Type": "application/json"}
        body = {
            "character_setting_code": "",
            "chat_type": "agent",
            "doc_id": "",
            "include_doc_content_as_context": False,
            "name": "raccoon",
            "project_uuid": self.project_uuid,
        }
        try:
            resp = requests.post(url=url, headers=headers, json=body)
            session_id = json.loads(resp.text)["data"]["id"]
            return session_id, resp.text
        except Exception as e:
            return None, str(e)

    def delete_session(self, session_id):
        url = self.host + f"/api/web/office/v3/sessions/{session_id}"
        try:
            resp = requests.delete(url, headers=self._common_headers())
            if resp.status_code == 200:
                time.sleep(3)
                return True
        except Exception:
            pass
        return False

    def upload_file(self, batch_id: str, filepath: str):
        url = self.host + f"/api/web/office/v3/sessions/default_session/{batch_id}/files"
        try:
            with open(filepath, "rb") as f:
                files = {"file": (os.path.basename(filepath), f)}
                resp = requests.post(url, headers=self._common_headers(), files=files)
            if resp.status_code == 200:
                file_list = json.loads(resp.text)["data"].get("file_list", [])
                return [item["id"] for item in file_list], resp.text
            raise Exception(resp.text)
        except Exception as e:
            return None, str(e)

    def _extract_image_url(self, content: str) -> str:
        if not content or not isinstance(content, str):
            return content or ""
        if content.strip().startswith("[") and "]" in content:
            return content.split("]", 1)[1].strip()
        return content.strip()

    def _extract_sandbox_file_paths(self, content: str) -> list:
        if not content or not isinstance(content, str):
            return []
        paths = set()
        trail_chars = (")", "}", "]", "'", '"', "*")
        for pattern in [
            r"sandbox:/mnt/data/[^\s\[\]<>\"']+",
            r"(?<![a-zA-Z])/mnt/data/[^\s\[\]<>\"']+",
        ]:
            for m in re.finditer(pattern, content):
                p = m.group(0).strip()
                while p and p.endswith(trail_chars):
                    p = p[:-1].rstrip()
                if p.startswith("/mnt/data/"):
                    p = "sandbox:" + p
                if p and _sandbox_path_looks_like_file(p):
                    paths.add(p)
        return list(paths)

    def _verify_file_accessible(self, session_id: str, file_path: str) -> dict:
        url = f"{self.host}/api/web/office/v3/sessions/{session_id}/files?file_path={quote(file_path)}"
        try:
            resp = requests.get(url, headers=self._common_headers(), timeout=10)
            return {
                "path": file_path,
                "accessible": resp.status_code == 200 and len(resp.content) > 0,
                "status_code": resp.status_code,
                "content_length": len(resp.content),
            }
        except Exception as e:
            return {"path": file_path, "accessible": False, "status_code": None, "error": str(e)}

    def download_file_content(self, session_id: str, file_path: str):
        url = f"{self.host}/api/web/office/v3/sessions/{session_id}/files?file_path={quote(file_path)}"
        try:
            resp = requests.get(url, headers=self._common_headers(), timeout=60)
            if resp.status_code == 200 and resp.content:
                return resp.content, None
            return None, f"status={resp.status_code}"
        except Exception as e:
            return None, str(e)

    def get_image_from_url(self, url):
        url = self._extract_image_url(url) if isinstance(url, str) else url
        resp = requests.get(url)
        if resp.status_code == 200:
            return base64.b64encode(resp.content).decode("utf-8")
        return None

    def chat(self, session_id: str, content: str, upload_file_ids: list):
        url = self.host + f"/api/web/office/v3/sessions/{session_id}/chat-conversations"
        headers = {**self._common_headers(), "Content-Type": "application/json"}
        data = {
            "content": content,
            "session_id": session_id,
            "verbose": True,
            "enable_web_search": self.enable_web_search,
            "deep_think": self.deep_think,
            "upload_file_id": upload_file_ids,
        }
        results = []
        start_time = time.time()
        try:
            resp = requests.post(url, headers=headers, json=data, stream=True)
            if resp.status_code == 200:
                for chunk in resp.iter_lines():
                    try:
                        chunk_str = chunk.decode("utf-8").replace("data:", "")
                        parsed = json.loads(chunk_str)
                        stage = parsed["stage"]
                        output = parsed["data"]["delta"]
                        if results and stage == results[-1]["type"]:
                            results[-1]["content"] += output
                        elif output:
                            results.append({"type": stage, "content": output})
                    except Exception:
                        pass
            else:
                raise Exception(resp.text)
        except Exception as e:
            print(f"chat request error: {e}")
            return [], 0
        return results, round(time.time() - start_time, 8)

    def get_api_result(self, file_desc: dict, query_list: list):
        """
        Main flow: upload files (shared batch_id) -> create session -> chat.
        Returns (answer_list, extra_info, time_cost).
        """
        self._refresh_token_if_needed()
        filepath = file_desc.get("file_path")
        if isinstance(filepath, list):
            filepaths = [p for p in filepath if p]
        elif filepath:
            filepaths = [filepath]
        else:
            filepaths = []

        upload_file_ids = []
        if filepaths:
            batch_id = str(uuid.uuid4())
            for fp in filepaths:
                file_ids, resp_text = self.upload_file(batch_id, fp)
                if file_ids is None:
                    print(f"Upload failed: {fp} -> {resp_text}")
                    continue
                upload_file_ids.extend(file_ids)

        session_id, _ = self.create_session()
        if session_id is None:
            return [], {}, 0
        time.sleep(8)

        answer_list = []
        time_cost = 0
        for query in query_list:
            ans, time_cost = self.chat(session_id, query, upload_file_ids)
            answer_list.append(ans)
            time.sleep(3)

            new_ans = []
            for a in ans:
                try:
                    if a["type"] in ("code", "execution"):
                        new_ans.append(a)
                    elif a["type"] == "generate":
                        if a.get("content"):
                            new_ans.append({"type": "text", "content": a["content"]})
                    elif a["type"] == "image":
                        url = self._extract_image_url(a["content"])
                        new_ans.append({"type": "image", "content": url or a["content"]})
                except Exception:
                    new_ans.append(a)
            answer_list.append(new_ans)

        # Collect sandbox file paths from text blocks, and image block URLs
        all_paths = set()
        image_urls = []
        for item in answer_list:
            for block in item:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    all_paths.update(self._extract_sandbox_file_paths(str(block.get("content", ""))))
                elif block.get("type") == "image":
                    url = self._extract_image_url(str(block.get("content", "")))
                    if url:
                        image_urls.append(url)

        file_urls = []
        for p in sorted(all_paths):
            r = self._verify_file_accessible(session_id, p)
            file_urls.append(r)

        extra_info = {"session_id": session_id, "file_urls": file_urls, "image_urls": image_urls}
        return answer_list, extra_info, time_cost


# ---------------------------------------------------------------------------
# AIDABench agent interface
# ---------------------------------------------------------------------------

class XHXPipelineAgent:
    """
    AIDABench agent that routes tasks through the Raccoon office-chat pipeline.

    Accepts the same constructor signature as other agents in this project
    (api_key, base_url, model_name are accepted but unused — Raccoon uses its
    own hardcoded credentials).
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model_name: str = "raccoon-office-chat",
        data_root_path: str = "",
        max_rounds: int = 20,
        raccoon_project_uuid: str = "",
        enable_web_search: bool = False,
        deep_think: bool = False,
        **kwargs,
    ):
        self.data_root_path = data_root_path
        self._client = _RaccoonClient(
            project_uuid=raccoon_project_uuid,
            enable_web_search=enable_web_search,
            deep_think=deep_think,
        )

    def interact(
        self,
        query: str,
        system_prompt: str = "",
        run_code_func=None,
        path_info: dict = None,
    ) -> dict:
        """
        Send query (with optional files) to Raccoon and return a standard result dict.

        path_info keys used:
          real_input_dir  — directory containing the input files
          (file paths are already embedded in the query by run_QA.process_row)
        """
        path_info = path_info or {}
        real_input_dir = path_info.get("real_input_dir", self.data_root_path)

        # Collect file paths: look for /mnt/data/<filename> references in the query
        # run_QA.process_row appends "你所用到的文件在: /mnt/data/file1, /mnt/data/file2"
        file_paths = []
        mnt_prefix = "/mnt/data/"
        for part in re.split(r"[,，\s]+", query):
            part = part.strip()
            if part.startswith(mnt_prefix):
                fname = part[len(mnt_prefix):]
                candidate = os.path.join(real_input_dir, fname)
                if os.path.isfile(candidate):
                    file_paths.append(candidate)

        if len(file_paths) == 1:
            file_desc = {"file_path": file_paths[0]}
        elif file_paths:
            file_desc = {"file_path": file_paths}
        else:
            file_desc = {"file_path": None}

        try:
            result = self._client.get_api_result(file_desc, [query])
        except Exception as e:
            return {
                "model_response": f"Error during API call: {e}",
                "history": [],
                "extra_info": {},
                "total_tokens": 0,
                "rounds": 1,
            }

        if not result or len(result) < 2:
            return {"model_response": "", "history": [], "extra_info": {}, "total_tokens": 0, "rounds": 1}

        answer_list, extra_info, _ = result

        # history: full answer_list for traceability
        history = answer_list

        # model_response: last text block from the last non-empty reply round
        model_response = ""
        for round_blocks in reversed(answer_list):
            if not isinstance(round_blocks, list):
                continue
            for block in reversed(round_blocks):
                if isinstance(block, dict) and block.get("type") == "text":
                    content = block.get("content", "").strip()
                    if content:
                        model_response = content
                        break
            if model_response:
                break

        # 如果 runner 指定了输出目录（data_visualization/file_generation），下载 sandbox 文件到本地
        real_output_dir = path_info.get("real_output_dir")
        if real_output_dir:
            os.makedirs(real_output_dir, exist_ok=True)
            session_id = extra_info.get("session_id")
            # 下载 sandbox 路径文件
            for fu in extra_info.get("file_urls", []):
                if not fu.get("accessible"):
                    continue
                content, err = self._client.download_file_content(session_id, fu["path"])
                if content:
                    fname = fu["path"].split("/")[-1]
                    with open(os.path.join(real_output_dir, fname), "wb") as f:
                        f.write(content)
                else:
                    print(f"下载文件失败: {fu['path']} -> {err}")
            # 下载 image block 中的图片 URL
            for i, url in enumerate(extra_info.get("image_urls", [])):
                try:
                    resp = requests.get(url, timeout=60)
                    if resp.status_code == 200 and resp.content:
                        fname = f"image_{i}.png"
                        with open(os.path.join(real_output_dir, fname), "wb") as f:
                            f.write(resp.content)
                    else:
                        print(f"下载图片失败: {url} -> status={resp.status_code}")
                except Exception as e:
                    print(f"下载图片失败: {url} -> {e}")

        return {
            "model_response": model_response,
            "history": history,
            "extra_info": extra_info,
            "total_tokens": 0,
            "rounds": 1,
        }
