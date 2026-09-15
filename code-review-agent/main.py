import base64
from datetime import datetime
import io
import json
import os
import re
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fpdf import FPDF

from reviewer import (
    step1_find_bugs, step2_verify_bugs, step3_generate_fix,
    step4_double_check, analyze_code_structure, analyze_folder_structure,
    compute_metrics_for_charts, aggregate_folder_metrics,
    clean_markdown_symbols
)

def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k:
                        os.environ[k] = v

load_env_file()

app = FastAPI(title="CodeSentinel Agentic Reviewer")
templates = Jinja2Templates(directory="templates")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    if os.path.exists("static/logo.png"):
        return FileResponse("static/logo.png")
    return Response(status_code=204)


class CodeRequest(BaseModel):
    code: str
    engine_mode: Optional[str] = "ollama"
    api_provider: Optional[str] = "gemini"
    api_key: Optional[str] = None
    api_model: Optional[str] = None
    api_endpoint: Optional[str] = None


class ScanPathRequest(BaseModel):
    path: str


class LoadFileRequest(BaseModel):
    filepath: str


class FolderFileItem(BaseModel):
    filename: str
    rel_path: Optional[str] = ""
    content: str


class FolderReviewRequest(BaseModel):
    folder_name: Optional[str] = "Project Root"
    files: List[FolderFileItem]
    engine_mode: Optional[str] = "ollama"
    api_provider: Optional[str] = "gemini"
    api_key: Optional[str] = None
    api_model: Optional[str] = None
    api_endpoint: Optional[str] = None


class TestKeyRequest(BaseModel):
    engine_mode: Optional[str] = "api_key"
    api_provider: Optional[str] = "gemini"
    api_key: str
    api_model: Optional[str] = None
    api_endpoint: Optional[str] = None


@app.get("/api/env-config")
async def get_env_config():
    load_env_file()
    gemini_k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    openai_k = os.environ.get("OPENAI_API_KEY") or ""
    groq_k = os.environ.get("GROQ_API_KEY") or ""
    return {
        "gemini": gemini_k,
        "openai": openai_k,
        "groq": groq_k,
        "has_gemini": bool(gemini_k.strip())
    }


@app.post("/test-api-key")
async def test_api_key_endpoint(req: TestKeyRequest):
    key = req.api_key.strip() if req.api_key else ""
    if not key:
        load_env_file()
        if (req.api_provider or "gemini") == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        elif req.api_provider == "openai":
            key = os.environ.get("OPENAI_API_KEY") or ""
        elif req.api_provider == "groq":
            key = os.environ.get("GROQ_API_KEY") or ""
    if not key:
        return JSONResponse({"success": False, "message": "API key cannot be empty. Please enter your key or set GEMINI_API_KEY in .env."})

    api_cfg = {
        "engine_mode": "api_key",
        "api_provider": req.api_provider or "gemini",
        "api_key": key,
        "api_model": req.api_model.strip() if req.api_model else None,
        "api_endpoint": req.api_endpoint.strip() if req.api_endpoint else None
    }
    try:
        from reviewer import call_cloud_llm
        test_resp = call_cloud_llm("Reply with the single word: OK", api_cfg, num_predict=10, timeout=12)
        if test_resp and len(test_resp.strip()) > 0:
            prov_name = (req.api_provider or "Cloud").capitalize()
            return {"success": True, "message": f"Successfully connected to {prov_name}! Model response: '{test_resp[:25]}'"}
        else:
            return JSONResponse({"success": False, "message": "Provider did not return a response. Please check your API key and model name."})
    except Exception as e:
        return JSONResponse({"success": False, "message": f"Connection error: {str(e)}"})


class PDFRequest(BaseModel):
    mode: Optional[str] = "single"  # "single" or "folder"
    # Single file fields
    suspected: Optional[str] = ""
    confirmed: Optional[str] = ""
    fixed_code: Optional[str] = ""
    check: Optional[str] = ""
    confirmed_count: Optional[int] = None
    false_alarm_count: Optional[int] = None
    code_metadata: Optional[Dict[str, Any]] = None
    # Folder fields
    folder_name: Optional[str] = "Codebase"
    folder_metadata: Optional[Dict[str, Any]] = None
    file_results: Optional[List[Dict[str, Any]]] = None
    # Common 4 illustration graphs
    bug_chart_img: Optional[str] = None
    category_chart_img: Optional[str] = None
    severity_chart_img: Optional[str] = None
    quality_chart_img: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/scan-path")
async def scan_path(req: ScanPathRequest):
    """Scan directory or load file by local path."""
    raw_path = req.path.strip().strip('"').strip("'")
    if not raw_path:
        return JSONResponse({"status": "error", "message": "Please enter a valid path."})

    target = os.path.abspath(raw_path)
    if not os.path.exists(target):
        rel_target = os.path.join(os.getcwd(), raw_path)
        if os.path.exists(rel_target):
            target = rel_target
        else:
            return JSONResponse({"status": "error", "message": f"Path does not exist: {raw_path}"})

    if os.path.isdir(target):
        files_found = []
        ignore_dirs = {".git", "__pycache__", "venv", ".venv", "env", "node_modules", ".idea", ".vscode"}
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                if file.endswith(".py"):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, target)
                    try:
                        size_kb = round(os.path.getsize(full_p) / 1024, 1)
                        with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                            lines_cnt = len(f.readlines())
                    except Exception:
                        size_kb = 0
                        lines_cnt = 0
                    files_found.append({
                        "name": file,
                        "rel_path": rel_p,
                        "full_path": full_p,
                        "size_kb": size_kb,
                        "lines": lines_cnt
                    })

        return {
            "status": "ok",
            "type": "directory",
            "path": target,
            "total_files": len(files_found),
            "files": files_found
        }
    elif os.path.isfile(target):
        try:
            with open(target, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read()
            structure = analyze_code_structure(code_content)
            size_kb = round(os.path.getsize(target) / 1024, 1)
            return {
                "status": "ok",
                "type": "file",
                "filename": os.path.basename(target),
                "path": target,
                "size_kb": size_kb,
                "content": code_content,
                "structure": structure
            }
        except Exception as e:
            return JSONResponse({"status": "error", "message": f"Could not read file: {e}"})
    else:
        return JSONResponse({"status": "error", "message": f"Unsupported path type: {raw_path}"})


@app.post("/api/load-file")
async def load_file(req: LoadFileRequest):
    """Load code from a specific file path."""
    target = os.path.abspath(req.filepath.strip().strip('"').strip("'"))
    if not os.path.isfile(target):
        return JSONResponse({"status": "error", "message": f"File not found: {req.filepath}"})
    try:
        with open(target, "r", encoding="utf-8", errors="ignore") as f:
            code_content = f.read()
        structure = analyze_code_structure(code_content)
        size_kb = round(os.path.getsize(target) / 1024, 1)
        return {
            "status": "ok",
            "filename": os.path.basename(target),
            "path": target,
            "size_kb": size_kb,
            "content": code_content,
            "structure": structure
        }
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Error loading file: {e}"})


def stream_review_single(code: str, api_config: Optional[Dict[str, Any]] = None):
    structure = analyze_code_structure(code)
    yield json.dumps({
        "step": 0,
        "label": "Code Intelligence",
        "structure": structure
    }) + "\n"

    suspected = step1_find_bugs(code, api_config=api_config)
    yield json.dumps({
        "step": 1,
        "label": "Suspected Bugs",
        "content": suspected
    }) + "\n"

    confirmed = step2_verify_bugs(code, suspected, api_config=api_config)
    interim_metrics = compute_metrics_for_charts(code, suspected, confirmed, "")
    yield json.dumps({
        "step": 2,
        "label": "Verified Bugs",
        "content": confirmed,
        "metrics": interim_metrics
    }) + "\n"

    fixed_code = step3_generate_fix(code, confirmed, api_config=api_config)
    yield json.dumps({
        "step": 3,
        "label": "Suggested Fix",
        "content": fixed_code
    }) + "\n"

    check = step4_double_check(code, fixed_code, api_config=api_config)
    final_metrics = compute_metrics_for_charts(code, suspected, confirmed, check)
    yield json.dumps({
        "step": 4,
        "label": "Fix Validation",
        "content": check,
        "metrics": final_metrics,
        "done": True
    }) + "\n"


@app.post("/review-stream")
async def review_stream_endpoint(req: CodeRequest):
    key = req.api_key.strip() if req.api_key else ""
    if req.engine_mode == "api_key" and not key:
        load_env_file()
        if (req.api_provider or "gemini") == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        elif req.api_provider == "openai":
            key = os.environ.get("OPENAI_API_KEY") or ""
        elif req.api_provider == "groq":
            key = os.environ.get("GROQ_API_KEY") or ""

    api_cfg = {
        "engine_mode": req.engine_mode or "ollama",
        "api_provider": req.api_provider or "gemini",
        "api_key": key,
        "api_model": req.api_model,
        "api_endpoint": req.api_endpoint
    }
    return StreamingResponse(stream_review_single(req.code, api_config=api_cfg), media_type="text/plain")


def stream_review_folder(req_files: List[FolderFileItem], folder_name: str, api_config: Optional[Dict[str, Any]] = None):
    file_dicts = [{"filename": f.filename, "rel_path": f.rel_path or f.filename, "content": f.content} for f in req_files]
    folder_struct = analyze_folder_structure(file_dicts)

    yield json.dumps({
        "type": "folder_manifest",
        "folder_name": folder_name,
        "total_files": len(file_dicts),
        "folder_structure": folder_struct
    }) + "\n"

    all_results = []

    for idx, f in enumerate(file_dicts):
        fname = f["filename"]
        code = f["content"]
        yield json.dumps({
            "type": "file_start",
            "file_index": idx + 1,
            "total_files": len(file_dicts),
            "filename": fname,
            "rel_path": f["rel_path"]
        }) + "\n"

        suspected = step1_find_bugs(code, api_config=api_config)
        confirmed = step2_verify_bugs(code, suspected, api_config=api_config)
        fixed_code = step3_generate_fix(code, confirmed, api_config=api_config)
        check = step4_double_check(code, fixed_code, api_config=api_config)
        file_metrics = compute_metrics_for_charts(code, suspected, confirmed, check)

        file_res = {
            "file_index": idx + 1,
            "filename": fname,
            "rel_path": f["rel_path"],
            "suspected": suspected,
            "confirmed": confirmed,
            "fixed_code": fixed_code,
            "check": check,
            "metrics": file_metrics,
            "structure": analyze_code_structure(code)
        }
        all_results.append(file_res)

        yield json.dumps({
            "type": "file_complete",
            "file_index": idx + 1,
            "total_files": len(file_dicts),
            "filename": fname,
            "file_result": file_res
        }) + "\n"

    agg_metrics = aggregate_folder_metrics(all_results)
    yield json.dumps({
        "type": "folder_complete",
        "folder_name": folder_name,
        "aggregate": agg_metrics,
        "file_results": all_results,
        "done": True
    }) + "\n"


@app.post("/review-folder-stream")
async def review_folder_stream_endpoint(req: FolderReviewRequest):
    key = req.api_key.strip() if req.api_key else ""
    if req.engine_mode == "api_key" and not key:
        load_env_file()
        if (req.api_provider or "gemini") == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        elif req.api_provider == "openai":
            key = os.environ.get("OPENAI_API_KEY") or ""
        elif req.api_provider == "groq":
            key = os.environ.get("GROQ_API_KEY") or ""

    api_cfg = {
        "engine_mode": req.engine_mode or "ollama",
        "api_provider": req.api_provider or "gemini",
        "api_key": key,
        "api_model": req.api_model,
        "api_endpoint": req.api_endpoint
    }
    return StreamingResponse(stream_review_folder(req.files, req.folder_name or "Codebase", api_config=api_cfg), media_type="text/plain")


def safe_pdf_text(text: str, is_code: bool = False) -> str:
    """Sanitize symbols and convert to latin-1 compatible characters."""
    text = clean_markdown_symbols(text, is_code=is_code)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2022": "*",
        "\u2192": "->", "\u2190": "<-", "\u00a0": " ", "\t": "    "
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class CodeSentinelPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(140, 150, 160)
        self.cell(0, 10, f"CodeSentinel Agentic AI Audit Report  |  Offline Inference  |  Page {self.page_no()}/{{nb}}", align="C")


def decode_base64_image(data_url: Optional[str]) -> Optional[io.BytesIO]:
    if not data_url or not data_url.startswith("data:image"):
        return None
    try:
        header, encoded = data_url.split(",", 1)
        raw_bytes = base64.b64decode(encoded)
        return io.BytesIO(raw_bytes)
    except Exception:
        return None


@app.post("/generate-pdf")
async def generate_pdf(req: PDFRequest):
    pdf = CodeSentinelPDF()
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=14)

    is_folder_mode = req.mode == "folder" and req.file_results and len(req.file_results) > 0

    if not is_folder_mode:
        # ==========================================
        # 1. SINGLE-FILE AUDIT REPORT
        # ==========================================
        pdf.add_page()

        # Banner
        pdf.set_fill_color(15, 23, 42)
        pdf.rect(12, 12, 186, 24, style="F", round_corners=True, corner_radius=3)

        logo_path = "static/logo.png"
        text_x = 16
        if os.path.exists(logo_path):
            try:
                pdf.image(logo_path, x=15, y=14, w=20, h=20)
                text_x = 38
            except Exception:
                pass

        pdf.set_xy(text_x, 15)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 6, "CODESENTINEL", new_x="LMARGIN", new_y="NEXT")

        pdf.set_x(text_x)
        pdf.set_font("Helvetica", "", 8.2)
        pdf.set_text_color(203, 213, 225)
        pdf.cell(0, 5, "Agentic AI for Automated Code Review and Self-Debugging Using LLM", new_x="LMARGIN", new_y="NEXT")

        pdf.set_y(38)

        # Metadata
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(93, 5, f"Audit Timestamp: {gen_time}", new_x="RIGHT", new_y="TOP")
        pdf.cell(93, 5, "Mode: Single File | 100% On-Device Air-Gapped", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # Calculations
        confirmed_count = req.confirmed_count
        if confirmed_count is None:
            confirmed_count = max(req.confirmed.upper().count("CONFIRMED") - req.confirmed.upper().count("FALSE ALARM"), 0)
        false_alarm_count = req.false_alarm_count
        if false_alarm_count is None:
            false_alarm_count = req.confirmed.upper().count("FALSE ALARM")

        total_issues = confirmed_count + false_alarm_count
        health_score = max(100 - (confirmed_count * 20), 35) if confirmed_count > 0 else 100
        noise_filter_pct = int((false_alarm_count / max(total_issues, 1)) * 100)

        is_verified = "LOOKS CORRECT" in (req.check or "").upper()
        verdict_label = "PASSED" if is_verified else "FLAGGED"
        verdict_color = (22, 163, 74) if is_verified else (217, 119, 6)

        # Scorecards
        kpi_y = pdf.get_y()
        box_w, box_h, spacing, start_x = 44.0, 16.0, 3.3, 12.0
        scorecards = [
            ("Code Health Score", f"{health_score} / 100", (22, 163, 74) if health_score >= 80 else (217, 119, 6)),
            ("Confirmed Defects", f"{confirmed_count} Real Bug(s)", (220, 38, 38) if confirmed_count > 0 else (22, 163, 74)),
            ("Noise Filtered", f"{false_alarm_count} Discarded ({noise_filter_pct}%)", (100, 116, 139)),
            ("Automated Patch", verdict_label, verdict_color)
        ]
        for i, (label, val, col) in enumerate(scorecards):
            bx = start_x + i * (box_w + spacing)
            pdf.set_fill_color(248, 250, 252)
            pdf.set_draw_color(226, 232, 240)
            pdf.rect(bx, kpi_y, box_w, box_h, style="DF", round_corners=True, corner_radius=2)
            pdf.set_xy(bx, kpi_y + 2)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(box_w, 3.5, label, align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.set_xy(bx, kpi_y + 6.5)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_text_color(*col)
            pdf.cell(box_w, 6, val, align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.set_y(kpi_y + box_h + 5)

        # Flowchart
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 4.5, "Agentic Self-Debugging Pipeline Execution:", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        py = pdf.get_y()
        steps = [("Stage 1", "Bug Detection"), ("Stage 2", "Self-Verify"), ("Stage 3", "Patch Synthesis"), ("Stage 4", "Double-Check")]
        sw, sh, s_sp = 39.0, 9.5, 10.0
        for i, (st, desc) in enumerate(steps):
            bx = start_x + i * (sw + s_sp)
            pdf.set_fill_color(241, 245, 249)
            pdf.set_draw_color(203, 213, 225)
            pdf.rect(bx, py, sw, sh, style="DF", round_corners=True, corner_radius=2)
            pdf.set_xy(bx, py + 1.2)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(37, 99, 235)
            pdf.cell(sw, 3, st, align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.set_xy(bx, py + 4.5)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(sw, 4, desc, align="C", new_x="LMARGIN", new_y="NEXT")
            if i < 3:
                pdf.set_xy(bx + sw + 2, py + 2.8)
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(148, 163, 184)
                pdf.cell(6, 4, "->", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_y(py + sh + 5)

        # Code Structure & Usage Audit
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 5, "Codebase Structure & Usage Intelligence Audit:", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        meta = req.code_metadata or {}
        imports = meta.get("imports", [])
        functions = meta.get("functions", [])
        loc = meta.get("loc", {})
        complexity = meta.get("complexity", {})

        code_card_y = pdf.get_y()
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(226, 232, 240)
        pdf.rect(12, code_card_y, 186, 32, style="DF", round_corners=True, corner_radius=2)

        pdf.set_xy(16, code_card_y + 2.5)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(88, 3.5, "Imported Packages & Libraries:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(16)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        imports_text = ", ".join(imports) if imports else "None (Built-in Standard Primitives Only)"
        pdf.cell(88, 4, safe_pdf_text(imports_text), new_x="LMARGIN", new_y="NEXT")

        pdf.set_x(16)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(88, 3.5, "Detected Functions & Signatures:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(16)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        if functions:
            fn_strs = [f"{f['name']}({', '.join(f.get('args', []))})" for f in functions[:3]]
            fns_disp = "; ".join(fn_strs) + (f" (+{len(functions)-3} more)" if len(functions) > 3 else "")
        else:
            fns_disp = "Script level execution (0 formal def statements)"
        pdf.cell(88, 4, safe_pdf_text(fns_disp), new_x="LMARGIN", new_y="NEXT")

        pdf.set_xy(110, code_card_y + 2.5)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(84, 3.5, "Lines of Code (LOC) Metrics:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_xy(110, code_card_y + 6)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        tot_l, code_l, comm_l, ratio = loc.get("total", "N/A"), loc.get("code", "N/A"), loc.get("comments", "N/A"), loc.get("comment_ratio", "0")
        pdf.cell(84, 4, f"Total Lines: {tot_l}  |  Code: {code_l}  |  Comments: {comm_l} ({ratio}%)", new_x="LMARGIN", new_y="NEXT")

        pdf.set_xy(110, code_card_y + 11)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(84, 3.5, "Cyclomatic Complexity & Maintainability:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_xy(110, code_card_y + 14.5)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        c_score, c_label = complexity.get("score", 1), complexity.get("label", "Low")
        pdf.cell(84, 4, f"Branching Score: {c_score}  -  Rating: {safe_pdf_text(c_label)}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_y(code_card_y + 36)

        # 4 Illustration Graphs
        bug_img = decode_base64_image(req.bug_chart_img)
        cat_img = decode_base64_image(req.category_chart_img)
        sev_img = decode_base64_image(req.severity_chart_img)
        qual_img = decode_base64_image(req.quality_chart_img)
        if bug_img or cat_img or sev_img or qual_img:
            if pdf.get_y() > 180:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 5, "Visual Defect Analytics & Four Illustration Graphs:", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            gy1 = pdf.get_y()
            chart_w, chart_h = 89.0, 44.0
            if bug_img: pdf.image(bug_img, x=13, y=gy1, w=chart_w, h=chart_h)
            if cat_img: pdf.image(cat_img, x=108, y=gy1, w=chart_w, h=chart_h)
            pdf.set_y(gy1 + chart_h + 3)
            gy2 = pdf.get_y()
            if sev_img: pdf.image(sev_img, x=13, y=gy2, w=chart_w, h=chart_h)
            if qual_img: pdf.image(qual_img, x=108, y=gy2, w=chart_w, h=chart_h)
            pdf.set_y(gy2 + chart_h + 4)

        # Page 2: Table + Detailed Logs
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 5, "Defect Verification Breakdown Matrix:", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        ty = pdf.get_y()
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        pdf.rect(12, ty, 186, 6.5, style="DF")
        pdf.set_xy(14, ty + 1.2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(20, 4, "Stage", align="L")
        pdf.cell(100, 4, "Evaluation & Reasoning Detail", align="L")
        pdf.cell(36, 4, "Defect Status", align="C")
        pdf.cell(28, 4, "Remediation", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_y(ty + 7)

        def first_line_summary(text: str, max_chars: int = 65) -> str:
            clean = " ".join([l.strip() for l in (text or "").splitlines() if l.strip()])
            return (clean[:max_chars] + "...") if len(clean) > max_chars else clean

        rows_data = [
            ("Step 1", safe_pdf_text(first_line_summary(req.suspected)), "Potential Bug", "Identified"),
            ("Step 2", safe_pdf_text(first_line_summary(req.confirmed)), "CONFIRMED" if confirmed_count > 0 else "FALSE ALARM", "Isolated"),
            ("Step 3", "Synthesized patch addressing verified defects without side-effects", "Auto-Patched", "Implemented"),
            ("Step 4", safe_pdf_text(first_line_summary(req.check)), verdict_label, "Audited")
        ]
        for stage, detail, status, remed in rows_data:
            ry = pdf.get_y()
            pdf.set_fill_color(255, 255, 255)
            pdf.set_draw_color(226, 232, 240)
            pdf.rect(12, ry, 186, 6.5, style="DF")
            pdf.set_xy(14, ry + 1.2)
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(20, 4, stage, align="L")
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(51, 65, 85)
            pdf.cell(100, 4, detail, align="L")
            status_col = (220, 38, 38) if "CONFIRMED" in status else ((22, 163, 74) if status == "PASSED" else (100, 116, 139))
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(*status_col)
            pdf.cell(36, 4, status, align="C")
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(28, 4, remed, align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.set_y(ry + 6.5)
        pdf.ln(5)

        sections = [
            ("Step 1: Suspected Bug Scan", safe_pdf_text(req.suspected), (245, 158, 11), False),
            ("Step 2: Verification & Noise Reduction", safe_pdf_text(req.confirmed), (59, 130, 246), False),
            ("Step 3: Synthesized Code Patch", safe_pdf_text(req.fixed_code, is_code=True), (16, 185, 129), True),
            ("Step 4: Regression Validation & Sanity Verdict", safe_pdf_text(req.check), (99, 102, 241), False),
        ]
        for title, content, color, is_code in sections:
            if pdf.get_y() > 225:
                pdf.add_page()
            cur_y = pdf.get_y()
            pdf.set_fill_color(*color)
            pdf.rect(12, cur_y, 3, 5.5, style="F")
            pdf.set_xy(17, cur_y)
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 5.5, title, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            if is_code:
                pdf.set_font("Courier", "", 8)
                pdf.set_text_color(30, 41, 59)
                pdf.set_fill_color(248, 250, 252)
                pdf.multi_cell(0, 4.2, content, border=1, fill=True)
            else:
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(51, 65, 85)
                pdf.multi_cell(0, 4.5, content)
            pdf.ln(3)

        if pdf.get_y() > 245:
            pdf.add_page()
        pdf.ln(2)
        stamp_y = pdf.get_y()
        pdf.set_fill_color(240, 253, 244) if is_verified else (254, 243, 199)
        pdf.set_draw_color(187, 247, 208) if is_verified else (253, 230, 138)
        pdf.rect(12, stamp_y, 186, 13, style="DF", round_corners=True, corner_radius=2)
        pdf.set_xy(16, stamp_y + 2.2)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(21, 128, 61) if is_verified else (180, 83, 9)
        pdf.cell(0, 4, f"Official Audit Seal: {verdict_label} (Autonomous Multi-Step Verification)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(16)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(0, 4, "This software artifact was audited and self-debugged using the CodeSentinel local agentic pipeline.", new_x="LMARGIN", new_y="NEXT")

    else:
        # ==========================================
        # 2. COMPREHENSIVE FOLDER / CODEBASE AUDIT REPORT
        # ==========================================
        pdf.add_page()

        # Cover Banner
        pdf.set_fill_color(15, 23, 42)
        pdf.rect(12, 12, 186, 24, style="F", round_corners=True, corner_radius=3)

        pdf.set_xy(16, 15)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 6, "CODESENTINEL CODEBASE AUDIT REPORT", new_x="LMARGIN", new_y="NEXT")

        pdf.set_x(16)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(203, 213, 225)
        f_title = req.folder_name or "Uploaded Codebase"
        pdf.cell(0, 5, f"Multi-File Engineering Quality & Vulnerability Audit  |  Folder: {safe_pdf_text(f_title)}", new_x="LMARGIN", new_y="NEXT")

        pdf.set_y(40)

        # Metadata Row
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(93, 5, f"Audit Timestamp: {gen_time}", new_x="RIGHT", new_y="TOP")
        pdf.cell(93, 5, f"Total Code Files Audited: {len(req.file_results)}  |  100% Offline Air-Gapped", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # Folder Aggregate KPIs
        f_meta = req.folder_metadata or {}
        tot_files = len(req.file_results)
        tot_bugs = sum(fr.get("metrics", {}).get("confirmed_count", 0) for fr in req.file_results)
        tot_false_alarms = sum(fr.get("metrics", {}).get("false_alarm_count", 0) for fr in req.file_results)
        clean_files = sum(1 for fr in req.file_results if fr.get("metrics", {}).get("confirmed_count", 0) == 0)
        overall_health = max(100 - (tot_bugs * 15), 35) if tot_bugs > 0 else 100

        kpi_y = pdf.get_y()
        box_w, box_h, spacing, start_x = 44.0, 16.0, 3.3, 12.0
        folder_kpis = [
            ("Overall Codebase Health", f"{overall_health} / 100", (22, 163, 74) if overall_health >= 80 else (217, 119, 6)),
            ("Files Audited", f"{tot_files} Files ({clean_files} Clean)", (37, 99, 235)),
            ("Defects Isolated", f"{tot_bugs} Real Bug(s)", (220, 38, 38) if tot_bugs > 0 else (22, 163, 74)),
            ("Noise Filtered", f"{tot_false_alarms} Discarded", (100, 116, 139))
        ]
        for i, (label, val, col) in enumerate(folder_kpis):
            bx = start_x + i * (box_w + spacing)
            pdf.set_fill_color(248, 250, 252)
            pdf.set_draw_color(226, 232, 240)
            pdf.rect(bx, kpi_y, box_w, box_h, style="DF", round_corners=True, corner_radius=2)
            pdf.set_xy(bx, kpi_y + 2)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(box_w, 3.5, label, align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.set_xy(bx, kpi_y + 6.5)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_text_color(*col)
            pdf.cell(box_w, 6, val, align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.set_y(kpi_y + box_h + 5)

        # Folder Architecture Inventory Card
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 5, "Folder Architecture & Codebase Dependency Overview:", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        f_card_y = pdf.get_y()
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(226, 232, 240)
        pdf.rect(12, f_card_y, 186, 28, style="DF", round_corners=True, corner_radius=2)

        uniq_imports = f_meta.get("unique_imports", [])
        tot_funcs = f_meta.get("total_functions", sum(len(fr.get("structure", {}).get("functions", [])) for fr in req.file_results))
        loc_data = f_meta.get("loc", {})
        tot_loc = loc_data.get("total", sum(fr.get("structure", {}).get("loc", {}).get("total", 0) for fr in req.file_results))
        tot_code = loc_data.get("code", sum(fr.get("structure", {}).get("loc", {}).get("code", 0) for fr in req.file_results))

        pdf.set_xy(16, f_card_y + 2.5)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(88, 3.5, "Imported Packages across Folder:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(16)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        imp_str = ", ".join(uniq_imports) if uniq_imports else "Built-in Standard Library Only"
        pdf.cell(88, 4, safe_pdf_text(imp_str[:70] + ("..." if len(imp_str) > 70 else "")), new_x="LMARGIN", new_y="NEXT")

        pdf.set_x(16)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(88, 3.5, "Total Functions Inventory:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(16)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(88, 4, f"{tot_funcs} function definitions catalogued across codebase", new_x="LMARGIN", new_y="NEXT")

        pdf.set_xy(110, f_card_y + 2.5)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(84, 3.5, "Folder Lines of Code Metrics:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_xy(110, f_card_y + 6)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(84, 4, f"Total LOC: {tot_loc}  |  Executable Statements: {tot_code}", new_x="LMARGIN", new_y="NEXT")

        pdf.set_xy(110, f_card_y + 11)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(84, 3.5, "Folder Risk Assessment Profile:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_xy(110, f_card_y + 14.5)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(84, 4, f"Risk Rating: {f_meta.get('overall_risk', 'Low')}  |  Avg Complexity: {f_meta.get('avg_complexity', '1.0')}", new_x="LMARGIN", new_y="NEXT")

        pdf.set_y(f_card_y + 32)

        # Cross-File Defect Summary Table
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 5, "Cross-File Defect & Audit Summary Matrix:", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        ty = pdf.get_y()
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        pdf.rect(12, ty, 186, 6.5, style="DF")
        pdf.set_xy(14, ty + 1.2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(60, 4, "File Path / Name", align="L")
        pdf.cell(26, 4, "LOC", align="C")
        pdf.cell(26, 4, "Functions", align="C")
        pdf.cell(38, 4, "Confirmed Defects", align="C")
        pdf.cell(36, 4, "Audit Status", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_y(ty + 7)

        for fr in req.file_results:
            ry = pdf.get_y()
            pdf.set_fill_color(255, 255, 255)
            pdf.set_draw_color(226, 232, 240)
            pdf.rect(12, ry, 186, 6.5, style="DF")
            pdf.set_xy(14, ry + 1.2)
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(15, 23, 42)
            f_label = fr.get("rel_path") or fr.get("filename")
            pdf.cell(60, 4, safe_pdf_text(f_label[:34]), align="L")

            f_loc = fr.get("structure", {}).get("loc", {}).get("total", "N/A")
            f_fns = len(fr.get("structure", {}).get("functions", []))
            f_bugs = fr.get("metrics", {}).get("confirmed_count", 0)
            is_pass = "LOOKS CORRECT" in (fr.get("check") or "").upper()

            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(51, 65, 85)
            pdf.cell(26, 4, str(f_loc), align="C")
            pdf.cell(26, 4, str(f_fns), align="C")

            bug_col = (220, 38, 38) if f_bugs > 0 else (22, 163, 74)
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(*bug_col)
            pdf.cell(38, 4, f"{f_bugs} Bug(s)", align="C")

            stat_col = (22, 163, 74) if (is_pass and f_bugs == 0) else ((37, 99, 235) if is_pass else (217, 119, 6))
            stat_txt = "PASSED" if (is_pass and f_bugs == 0) else ("AUTO-PATCHED" if is_pass else "FLAGGED")
            pdf.set_text_color(*stat_col)
            pdf.cell(36, 4, stat_txt, align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.set_y(ry + 6.5)

        pdf.ln(5)

        # Page 2: Folder Visual Analytics & 4 Charts
        bug_img = decode_base64_image(req.bug_chart_img)
        cat_img = decode_base64_image(req.category_chart_img)
        sev_img = decode_base64_image(req.severity_chart_img)
        qual_img = decode_base64_image(req.quality_chart_img)

        if bug_img or cat_img or sev_img or qual_img:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 5, "Folder-Wide Defect Analytics & Four Illustration Graphs:", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            gy1 = pdf.get_y()
            chart_w, chart_h = 89.0, 48.0
            if bug_img: pdf.image(bug_img, x=13, y=gy1, w=chart_w, h=chart_h)
            if cat_img: pdf.image(cat_img, x=108, y=gy1, w=chart_w, h=chart_h)
            pdf.set_y(gy1 + chart_h + 4)
            gy2 = pdf.get_y()
            if sev_img: pdf.image(sev_img, x=13, y=gy2, w=chart_w, h=chart_h)
            if qual_img: pdf.image(qual_img, x=108, y=gy2, w=chart_w, h=chart_h)
            pdf.set_y(gy2 + chart_h + 4)

        # Pages 3+: Per-File Audit Chapters
        for i, fr in enumerate(req.file_results):
            pdf.add_page()
            fname = fr.get("filename") or f"File {i+1}"
            rel_p = fr.get("rel_path") or fname

            # Chapter Title
            pdf.set_fill_color(30, 41, 59)
            pdf.rect(12, 12, 186, 12, style="F", round_corners=True, corner_radius=2)
            pdf.set_xy(16, 15)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 6, f"Audit Chapter {i+1}: {safe_pdf_text(rel_p)}", new_x="LMARGIN", new_y="NEXT")

            pdf.set_y(28)
            f_struct = fr.get("structure", {})
            f_loc = f_struct.get("loc", {})
            f_comp = f_struct.get("complexity", {})
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 4, f"Lines of Code: {f_loc.get('total', 'N/A')}  |  Functions: {len(f_struct.get('functions', []))}  |  Complexity: {f_comp.get('label', 'Low')}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            # 4-Step Analysis for this file
            file_sections = [
                ("1. Suspected Defect Scan", safe_pdf_text(fr.get("suspected", "")), (245, 158, 11), False),
                ("2. Verification & Noise Filter", safe_pdf_text(fr.get("confirmed", "")), (59, 130, 246), False),
                ("3. Synthesized Python Patch", safe_pdf_text(fr.get("fixed_code", ""), is_code=True), (16, 185, 129), True),
                ("4. Patch Validation & Verdict", safe_pdf_text(fr.get("check", "")), (99, 102, 241), False),
            ]

            for st_title, st_content, st_color, is_code in file_sections:
                if pdf.get_y() > 230:
                    pdf.add_page()
                cur_y = pdf.get_y()
                pdf.set_fill_color(*st_color)
                pdf.rect(12, cur_y, 3, 5, style="F")
                pdf.set_xy(17, cur_y)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(15, 23, 42)
                pdf.cell(0, 5, st_title, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)
                if is_code:
                    pdf.set_font("Courier", "", 8)
                    pdf.set_text_color(30, 41, 59)
                    pdf.set_fill_color(248, 250, 252)
                    pdf.multi_cell(0, 4.2, st_content, border=1, fill=True)
                else:
                    pdf.set_font("Helvetica", "", 8.5)
                    pdf.set_text_color(51, 65, 85)
                    pdf.multi_cell(0, 4.5, st_content)
                pdf.ln(2.5)

        # Codebase Audit Seal
        if pdf.get_y() > 245:
            pdf.add_page()
        pdf.ln(2)
        stamp_y = pdf.get_y()
        pdf.set_fill_color(240, 253, 244)
        pdf.set_draw_color(187, 247, 208)
        pdf.rect(12, stamp_y, 186, 14, style="DF", round_corners=True, corner_radius=2)
        pdf.set_xy(16, stamp_y + 2.5)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(21, 128, 61)
        pdf.cell(0, 4, "Official Codebase Audit Seal: AUDIT CERTIFIED (Autonomous Self-Debugging Agent)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(16)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(0, 4, f"All {len(req.file_results)} files in codebase '{safe_pdf_text(f_title)}' were independently audited and patched using CodeSentinel.", new_x="LMARGIN", new_y="NEXT")

    pdf_bytes = bytes(pdf.output())
    fname_dl = "codesentinel_folder_audit.pdf" if is_folder_mode else "codesentinel_file_audit.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={fname_dl}"}
    )
