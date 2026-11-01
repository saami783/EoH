from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Codex Exec Wrapper for EoH")

ReturnType = Literal["text", "json", "markdown"]


class CodexRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    repeat_prompt: int | None = 1
    params: dict[str, Any] | None = None
    model: str | None = None
    temperature: float | None = None
    return_type: ReturnType = "text"


class CodexResponse(BaseModel):
    content: list[str]
    success: bool = True
    response: Any = None
    return_type: ReturnType = "text"
    model: str | None = None


def find_codex_executable() -> str:
    env_path = os.getenv("CODEX_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    found_cmd = shutil.which("codex.cmd")
    if found_cmd:
        return found_cmd

    found_plain = shutil.which("codex")
    if found_plain:
        return found_plain

    candidates = [
        os.path.expandvars(r"%APPDATA%\npm\codex.cmd"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Roaming\npm\codex.cmd"),
        os.path.expandvars(r"%APPDATA%\npm\codex"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Roaming\npm\codex"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise RuntimeError(
        "Impossible de trouver `codex`. Définis CODEX_PATH vers le chemin complet de codex.cmd."
    )


def run_codex_exec(prompt: str, model: str | None = None) -> str:
    codex_bin = find_codex_executable()

    base_cmd = [codex_bin, "exec", "--skip-git-repo-check"]
    if model:
        base_cmd += ["-m", model]
    base_cmd.append("-")

    if codex_bin.lower().endswith(".cmd"):
        cmd = ["cmd", "/c"] + base_cmd
    else:
        cmd = base_cmd

    # Empêche l'agent CLI Codex de créer/modifier des fichiers sur le PC
    # lorsqu'EoH lui demande d'implémenter une fonction Python.
    if prompt.strip() != "1+1=?":
        effective_prompt = (
            "IMPORTANT: Do NOT read, create, or edit any files, and do NOT run shell commands. "
            "Reply ONLY in plain text directly to stdout following the exact format requested below.\n\n"
            f"{prompt}"
        )
    else:
        effective_prompt = prompt

    try:
        result = subprocess.run(
            cmd,
            input=effective_prompt.encode("utf-8"),
            capture_output=True,
            timeout=600,
            shell=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Exécutable introuvable: {codex_bin}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Timeout : `codex exec` a mis trop de temps à répondre.") from e

    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()

    if result.returncode != 0:
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"Erreur `codex exec` : {detail}")

    if not stdout:
        raise RuntimeError("`codex exec` a renvoyé une sortie vide.")

    return stdout


@app.get("/health")
def health():
    try:
        codex_path = find_codex_executable()
        return {"status": "ok", "codex_found": True, "codex_path": codex_path}
    except Exception as e:
        return {"status": "degraded", "codex_found": False, "detail": str(e)}


@app.post("/api/chat/codex", response_model=CodexResponse)
def chat_codex(req: CodexRequest):
    print(f"REQ [Codex] - Prompt length: {len(req.prompt)}")

    try:
        raw_output = run_codex_exec(req.prompt, req.model)

        parsed_content: Any = raw_output
        if req.return_type == "json":
            try:
                parsed_content = json.loads(raw_output)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "La réponse de Codex n'est pas un JSON valide",
                        "raw_output": raw_output,
                        "exception": str(e),
                    },
                ) from e

        print("RAW OUTPUT:", repr(raw_output[:200] + "..." if len(raw_output) > 200 else raw_output))

        return CodexResponse(
            content=[raw_output],
            success=True,
            response=parsed_content,
            return_type=req.return_type,
            model=req.model,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)}) from e