"""
NOTE: Nothing containerized about this any more. This is just a helper
function that evaluates a string script in a given language. The function
is used by the custom_metrics/multiple_metrics/evaluation.py script to
evaluate the output of a model on a given problem. This function is
called by the cached_eval_script function in the same script.
"""

import tempfile
from pathlib import Path

from . import (
    eval_clj,
    eval_cpp, 
    eval_cs,
    eval_dlang, 
    eval_fs,
    eval_go, 
    eval_hs,
    eval_java, 
    eval_javascript, 
    eval_julia,
    eval_lua, 
    eval_ocaml,
    eval_php, 
    eval_pl, 
    eval_python, 
    eval_r, 
    eval_ruby,
    eval_rust, 
    eval_scala, 
    eval_sh, 
    eval_swift, 
    eval_ts, 
)

EVALUATORS = {
    "clj": (eval_clj.eval_script, ".clj"),
    "cpp": (eval_cpp.eval_script, ".cpp"),
    "cs": (eval_cs.eval_script, ".cs"),
    "d": (eval_dlang.eval_script, ".d"),
    "fs": (eval_fs.eval_script, ".fsx"),
    "go": (eval_go.eval_script, "_test.go"),
    "hs": (eval_hs.eval_script, ".hs"),
    "java": (eval_java.eval_script, ".java"),
    "javascript": (eval_javascript.eval_script, ".js"),
    "jl": (eval_julia.eval_script, ".jl"),
    "js": (eval_javascript.eval_script, ".js"),
    "lua": (eval_lua.eval_script, ".lua"),
    "ml": (eval_ocaml.eval_script, ".ml"),
    "php": (eval_php.eval_script, ".php"),
    "pl": (eval_pl.eval_script, ".pl"),
    "py": (eval_python.eval_script, ".py"),
    "python": (eval_python.eval_script, ".py"),
    "r": (eval_r.eval_script, ".r"),
    "rb": (eval_ruby.eval_script, ".rb"),
    "rs": (eval_rust.eval_script, ".rs"),
    "rust": (eval_rust.eval_script, ".rs"),
    "scala": (eval_scala.eval_script, ".scala"),
    "sh": (eval_sh.eval_script, ".sh"),
    "swift": (eval_swift.eval_script, ".swift"),
    "ts": (eval_ts.eval_script, ".ts"),
}


def eval_string_script(language, program):
    if language in EVALUATORS:
        (eval_script, file_ext) = EVALUATORS[language]
    else:
        raise ValueError(f"Unsupported language: {language}")
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=True) as f:
        f.write(program.encode("utf-8"))
        f.flush()
        result = eval_script(Path(f.name))
        # Only save the first 4K of output from the running program. Any futher
        # output is very likely an exceptionally long stack trace or a long
        # series of prints.
        if type(result["stdout"]) == bytes:
            result["stdout"] = result["stdout"].decode("utf-8", errors="ignore")
        if result["stdout"] is None:
            result["stdout"] = ""
        if result["stderr"] is None:
            result["stderr"] = ""
        if type(result["stderr"]) == bytes:
            result["stderr"] = result["stderr"].decode("utf-8", errors="ignore")
        assert type(result["stdout"]) == str
        assert type(result["stderr"]) == str
        return {
            "program": program,
            "stdout": result["stdout"].replace("!!int", "")[:4096],
            "stderr": result["stderr"][:4096],
            "exit_code": result["exit_code"],
            "status": result["status"],
        }
