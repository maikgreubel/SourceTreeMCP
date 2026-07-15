import os
import logging
import argparse
import mimetypes
import datetime
import lizard
import pandas as pd
import multiprocessing
import re
import json

from llama_cpp import Llama
from pygount import SourceAnalysis, ProjectSummary
from git import Repo
from pathlib import Path
from collections import Counter, deque
from fastmcp import FastMCP
from typing import Dict, Annotated
from pydantic import Field

# The to llama-cpp-python compatible model
MODEL_PATH = "models/qwen3-moe-4x4b-16b-jan-polaris-instruct-power-house-q5_k_m.gguf"

# We only allow to use maximum cores minus two in order to provide OS stability
# and use as much resources as possible
N_THREADS = multiprocessing.cpu_count() - 2

# Size of context
N_CONTEXT = 131072

companion_llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CONTEXT,
    n_threads=N_THREADS,
    n_gpu_layers=0, # We want to use only CPU
    verbose=True
)

logger = logging.getLogger(__name__)

mcp:FastMCP = FastMCP("Source Tree Server")

basedir = "unsloth/Llama-3.3-70B-Instruct-GGUF"

def sanitize_path(path: str) -> str:
    """
    Sanitize the given path by replacing any backslashes with forward slashes. Also checking for directory traversal.

    Args:
        path (str): The path to sanitize.

    Returns:
        str: The sanitized path.
    """
    if not basedir:
        raise ValueError("Base directory (basedir) is not set.")
    
    # Replace backslashes with forward slashes
    realPath = os.path.realpath(path).replace("\\", "/")
    
    # Normalize both paths to handle any symbolic links
    basePath = os.path.abspath(os.path.realpath(basedir)).replace("\\", "/")
    realPath = os.path.abspath(realPath).replace("\\", "/")
    
    # Check if the sanitized path starts with the base directory
    if not realPath.startswith(basePath):
        raise ValueError("Path traversal detected")
    
    return realPath

def get_files_from_git_tree(path: str, file_extension:str = "", recursive: bool = True) -> list[str]:
    """
    Get a list of all files in the given Git tree.

    Args:
        path (str): The path to the Git tree.
        file_extension (str): Optional file extension filter (e.g., ``.txt``).
        recursive (bool): If True, traverse recursively; if False, only top-level files.

    Returns:
        list[str]: A list of file paths or an empty list in case nothing found or error.
    """
    try:
        repo = Repo(os.path.abspath(basedir))
        
        entries:list[str]  = []
        
        for entry in repo.commit().tree.traverse():
            epath = str(entry.path) # type: ignore
            # Check if path matches (empty path means all files from basedir)
            path_matches = len(path) == 0 or epath.startswith(path)
            # Check if extension matches (empty extension means all files)
            ext_matches = file_extension == "" or epath.endswith(file_extension)
            
            if path_matches and ext_matches:
                # For non-recursive, only include files directly under the path
                if recursive:
                    entries.append(epath)
                else:
                    # Calculate depth - count slashes after the path prefix
                    if len(path) == 0:
                        # Empty path means basedir, check if file is at root level
                        if epath.count('/') == 0:
                            entries.append(epath)
                    else:
                        # For a specific path, check if it's one level deep
                        suffix = epath[len(path):]
                        if suffix.startswith('/') and suffix.count('/') == 1:
                            entries.append(epath)
        return entries
    except Exception as e:
        logger.error(e, exc_info=True)
        return []

@mcp.tool()
def get_file_size(
    path: Annotated[str, Field(description="The file inside of source tree to get the size of or -1 in case of error.")]
) -> int:
    """Get the size of given file in bytes."""
    logger.info(f"get_file_size called with path {path}")

    try:
        path = sanitize_path(os.path.join(basedir, path))
        return os.path.getsize(path)
    except Exception as e:
        logger.error(e, exc_info=True)
        return -1
    
def is_git_repo() -> bool:
    try:
        return os.path.exists(os.path.join(basedir, ".git"))
    except Exception as e:
        logger.error(e, exc_info=True)
        return False

def transpile_code(
    code: Annotated[str, Field(description="The C++ code to transpile.")]
) -> str:
    """Transpiles C++ code into programming language C# using companion LLM."""
    result = ""
    
    logger.info("Have been asked to transpile code")

    try:
        
        prompt = f"""
        You act as a expert coding companion for an agent, in order to understand, analyze and transpile
        code into C# programming language.
        
        C++-Code:
        {code}
        
        C# ported code:
        """
        
        output = companion_llm(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.2,
            stop=["\nC++-Code:"],
            echo=False
        )
        
        logger.info("The transpiled code is read")
        
        result = f"\n--- Expert companion ported code ---{output["choices"][0]["text"]}"

    except Exception as e:
        logger.error(e, exc_info=True)

    return result

def ask_companion(
    question: Annotated[str, Field(description="The question to ask.  Max context size 131072.")]
) -> str:
    """Ask the companion LLM a question and returns the answer. The companion LLM acts as a expert in programming languages especially C++ and C#. It accepts a maximum context size of 131072."""
    result = ""
    
    logger.info("Have been asked for a question")
    
    try:
        prompt = f"""
        You are an expert for programming languages, especially for C++ and C#. You are also
        have an extensive knowledge of software design patterns and best practices. Transpiling
        and portation of existing code into other languages is also part of your expertise.
        
        
        Question:
        {question}
        
        Answer:
        """
        
        output = companion_llm(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.2,
            stop=["\nQuestion:"],
            echo=False
        )
        
        result = f"\n--- Expert companion expertise ---{output["choices"][0]["text"]}"
        
        logger.info(f"My answer is ready:{result}")
    except Exception as e:
        logger.error(e, exc_info=True)

    return result

@mcp.tool()
def get_files(
    path: Annotated[
        str,
        Field(description="Optional path to the sub directory in basedir from which files should be listed. If an empty string is passed, the basedir is used as path. If a non empty string is passed, it will be joined to basedir global parameter.")
    ],
    file_extension: Annotated[
        str,
        Field(description="Optional file extension filter (e.g., ``.txt``). Files must end with this extension to be included in the result. If an empty string is passed, all files are returned.")
    ],
    recursive: Annotated[
        bool,
        Field(description="If True, traverse recursively through subdirectories. If False, only list files in the specified directory (non-recursive). Default is True.")
    ] = True
) -> list[str]:
    """Retrieve a list of file names in *folder_path*, which is, if provided, a directory inside the global runtime parameter basedir provided at start of server, that have the specified *file_extension*. If no path is provided, all files in the global basedir are listed."""
    
    files:list[str] = []
    try:
        if is_git_repo():
            files = get_files_from_git_tree(path, file_extension, recursive)
        else:
            dir = sanitize_path(os.path.join(basedir, path))
            
            logger.info(f"About to retrieve files from {dir}")
            
            entries = []
            
            if os.path.exists(dir):
                if recursive:
                    # Recursive traversal using os.walk
                    for root, _, filenames in os.walk(dir):
                        for filename in filenames:
                            full_path = os.path.join(root, filename)
                            if file_extension == "" or filename.endswith(file_extension):
                                rel_path = os.path.relpath(full_path, dir).replace("\\", "/")
                                files.append(rel_path)
                else:
                    # Non-recursive: only list top-level files
                    entries = os.listdir(dir)
                    if file_extension == "":
                        files = [f for f in entries if os.path.isfile(os.path.join(dir, f))]
                    else:
                        files = [f for f in entries if f.endswith(file_extension) and os.path.isfile(os.path.join(dir, f))]
    except Exception as e:
        logger.error(e, exc_info=True)

    return files

@mcp.tool()
def get_file_info(
    path: Annotated[str, Field(description="Path to the file for which information should be retrieved.")] 
) -> Dict[str, str]:
    """Retrieve information about the file located at *path*, which is a directory inside
    the global runtime parameter basedir provided at start of server. The returned dictionary will include the following keys:

    - `name`: The name of the file.
    - `size`: The size of the file in bytes.
    - `modified_time`: The timestamp when the file was last modified, in ISO format (YYYY-MM-DDTHH:MM:SS).
    - `type`: The mime type of file if it can be detected, as fallback 'application/octet-stream'.
    - `encoding`: In case of a text like file, the character encoding is tried to be determined, otherwise the field won't exist in result dict. 
    """
    logger.info(f"About to retrieve file info from {path}")

    fullPath = sanitize_path(os.path.join(basedir, path))

    if not os.path.isfile(fullPath):
        return {}

    try:
        mimeType, _ = mimetypes.guess_type(fullPath)

        # Try to detect encoding for text-like files
        encoding = None
        if mimeType and mimeType.startswith("text/"):
            try:
                # Read a small sample to detect encoding
                with open(fullPath, 'rb') as f:
                    raw_data = f.read(1024)  # read first 1KB
                
                # Simple heuristic: if all bytes are ASCII-compatible, assume UTF-8
                try:
                    raw_data.decode('utf-8')
                    encoding = 'utf-8'
                except UnicodeDecodeError:
                    # Fallback to latin-1 which can decode any byte sequence
                    raw_data.decode('latin-1')
                    encoding = 'latin-1'
            except Exception as e:
                logger.warning(f"Could not detect encoding for {fullPath}: {e}")
                encoding = 'unknown'


        stats = os.stat(fullPath)
        info = {
            'name': os.path.basename(fullPath),
            'size': str(stats.st_size),
            'modified_time': datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'type': str(mimeType) if mimeType else 'application/octet-stream',
        }

        # Add encoding only for text files where it's detected
        if encoding:
            info['encoding'] = encoding

        return info
    except Exception as e:
        logger.error(e, exc_info=True)
        return {}
    
@mcp.tool()
def get_file_content(
    path: Annotated[str, Field(description="Path to the file from which content should be retrieved.")],
    start_line: Annotated[int, Field(description="The line number where to start reading.", ge=0)] = 0,
    end_line: Annotated[int, Field(description="The line number where to stop reading.", ge=0)] = 0
) -> str:
    """Retrieve the content of the file located at *path*, which is a directory inside the global runtime parameter basedir provided at start of server. The returned string will contain the contents of the file."""
    logger.info(f"About to retrieve file content from {path}")

    fullPath = sanitize_path(os.path.join(basedir, path))

    if not os.path.isfile(fullPath):
        error_msg = f"File not found: {fullPath}"
        logger.warning(error_msg)
        return error_msg

    content = ""
    try:    
        if start_line > 0 and end_line > 0 and start_line < end_line:
            with open(fullPath, 'r', encoding='utf-8', errors='replace') as fp:
                for i, line in enumerate(fp):
                    if i >= start_line and i <= end_line:
                        content = content + line + "\n"
        else:
            # Try to read with UTF-8 first, then fallback to latin-1 which can handle any byte sequence
            try:
                content = Path(fullPath).read_text(encoding='utf-8', errors='replace')
            except UnicodeDecodeError:
                # Fallback: use latin-1 which maps bytes 0-255 directly to characters
                with open(fullPath, 'r', encoding='latin-1', errors='replace') as fp:
                    content = fp.read()
    except Exception as e:
        error_msg = f"Error reading file {fullPath}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

    return content

@mcp.tool()
def grep(
    path: Annotated[str, Field(description="The path inside the source tree to search in.")],
    pattern: Annotated[str, Field(description="The pattern to search for.")],
    lines_before: Annotated[int, Field(description="The number of lines to return before the pattern.", ge=0)] = 0,
    lines_after: Annotated[int, Field(description="The number of lines to return after the pattern.", ge=0)] = 0
) -> list[tuple[int,str]]:
    """Searches for pattern in lines and returning them using their line number. Additionally, a specified number of lines before and after the matching line can be included in the result. The returned list contains tuples of line number and line content."""
    path = sanitize_path(os.path.join(basedir, path))
    
    result:list[tuple[int,str]] = []
    
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
            for i, line in enumerate(lines):
                match = re.search(pattern, line)
                if match is not None:
                    # Add lines before
                    start_idx = max(0, i - lines_before)
                    for j in range(start_idx, i):
                        result.append((j + 1, lines[j].rstrip('\n')))
                    
                    # Add the matching line
                    result.append((i + 1, line.rstrip('\n')))
                    
                    # Add lines after
                    end_idx = min(len(lines), i + lines_after + 1)
                    for j in range(i + 1, end_idx):
                        result.append((j + 1, lines[j].rstrip('\n')))
    except Exception as e:
        logger.error(f"Error in grep for pattern '{pattern}' in path '{path}'")
        logger.error(e, exc_info=True)
    
    return result

@mcp.tool()
def get_directories(
    path: Annotated[str, Field(description="The path inside the source tree to search in.")]
) -> list[str]:
    """Retrieve a list of directory names in *path*, which is, if provided, a subdirectory inside the global runtime parameter basedir provided at start of server. If an empty string is passed, the basedir is used as path. If a non-empty string is passed, it will be joined to basedir global parameter."""
    logger.info(f"About to retrieve directories from {path}")

    fullPath = sanitize_path(os.path.join(basedir, path))
    
    if not os.path.isdir(fullPath):
        return []
    
    try:
        return [d for d in os.listdir(fullPath) if os.path.isdir(os.path.join(fullPath, d))]
    except Exception as e:
        logger.error(e, exc_info=True)
        return []

def detect_languages(
    path: Annotated[str, Field(description="The path inside the source tree to search in.")],
    languageCounter: Annotated[Counter[str], Field(description="A counter object that keeps track of the number of files found for each language.")]
) -> Counter[str]:
    """Retrieve a list of directory names in *path*, which is, if provided, a subdirectory inside the global runtime parameter basedir provided at start of server. If an empty string is passed, the basedir is used as path. If a non-empty string is passed, it will be joined to basedir global parameter. The returned list contains the names of the directories located directly under *path*."""
    languageMap = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c' : 'C',
        '.cs' : 'C#',
        '.rb' : 'Ruby',
        '.php' : 'PHP',
        '.go' : 'Go',
        '.ts' : 'TypeScript',
        '.rs' : 'Rust',
        '.swift' : 'Swift'
    }
    
    numFiles = 0
    
    path = sanitize_path(os.path.join(basedir, path))

    try:
        for root, _, files in os.walk(path): # type: ignore
            for file in files:
                if os.path.isdir(os.path.join(path, file)):
                    detect_languages(os.path.join(path, file), languageCounter)
                else:
                    _, ext = os.path.splitext(file)
                    if ext in languageMap:
                        languageCounter[languageMap[ext]] += 1
                        numFiles = numFiles + 1
        return languageCounter
    except Exception as e:
        logger.error(e, exc_info=True)
        return Counter()


def get_language_stats() -> dict[str,str]:
    """Retrieve a dictionary of languages and the count of files in each language in the global runtime parameter basedir provided at start of server."""
    try:
        languageCounter:Counter[str] = Counter()
        
        languageCounter = detect_languages(basedir, languageCounter)
        
        languages:dict[str,str] = {}
        
        for lang, count in languageCounter.most_common():
            languages[lang] = str(count)
        
        return languages
    except Exception as e:
        logger.error(e, exc_info=True)
        return {}

def get_source_extensions() -> list[str]:
    """Retrieve a list of source file extensions found in the global runtime parameter basedir provided at start of server. This method is used to determine which files should be analyzed for code metrics and other code related information. The returned list contains the file extensions without the dot, e.g., "py" for Python files."""
    try:
        languages = get_language_stats()
        
        ext:list[str] = []
        match(next(iter(languages.keys()))):
            case "Java":
                ext.append("java")
            case "C++":
                ext.append("cpp")
                ext.append("hpp")
                ext.append("c")
                ext.append("h")
            case "C":
                ext.append("c")
            case "PHP":
                ext.append("php")
            case "Python":
                ext.append("py")
            case _:
                pass
        return ext                        
    except Exception as e:
        logger.error(e, exc_info=True)
        return []

@mcp.tool()
def get_languages() -> dict[str,str]:
    """Retrieve a dictionary of languages used inside the source tree in the global runtime parameter basedir provided at start of server. The returned dictionary will include the languages as keys and the number of files found for each language as values (as string)."""
    try:
        return get_language_stats()
    except Exception as e:
        logger.error(e, exc_info=True)
        return {}

@mcp.tool()
def get_repo_info() -> dict[str,str]:
    """Retrieve information about the git repository located at the global runtime parameter basedir provided at start of server. The returned dictionary will include the following keys:

    - `currentBranch`: The name of the current branch
    - `lastCommit`: The hash of the last commit
    - `author`: The author of the last commit
    - `date`: The date and time of last commit
    - `message`: The commit message of the last commit
    - `branches`: A list of all branch names in the repository.
    - `last_five_commits`: A list of the last five commit hashes.
    """
    repoInfo:dict[str,str] = {}
    
    try:
        gitDir = sanitize_path(os.path.join(basedir, '.git'))
        
        if os.path.exists(gitDir) and os.path.isdir(gitDir):
            repo = Repo(basedir)
            current_branch = repo.head.ref.name
            repoInfo['currentBranch'] = current_branch
            commit = repo.head.commit
            repoInfo['lastCommit'] = commit.hexsha
            repoInfo['author'] = f"{commit.author.name} <{commit.author.email}>"
            repoInfo['date'] = commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S")
            repoInfo['message'] = str(commit.message)
            
            branches = ""
            for branch in repo.branches:
                branches = f"{branches}{branch.name}, "
            branches = branches[:-2] # remove
            
            repoInfo['branches'] = branches

            commits = ""
            for c in repo.iter_commits(current_branch, max_count=5):
                commits = f"{commits}{c.hexsha[:7]} - {c.author.name} - {c.summary}, "
            commits = commits[:-2] # remove
            
            repoInfo['last_five_commits'] = commits
    except Exception as e:
        logger.error(e, exc_info=True)

    return repoInfo

def lizard_analysis_to_dataframe(
    results: Annotated[list[lizard.FileInformation], Field(description="A list of Lizard file analysis results to convert to a pandas DataFrame. Each item in the list should be an instance of lizard.FileInformation, which contains information about a single file and its functions. The DataFrame will contain the following columns: filename, max_ccn, nloc, func_count.")],
) -> pd.DataFrame:
    """Convert a list of Lizard file analysis results to a pandas DataFrame. The DataFrame will contain the following columns: filename, max_ccn, nloc, func_count."""
    records:list[dict[str,str]] = []
    
    try:
        for fileinfo in results:
            nloc_file = fileinfo.nloc # type: ignore
            max_ccn:int = 0
            
            for func_info in fileinfo.function_list: # type: ignore
                if int(func_info.cyclomatic_complexity) > max_ccn: # type: ignore
                    max_ccn = int(func_info.cyclomatic_complexity) # type: ignore

            records.append({
                'filename': fileinfo.filename, # type: ignore
                'max_ccn': str(max_ccn),
                'nloc': nloc_file,
                'func_count': str(len(fileinfo.function_list)) # type: ignore
            })
    except Exception as e:
        logger.error(e, exc_info=True)

    return pd.DataFrame(records)

def get_files_for_extension(
    path: Annotated[str, Field(description="The path to the directory from which to retrieve files.")],
    exts: Annotated[list[str], Field(description="A list of file extensions to filter by (e.g., ``['.txt', '.md']``).")]
) -> list[str]:
    """Retrieve a list of file names in *path*, which is, if provided, a directory inside the global runtime parameter basedir provided at start of server, that have any of the specified *exts*. If no path is provided, all files in the global basedir are listed. The returned list contains the file paths of the files that match the given extensions."""
    logger.info(f"About to get files for extensions in path {path}")
    
    filesToAnalyze:list[str] = []
    
    try:
        if is_git_repo():
            repo = Repo(".")
            for entry in repo.commit().tree.traverse():
                file = str(entry.abspath) # type: ignore
                if file.endswith(tuple([f".{ext}" for ext in exts])):
                    filesToAnalyze.append(file)
        else:
            sanitize_path(os.path.join(basedir, path))
            
            for root, _, files in os.walk(path): # type: ignore
                for file in files:
                    entry = os.path.join(root, file).replace("\\", "/")
                    if entry.find("/.git/") > 0:
                        continue
                    
                    logger.debug(f"About to check file: {entry}")
                    
                    if not os.path.isdir(entry):
                        if file.endswith(tuple([f".{ext}" for ext in exts])):
                            logger.debug(f"Adding {entry} to list of files to analyze")
                            filesToAnalyze.append(entry)
    except Exception as e:
        logger.error(e, exc_info=True)

    return filesToAnalyze

@mcp.tool()
def get_code_metrics(
    path: Annotated[str, Field(description="The path to the directory for which to retrieve code metrics.")]) -> dict[str, str]:
    """Retrieve code metrics for the complete code base. The analysis is performed on all files in the global runtime parameter basedir provided at start of server that match the source file extensions found in the source tree. The code metrics include total number of files, total lines of code, average lines of code per file, and a breakdown of languages used in the source tree with their respective file counts.

    The returned dictionary will include the following keys:

    - `total_files`: The total number of files in the source tree.
    - `total_lines`: The total number of lines of code in the source tree.
    - `average_loc_per_file`: The average number of lines of code per file in the source tree.
    - `languages`: A dictionary containing the number (as string) of files found for each language.
    """

    path = sanitize_path(os.path.join(basedir, path))
    try:
        filesToAnalyze = get_files_for_extension(path, get_source_extensions())
        logger.info(f"About to analyze {len(filesToAnalyze)} files")
        
        fi:list[lizard.FileInformation] = []
        
        for file in filesToAnalyze:    
            fi.append(lizard.analyze_file.analyze_source_code(file, Path(file).read_text())) # type: ignore

        df = lizard_analysis_to_dataframe(fi)
        
        # Calculate additional metrics
        total_files = len(filesToAnalyze)
        total_lines = df['nloc'].astype(int).sum() if not df.empty else 0
        average_loc_per_file = round(total_lines / total_files, 2) if total_files > 0 else 0
        
        languages = get_language_stats()
        
        return {
            'total_files': str(total_files),
            'total_lines': str(total_lines),
            'average_loc_per_file': str(average_loc_per_file),
            'languages': json.dumps(languages)
        }
    except Exception as e:
        logger.error(e, exc_info=True)
        return {}
    
@mcp.tool()
def get_line_counts() -> dict[str, int]:
    """Retrieve a dictionary of file names and their corresponding line counts in the global runtime parameter basedir provided at start of server. The returned dictionary will have file names as keys and their respective line counts as values (as integers)."""
    result: dict[str, int] = {}
    try:
        filesToAnalyze = get_files_for_extension(basedir, get_source_extensions())
        logger.info(f"About to analyze {len(filesToAnalyze)} files")
    
        projectSummary = ProjectSummary()
        
        for file in filesToAnalyze:
            analysis = SourceAnalysis.from_file(file, group="pygount") # type: ignore
            projectSummary.add(analysis) # type: ignore

        
        # Add total counts
        result['total_files'] = len(filesToAnalyze)
        result['total_lines'] = sum(a.lines for a in projectSummary.sources) if hasattr(projectSummary, 'sources') else 0
        
        # If individual file line counts are needed:
        for source_analysis in getattr(projectSummary, 'sources', []):
            filename = os.path.basename(source_analysis.source_path) if hasattr(source_analysis, 'source_path') else str(source_analysis)
            result[filename] = source_analysis.lines if hasattr(source_analysis, 'lines') else 0
    except Exception as e:
        logger.error(e, exc_info=True)

    return result

@mcp.tool()
def get_last_n_commits(
    count: Annotated[int, Field(description="The number of commits to retrieve.")] = 10) -> list[str]:
    """Retrieve a list of the last *count* commit hashes in the global runtime parameter basedir provided at start of server. The returned list contains the commit hashes as strings."""
    commits:list[str] = []

    try:
        gitDir = os.path.join(basedir, '.git')
        
        if os.path.exists(gitDir) and os.path.isdir(gitDir):
            repo = Repo(basedir)
            
            for commit in repo.iter_commits(max_count=count):
                commits.append(str(commit))
    except Exception as e:
        logger.error(e, exc_info=True)

    return commits

@mcp.tool()
def get_diff_for_commit(
    older_commit_hash: Annotated[str, Field(description="The hash of the older commit.")] = 'HEAD~1',
    newer_commit_hash: Annotated[str, Field(description="The hash of the newer commit.")] = 'HEAD'
) -> str:
    """Retrieve the diff between two commits in the global runtime parameter basedir provided at start of server. The returned string contains the diff in unified format."""
    try:
        gitDir = os.path.join(basedir, '.git')
        
        if os.path.exists(gitDir) and os.path.isdir(gitDir):
            repo = Repo(basedir)
            diffs = repo.commit(older_commit_hash).diff(newer_commit_hash, create_patch=True)
            diff = ""
            for diffitem in diffs:
                diff = diff + str(diffitem)

            return diff
    except Exception as e:
        logger.error(e, exc_info=True)

    return "No git repository found in the basedir."

@mcp.tool()
def search_commits_containing_change(
    pattern: Annotated[str, Field(description="The pattern to search for in commit messages or diffs.")]
) -> str:
    """Retrieve a list of commit hashes that contain changes matching the given pattern in the global runtime parameter basedir provided at start of server. The search is performed in both commit messages and diffs. The returned string contains the matching commits in a human readable format."""
    commits:str = ""
    try:
        gitDir = os.path.join(basedir, '.git')
        if os.path.exists(gitDir) and os.path.isdir(gitDir):
            repo = Repo(basedir)
            commits = repo.git.log(G=pattern, pretty='oneline')
    except Exception as e:
        logger.error(e, exc_info=True)

    return commits

@mcp.tool()
def make_directory(
    path: Annotated[str, Field(description="The path where the new directory should be created.")]
) -> str:
    """Create a new directory at the specified path. It will create all sub directories neccessary as well. The path is relative to the global runtime parameter basedir provided at start of server. The returned string indicates whether the directory was successfully created or not."""
    path = sanitize_path(os.path.join(basedir, path))
    try:
        os.makedirs(path, exist_ok=True)
        return f"Directory {path} created successfully."
    except Exception as e:
        logger.error(e, exc_info=True)
        return f"Failed creating directory {path}"

@mcp.tool()
def write_file(
    path: Annotated[str, Field(description="The path where the file should be written.")],
    content: Annotated[str, Field(description="The content to write to the file.")],
    append: Annotated[bool, Field(description="If True, the content will be appended to the file if it exists. If False, the file will be overwritten.")] = False
) -> str:
    """Write the given content to a file at the specified path. Creates parent directories if necessary. The path is relative to the global runtime parameter basedir provided at start of server. The returned string indicates whether the file was successfully written or not."""
    try:
        path = sanitize_path(os.path.join(basedir, path))
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            logger.info(f"Creating directory {parent_dir}.")
            os.makedirs(parent_dir)
        with open(path, 'a' if append else 'w') as f:
            f.write(content)
        return "File written successfully."
    except Exception as e:
        logger.error(e, exc_info=True)
        return f"Error writing file {path}"

@mcp.tool()
def delete_file(
    path: Annotated[str, Field(description="The path to the file that should be deleted.")]
) -> str:
    """Delete the file located at the specified path. The path is relative to the global runtime parameter basedir provided at start of server. The returned string indicates whether the file was successfully deleted or not."""
    path = sanitize_path(os.path.join(basedir, path))
    try:
        os.remove(path)
        return f"File {path} deleted successfully"
    except Exception as e:
        logger.error(e, exc_info=True)
        return f"Error deleting file {path}"
    
@mcp.tool()
def read_online_pdf(
    url: Annotated[str, Field(description="The URL of the PDF file to read.")]
) -> str:
    """Read the content of a PDF file from an online URL. The returned string contains the content of the PDF file. This method will download the PDF file from the given URL, convert it to markdown format using the DocumentConverter from docling, and return the converted content as a string. In case of any error during the process, an empty string is returned."""
    try:
        import tempfile
        import requests
        from docling.document_converter import DocumentConverter
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        r = requests.get(url=url, stream=True)
        with open(tmpfile.name, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        converter = DocumentConverter()
        documents = converter.convert_all(tmpfile.name)
        result = ""
        for doc in documents:
            result = doc.document.export_to_markdown()
        return result
    except Exception as e:
        logger.error(e, exc_info=True)
        return ""

def extract_c_dependencies(content: str) -> list[str]:
    """
    Extract C/C++ include dependencies from file content.
    
    Args:
        content (str): The source code content to analyze.
    
    Returns:
        list[str]: List of included file paths (e.g., ["header.h", "utils.cpp"])
    """
    import re
    pattern = r'#include\s*[<"]([^>"]+)[>"]'
    matches = re.findall(pattern, content)
    return matches

def extract_csharp_dependencies(content: str) -> list[str]:
    """
    Extract C# using dependencies from file content.
    
    Args:
        content (str): The source code content to analyze.
    
    Returns:
        list[str]: List of imported namespaces/modules (e.g., ["System", "System.Collections"])
    """
    import re
    pattern = r'using\s+([^\s;]+)'
    matches = re.findall(pattern, content)
    return matches

def extract_python_dependencies(content: str) -> list[str]:
    """
    Extract Python import dependencies from file content.
    
    Args:
        content (str): The source code content to analyze.
    
    Returns:
        list[str]: List of imported modules (e.g., ["os", "sys", "module"])
    """
    import re
    # Match both 'import x' and 'from x import y'
    pattern = r'(?:import\s+([^\s,]+)|from\s+([^\s]+))'
    matches = re.findall(pattern, content)
    # Flatten the list of tuples and filter empty strings
    return [m[0] or m[1] for m in matches if m[0] or m[1]]

def extract_js_dependencies(content: str) -> list[str]:
    """
    Extract JavaScript/TypeScript import dependencies from file content.
    
    Args:
        content (str): The source code content to analyze.
    
    Returns:
        list[str]: List of imported modules (e.g., ["./helper", "lodash"])
    """
    import re
    # Match 'import ... from "module"' and 'import "module"'
    pattern = r'(?:import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]|import\s+[\'"]([^\'"]+)[\'"])'
    matches = re.findall(pattern, content)
    return [m[0] or m[1] for m in matches if m[0] or m[1]]

def extract_lua_dependencies(content: str) -> list[str]:
    """
    Extract Lua require dependencies from file content.
    
    Args:
        content (str): The source code content to analyze.
    
    Returns:
        list[str]: List of required modules (e.g., ["module", "./helper", "../utils"])
    """
    import re
    # Match 'require "module"' and 'require (module)' patterns
    pattern = r'require\s*[\'"]([^\'"]+)[\'"]|require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
    matches = re.findall(pattern, content)
    return [m[0] or m[1] for m in matches if m[0] or m[1]]

def detect_circular_dependencies(graph: Dict[str, list[str]]) -> list[list[str]]:
    """
    Detect circular dependencies in the dependency graph using DFS.
    
    Args:
        graph (Dict[str, list[str]]): The dependency graph as adjacency list.
    
    Returns:
        list[list[str]]: List of cycles found, each cycle is a list of file paths.
    """
    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []
    
    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
        
        path.pop()
        rec_stack.remove(node)
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return cycles

def build_dependency_graph(path: str, extensions: list[str]) -> Dict[str, list[str]]:
    """
    Build a dependency graph for all files with given extensions in the path.
    
    Args:
        path (str): The base directory path.
        extensions (list[str]): List of file extensions to analyze (e.g., ['.cpp', '.h', '.py']).
    
    Returns:
        Dict[str, list[str]]: The dependency graph as adjacency list.
    """
    import os
    from pathlib import Path
    
    graph: Dict[str, list[str]] = {}
    
    try:
        for root, _, files in os.walk(path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in extensions:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, path).replace("\\", "/")
                    
                    try:
                        content = Path(full_path).read_text(encoding='utf-8', errors='replace')
                        
                        if ext in ['.cpp', '.c', '.h', '.hpp']:
                            deps = extract_c_dependencies(content)
                        elif ext == '.cs':
                            deps = extract_csharp_dependencies(content)
                        elif ext == '.py':
                            deps = extract_python_dependencies(content)
                        elif ext in ['.js', '.ts']:
                            deps = extract_js_dependencies(content)
                        elif ext == '.lua':
                            deps = extract_lua_dependencies(content)
                        else:
                            deps = []
                        
                        # Filter out standard library imports for some languages
                        filtered_deps = []
                        for dep in deps:
                            # Keep relative paths (start with . or /) and module names
                            if dep.startswith(('.', '/')) or not dep.startswith('<'):
                                filtered_deps.append(dep)
                        
                        graph[rel_path] = filtered_deps
                    except Exception as e:
                        logger.warning(f"Could not read {full_path}: {e}")
                        graph[rel_path] = []
    except Exception as e:
        logger.error(e, exc_info=True)
    
    return graph

@mcp.tool()
def get_dependency_graph(
    path: Annotated[str, Field(description="The path inside the source tree to analyze for dependencies. If empty, basedir is used.")]
) -> str:
    """Build a dependency graph for all source code files in the given path. Returns JSON with the graph adjacency list and detected circular dependencies.
    
    The returned JSON contains:
    - `graph`: Adjacency list mapping each file to its direct dependencies
    - `circular_dependencies`: List of cycles found in the dependency graph
    
    Supported languages: C/C++, C#, Python, JavaScript, TypeScript, Lua
    """
    logger.info(f"Building dependency graph for path: {path}")
    
    try:
        analysis_path = sanitize_path(os.path.join(basedir, path)) if path else basedir
        
        # Define extensions to analyze
        extensions = ['.cpp', '.c', '.h', '.hpp', '.cs', '.py', '.js', '.ts', '.lua']
        
        # Build the graph
        graph = build_dependency_graph(analysis_path, extensions)
        
        # Detect circular dependencies
        circular_deps = detect_circular_dependencies(graph)
        
        result = {
            "graph": graph,
            "circular_dependencies": circular_deps
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(e, exc_info=True)
        return json.dumps({"error": str(e)})

def parse_cpp_class_declarations(content: str) -> list[dict]:
    """
    Parse C++ header file content to extract class/function declarations.
    
    Args:
        content (str): The header file content
    
    Returns:
        list[dict]: List of declarations with name, signature, and body
    """
    declarations = []
    
    # Pattern for class declarations
    class_pattern = r'class\s+(\w+)\s*(:\s*[^\{]*)?\s*\{([^}]*)\}'
    
    # Pattern for function declarations (not definitions)
    # Matches: return_type name(params); but not with {
    func_pattern = r'((?:virtual\s+)?(?:static\s+)?(?:const\s+)?(?:override\s+)?(?:\w+\s*::\s*)?[\w:<>,\*\&\s]+?)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:noexcept\s*)?\s*;'
    
    # Find class declarations
    for match in re.finditer(class_pattern, content, re.DOTALL):
        class_name = match.group(1)
        class_body = match.group(3)
        
        # Parse methods within class
        for func_match in re.finditer(func_pattern, class_body):
            return_type = func_match.group(1).strip()
            func_name = func_match.group(2)
            
            declarations.append({
                'type': 'method',
                'class': class_name,
                'name': func_name,
                'signature': f"{return_type} {func_name}(",
                'body': '',
                'is_declaration': True
            })
    
    # Find standalone function declarations
    for match in re.finditer(func_pattern, content):
        return_type = match.group(1).strip()
        func_name = match.group(2)
        
        declarations.append({
            'type': 'function',
            'class': None,
            'name': func_name,
            'signature': f"{return_type} {func_name}(",
            'body': '',
            'is_declaration': True
        })
    
    return declarations


def parse_cpp_definitions(content: str) -> list[dict]:
    """
    Parse C++ implementation file content to extract function definitions.
    
    Args:
        content (str): The implementation file content
    
    Returns:
        list[dict]: List of definitions with name, signature, and full body
    """
    definitions = []
    
    # Pattern for function definitions (with body)
    func_pattern = r'((?:virtual\s+)?(?:static\s+)?(?:const\s+)?(?:override\s+)?(?:\w+\s*::\s*)?[\w:<>,\*\&\s]+?)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:noexcept\s*)?\s*\{'
    
    # Find function definitions
    for match in re.finditer(func_pattern, content):
        return_type = match.group(1).strip()
        func_name = match.group(2)
        
        # Find the matching closing brace
        start_pos = match.end() - 1  # Position of opening brace
        brace_count = 1
        end_pos = start_pos
        
        for i in range(start_pos + 1, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i
                    break
        
        body = content[start_pos:end_pos + 1]
        
        definitions.append({
            'type': 'function',
            'name': func_name,
            'signature': f"{return_type} {func_name}(",
            'body': body,
            'is_declaration': False
        })
    
    return definitions


def extract_c_functions(content: str) -> tuple[list[str], list[str]]:
    """
    Extract C/C++ function declarations and definitions from file content.
    
    Args:
        content (str): The source code content to analyze.
    
    Returns:
        tuple[list[str], list[str]]: Tuple of (exported_functions, internal_functions)
    """
    exported = []
    internal = []
    
    # Pattern for function definitions with body
    func_pattern = r'((?:static\s+)?(?:inline\s+)?(?:const\s+)?(?:\w+\s*::\s*)?[\w:<>,\*\&\s]+?)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:noexcept\s*)?\s*\{'
    
    for match in re.finditer(func_pattern, content):
        return_type = match.group(1).strip()
        func_name = match.group(2)
        signature = f"{return_type} {func_name}("
        
        # Check if function is static (internal) or not
        if 'static' in return_type.lower() or return_type.startswith('static'):
            internal.append(signature + " end")
        else:
            exported.append(signature + " end")
    
    return exported, internal


def fuse_logical_unit(header_content: str, impl_content: str) -> str:
    """
    Fuse header and implementation content by inserting implementations into declarations.
    
    Args:
        header_content (str): The header file content
        impl_content (str): The implementation file content
    
    Returns:
        str: The fused virtual unit representation
    """
    # Parse declarations from header
    declarations = parse_cpp_class_declarations(header_content)
    
    # Parse definitions from implementation
    definitions = parse_cpp_definitions(impl_content)
    
    # Create a map of function names to their definitions
    def_map = {d['name']: d for d in definitions}
    
    # Replace declarations with their definitions
    fused_content = header_content
    
    for decl in declarations:
        if decl['name'] in def_map:
            definition = def_map[decl['name']]
            
            # Find and replace the declaration with the definition
            pattern = re.escape(decl['signature']) + r"[^;]*;"
            replacement = definition['body']
            
            fused_content = re.sub(pattern, replacement, fused_content, count=1)
    
    return fused_content


def find_unit_files(unit_name: str) -> list[str]:
    """
    Find all files related to a logical unit.
    
    Args:
        unit_name (str): The name of the unit (e.g., "user")
    
    Returns:
        list[str]: List of file paths related to the unit
    """
    import os
    
    # Common extensions for C++ units
    extensions = ['.h', '.hpp', '.hxx', '.c', '.cpp', '.cxx']
    
    files = []
    
    try:
        # Search in basedir for files matching the unit name
        for root, _, filenames in os.walk(basedir):
            for filename in filenames:
                base, ext = os.path.splitext(filename)
                if base == unit_name and ext in extensions:
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, basedir).replace("\\", "/")
                    files.append(rel_path)
    except Exception as e:
        logger.error(f"Error finding unit files for {unit_name}: {e}")
    
    return sorted(files)


@mcp.tool()
def get_logical_unit(
    unit_name: Annotated[str, Field(description="The name of the logical unit to retrieve. For example, 'user' would combine user.h and user.cpp into a single virtual representation.")]
) -> str:
    """Retrieve a fusionized virtual representation of a logical unit. If a unit is split across multiple files (e.g., header and implementation), this function combines them into a single cohesive view where implementations are inserted into their corresponding declarations."""
    logger.info(f"get_logical_unit called with unit_name: {unit_name}")
    
    try:
        # Find all files related to this unit
        unit_files = find_unit_files(unit_name)
        
        if not unit_files:
            return f"No files found for unit '{unit_name}'"
        
        # For C++ units, combine header and implementation
        header_content = ""
        impl_content = ""
        
        for file_path in unit_files:
            full_path = sanitize_path(os.path.join(basedir, file_path))
            
            try:
                content = Path(full_path).read_text(encoding='utf-8', errors='replace')
                
                ext = os.path.splitext(file_path)[1]
                
                if ext in ['.h', '.hpp', '.hxx']:
                    header_content = content
                elif ext in ['.c', '.cpp', '.cxx']:
                    impl_content = content
            except Exception as e:
                logger.warning(f"Could not read file {full_path}: {e}")
        
        # If we have both header and implementation, fuse them
        if header_content and impl_content:
            return fuse_logical_unit(header_content, impl_content)
        elif header_content:
            return header_content
        elif impl_content:
            return impl_content
        else:
            # Return all files concatenated if no fusion is possible
            return "\n\n".join([Path(sanitize_path(os.path.join(basedir, f))).read_text(encoding='utf-8', errors='replace') for f in unit_files])
    
    except Exception as e:
        logger.error(e, exc_info=True)
        return f"Error retrieving logical unit '{unit_name}': {str(e)}"


def extract_csharp_functions(content: str) -> tuple[list[str], list[str]]:
    """
    Extract C# function declarations and definitions from file content.
    
    Args:
        content (str): The source code content to analyze.
    
    Returns:
        tuple[list[str], list[str]]: Tuple of (exported_functions, internal_functions)
    """
    exported = []
    internal = []
    
    # Pattern for method/function definitions
    func_pattern = r'(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?(?:override\s+)?(?:virtual\s+)?(?:new\s+)?(?:\w+(?:<[^>]+>)?\s+)?(\w+)\s*\([^)]*\)\s*\{'
    
    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)
        
        # Determine if internal based on access modifier
        full_match = match.group(0)
        if 'private' in full_match or 'internal' in full_match:
            internal.append(f"function {func_name}() end")
        else:
            exported.append(f"function {func_name}() end")
    
    return exported, internal


def extract_python_functions(content: str) -> tuple[list[str], list[str]]:
    """
    Extract Python function definitions from file content.
    
    Args:
        content (str): The source code content to analyze.
    
    Returns:
        tuple[list[str], list[str]]: Tuple of (exported_functions, internal_functions)
    """
    exported = []
    internal = []
    
    # Pattern for function definitions
    func_pattern = r'def\s+(\w+)\s*\([^)]*\)\s*:'
    
    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)
        
        # Internal functions start with underscore
        if func_name.startswith('_'):
            internal.append(f"def {func_name}() end")
        else:
            exported.append(f"def {func_name}() end")
    
    return exported, internal


def extract_js_functions(content: str) -> tuple[list[str], list[str]]:
    """
    Extract JavaScript/TypeScript function declarations and definitions from file content.
    
    Args:
        content (str): The source code content to analyze.
    
    Returns:
        tuple[list[str], list[str]]: Tuple of (exported_functions, internal_functions)
    """
    exported = []
    internal = []
    
    # Patterns for different function types
    patterns = [
        r'function\s+(\w+)\s*\([^)]*\)\s*\{',  # function name()
        r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function\s*\([^)]*\)\s*\{',  # const name = function()
        r'(?:export\s+)?const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*',  # const name = () =>
        r'(?:export\s+)?function\s*(\w+)\s*\([^)]*\)\s*:',  # TypeScript function declaration
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, content):
            func_name = match.group(1)
            
            # Exported functions have 'export' keyword
            full_match = match.group(0)
            if 'export' in full_match:
                exported.append(f"function {func_name}() end")
            else:
                internal.append(f"function {func_name}() end")
    
    return exported, internal


def extract_lua_functions(content: str) -> tuple[list[str], list[str]]:
    """
    Extract Lua function definitions from file content.
    
    Args:
        content (str): The source code content to analyze.
    
    Returns:
        tuple[list[str], list[str]]: Tuple of (exported_functions, internal_functions)
    """
    exported = []
    internal = []
    
    # Pattern for function definitions
    func_pattern = r'(?:local\s+)?function\s+(\w+)\s*\([^)]*\)'
    
    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)
        
        # Local functions are internal
        if match.group(0).startswith('local'):
            internal.append(f"local function {func_name}() end")
        else:
            exported.append(f"function {func_name}() end")
    
    return exported, internal


@mcp.tool()
def get_source_signature(
    path: Annotated[str, Field(description="The path to the source file (relative to basedir) for which to generate a source signature. Supported languages: C, C++, C#, Python, JavaScript, TypeScript, Lua")]
) -> str:
    """Generate a compact source signature for a given file in markdown format. Returns module dependencies and function signatures (exported and internal/local functions).
    
    Supported languages: C, C++, C#, Python, JavaScript, TypeScript, Lua
    
    Example output:
    ```
    # SOURCE SIGNATURE FOR: src/network/router.lua
    
    -- Module Dependencies:
    -- [Requires: "utils.logger", "config.routes"]
    
    -- Exported Functions:
    function Router.new(config) end
    function Router:register_route(path, handler) end
    function Router:dispatch(request) end
    
    -- Internal/Local Functions:
    local function parse_url(url) end
    local function validate_headers(headers) end
    ```
    """
    return get_source_signature_impl(path)


def get_source_signature_impl(path: str) -> str:
    """Internal implementation of source signature generation."""
    full_path = sanitize_path(os.path.join(basedir, path))
    
    if not os.path.isfile(full_path):
        return f"# SOURCE SIGNATURE FOR: {path}\n\nError: File not found"
    
    try:
        content = Path(full_path).read_text(encoding='utf-8', errors='replace')
        _, ext = os.path.splitext(path)
        
        # Get dependencies based on file extension
        dependencies = []
        if ext in ['.c', '.h', '.cpp', '.hpp']:
            dependencies = extract_c_dependencies(content)
        elif ext == '.cs':
            dependencies = extract_csharp_dependencies(content)
        elif ext == '.py':
            dependencies = extract_python_dependencies(content)
        elif ext in ['.js', '.ts']:
            dependencies = extract_js_dependencies(content)
        elif ext == '.lua':
            dependencies = extract_lua_dependencies(content)
        
        # Get functions based on file extension
        exported_funcs, internal_funcs = [], []
        if ext in ['.c', '.h', '.cpp', '.hpp']:
            exported_funcs, internal_funcs = extract_c_functions(content)
        elif ext == '.cs':
            exported_funcs, internal_funcs = extract_csharp_functions(content)
        elif ext == '.py':
            exported_funcs, internal_funcs = extract_python_functions(content)
        elif ext in ['.js', '.ts']:
            exported_funcs, internal_funcs = extract_js_functions(content)
        elif ext == '.lua':
            exported_funcs, internal_funcs = extract_lua_functions(content)
        
        # Build markdown output
        lines = []
        lines.append(f"# SOURCE SIGNATURE FOR: {path}")
        lines.append("")
        
        # Dependencies section
        if dependencies:
            lines.append("-- Module Dependencies:")
            for dep in sorted(set(dependencies)):
                lines.append(f"-- [Requires: \"{dep}\"]")
            lines.append("")
        
        # Exported functions section
        if exported_funcs:
            lines.append("-- Exported Functions:")
            for func in exported_funcs:
                lines.append(func)
            lines.append("")
        
        # Internal functions section
        if internal_funcs:
            lines.append("-- Internal/Local Functions:")
            for func in internal_funcs:
                lines.append(func)
        
        return "\n".join(lines)
    
    except Exception as e:
        logger.error(f"Error generating source signature for {path}: {e}")
        return f"# SOURCE SIGNATURE FOR: {path}\n\nError: {str(e)}"


### Main method to start the server
def main():
    """Start the server and handle incoming requests. This method will initialize the global basedir variable with the value provided as runtime argument, then start the FastMCP server.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if os.access(os.getcwd(), os.W_OK):
        logging.basicConfig(
            filename="server.log",
            encoding="utf-8",
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-d %H:%M:%S'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-d %H:%M:%S'
        )

    logging.getLogger().setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser(description="MCP Server for a source tree")
    parser.add_argument("--base-dir", type=str, help="A full qualified path to a source code directory")
    parser.add_argument("--mcp-host", type=str, default="127.0.0.1", help="Host name to listen, default: 127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=8082, help="Port to listen on, default: 8082")
    parser.add_argument("--transport", type=str, default="stdio", choices=["stdio", "sse"],
                        help="Which protocol to use, default: stdio")
    parser.add_argument("--with-companion", type=bool, default=False, help="If set, the companion LLM will be used to answer questions and transpile code.")
    

    args = parser.parse_args()
    
    global basedir
    basedir = os.path.realpath(args.base_dir).replace("\\", "/")
    
    if args.with_companion:
        output = companion_llm(
            prompt="Tell us when you are ready to support us",
            max_tokens=1024,
            temperature=0.2,
            stop=["\nQuestion:"],
            echo=False
        )
        
        logger.info(f"Companion LLM answered: {output['choices'][0]['text']}")

        ask_companion.__name__ = "sourcetreemcp_ask_companion"
        transpile_code.__name__ = "sourcetreemcp_transpile_code"
        mcp.add_tool(ask_companion)
        mcp.add_tool(transpile_code)
    
    if args.transport == "sse":
        logger.info(f"Starting Source Tree MCP server")
        mcp.run(transport="sse", host=args.mcp_host, port=args.mcp_port)
    else:
        mcp.run(host=args.mcp_host, port=args.mcp_port)

# Execute the main method if started as an application
if __name__ == "__main__":
    main()