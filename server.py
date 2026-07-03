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

def get_files_from_git_tree(path: str, file_extension:str = "") -> list[str]:
    """
    Get a list of all files in the given Git tree.

    Args:
        path (str): The path to the Git tree.

    Returns:
        list[str]: A list of file paths or an empty list in case nothing found or error.
    """
    try:
        repo = Repo(os.path.abspath(basedir))
        
        entries:list[str]  = []
        
        for entry in repo.commit().tree.traverse():
            epath = str(entry.path) # type: ignore
            if len(path) == 0 or epath.startswith(path) and (file_extension == "" or epath.endswith(file_extension)):
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
    ]
) -> list[str]:
    """Retrieve a list of file names in *folder_path*, which is, if provided, a directory inside the global runtime parameter basedir provided at start of server, that have the specified *file_extension*. If no path is provided, all files in the global basedir are listed."""
    
    files:list[str] = []
    try:
        if is_git_repo():
            files = get_files_from_git_tree(path, file_extension)
        else:        
            dir = sanitize_path(os.path.join(basedir, path))
            
            logger.info(f"About to retrieve files from {dir}")
            
            entries = []
            
            if os.path.exists(dir):
                entries = os.listdir(dir)
            
            if not file_extension == "":
                logger.info(f"Filtering files with extension {file_extension}")
                files = [f for f in entries if f.endswith(file_extension) and os.path.isfile(os.path.join(dir, f))]
            else:
                files = [f for f in entries if os.path.isfile(os.path.join(dir, f))]
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
        return ""

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
        logger.error(e, exc_info=True)

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
    content: Annotated[str, Field(description="The content to write to the file.")]
) -> str:
    """Write the given content to a file at the specified path. Creates parent directories if necessary. The path is relative to the global runtime parameter basedir provided at start of server. The returned string indicates whether the file was successfully written or not."""
    try:
        path = sanitize_path(os.path.join(basedir, path))
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            logger.info(f"Creating directory {parent_dir}.")
            os.makedirs(parent_dir)
        with open(path, 'w') as f:
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